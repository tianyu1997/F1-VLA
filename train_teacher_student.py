"""
Teacher-Student Distillation Training Script for F1-VLA World Model.

This script trains a student policy to predict next frames using only wrist camera,
while distilling knowledge from a teacher that has access to both head and wrist cameras.

Usage:
    # Teacher-Student with memory distillation:
    python train_teacher_student.py --config-file f1_vla/config/teacher_student_config.yaml
    
    # Control group (student only, no distillation):
    python train_teacher_student.py --config-file f1_vla/config/student_only_config.yaml
"""

import os
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf

import transformers
from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from f1_vla.src.models.configuration_f1 import F1Config
from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.policies.teacher_student_policy import TeacherStudentPolicy, StudentOnlyPolicy
from f1_vla.src.utils.utils import (
    load_ckpt,
    clean_overrides,
    save_training_args, 
    set_policy_config,
    set_camera_config,
)
from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainer, PolicyTrainingArguments

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger()


def create_teacher_student_optimizer(policy, training_args):
    """Create optimizer for teacher-student policy (only student parameters)."""
    import torch
    from torch.optim import AdamW
    
    # Get trainable parameters from student
    trainable_params = []
    
    if hasattr(policy, 'get_optim_params'):
        trainable_params = policy.get_optim_params()
    else:
        trainable_params = [p for p in policy.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        trainable_params,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
    )
    
    return optimizer


def main(args, overrides):
    #########################################################
    # Set the policy config and training config
    #########################################################
    config = OmegaConf.load(Path(args.config_file))
    override_cfg = OmegaConf.from_dotlist(clean_overrides(overrides))
    config = OmegaConf.merge(config, override_cfg)
 
    policy_config = F1Config.from_pretrained(f"{config.policy.ckpt_path}")
    policy_config = set_policy_config(policy_config, config.policy)
    policy_config = set_camera_config(policy_config, config.exp)

    parser_training_args = HfArgumentParser((PolicyTrainingArguments))
    training_args = OmegaConf.to_container(config.exp.training_args, resolve=True)
    training_args = parser_training_args.parse_dict(training_args)[0]

    #########################################################
    # Save training args
    #########################################################
    worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
    local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))
    if worker_idx == 0 and local_rank_idx in [-1, 0]:
        save_training_args(training_args, policy_config, config)
        logger.info(f"saved training args on worker {worker_idx}, local rank {local_rank_idx}") 

    #########################################################
    # Log on each process summary
    #########################################################
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    set_seed(training_args.seed)
    logger.info(f"Training config: {args}")
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    #########################################################
    # Create dataset
    #########################################################
    use_mekvm_format = config.dataset.get('use_mekvm_format', False)
    use_memory = config.exp.get('use_memory', False)
    
    # Determine data loading mode
    if use_mekvm_format and use_memory:
        # Sequential data loading for memory-based training
        from f1_vla.src.processors.data_processors.sequential_dataset import (
            create_sequential_mekvm_data, SequentialCollateFn, SequentialBatchSampler
        )
        
        # Get distributed training info
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Create dataset
        (
            training_dataset,
            image_transforms,
            training_ds_sample_weights,
            cur_n_obs_img_steps,
            cur_n_pred_img_steps
        ) = create_sequential_mekvm_data(
            policy_config=policy_config,
            dataset_config=config.dataset,
            training_args=training_args,
            stage=config.exp.stage,
            rank=rank,
            world_size=world_size,
        )
        collate_fn = SequentialCollateFn(policy_config.max_state_dim, policy_config.max_action_dim)
        
        # Create sequential batch sampler
        sequential_sampler = SequentialBatchSampler(
            dataset=training_dataset,
            batch_size=training_args.per_device_train_batch_size,
            shuffle_episodes=True,
            drop_last=False,
            rank=0,
            world_size=1,
        )
        logger.info(f"Using SEQUENTIAL data loading for memory-based training")
    elif use_mekvm_format:
        from f1_vla.src.processors.data_processors.data_loader import create_mekvm_data, CollateFn
        from f1_vla.src.processors.data_processors.me_kvm_dataset import MEKVMCollateFn
        sequential_sampler = None
        (
            training_dataset,
            image_transforms,
            training_ds_sample_weights,
            cur_n_obs_img_steps,
            cur_n_pred_img_steps
        ) = create_mekvm_data(
            policy_config=policy_config,
            dataset_config=config.dataset,
            training_args=training_args,
            stage=config.exp.stage,
        )
        collate_fn = MEKVMCollateFn(policy_config.max_state_dim, policy_config.max_action_dim)
    else:
        raise ValueError("Teacher-Student training requires ME_KVM format dataset")

    logger.info(f"Training dataset:\n{training_dataset}")
    logger.info(f"len(training_dataset): {len(training_dataset)}")

    #########################################################
    # Create Teacher-Student Policy
    #########################################################
    ts_config = config.exp.get('teacher_student_config', {})
    use_teacher_student = ts_config.get('use_teacher_student', False)
    use_student_only = ts_config.get('use_student_only', False)
    
    logger.info("Creating Teacher-Student model")
    
    if use_teacher_student:
        # Teacher-Student with memory distillation
        teacher_ckpt = config.exp.get('teacher_ckpt', None)
        student_ckpt = config.exp.get('student_ckpt', None)
        memory_loss_weight = ts_config.get('memory_loss_weight', 0.5)
        use_memory_distillation = ts_config.get('use_memory_distillation', True)
        
        logger.info(f"Creating TeacherStudentPolicy:")
        logger.info(f"  teacher_ckpt: {teacher_ckpt}")
        logger.info(f"  student_ckpt: {student_ckpt}")
        logger.info(f"  memory_loss_weight: {memory_loss_weight}")
        logger.info(f"  use_memory_distillation: {use_memory_distillation}")
        
        policy = TeacherStudentPolicy(
            config=policy_config,
            teacher_ckpt=teacher_ckpt,
            student_ckpt=student_ckpt,
            memory_loss_weight=memory_loss_weight,
            use_memory_distillation=use_memory_distillation,
            training_args=training_args,
        )
        
    elif use_student_only:
        # Control group: Student only, no teacher
        student_ckpt = config.exp.get('student_ckpt', None)
        logger.info("Creating StudentOnlyPolicy (control group)")
        logger.info(f"  student_ckpt: {student_ckpt}")
        policy = StudentOnlyPolicy(
            config=policy_config,
            student_ckpt=student_ckpt,
            training_args=training_args,
        )
        
    else:
        raise ValueError("Must set either use_teacher_student=True or use_student_only=True in config")

    # Create optimizer
    optimizer = create_teacher_student_optimizer(policy, training_args)

    #########################################################
    # Resume from checkpoint
    #########################################################   
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
            )

    #########################################################
    # Create trainer
    #########################################################
    num_episodes = 0
    if hasattr(training_dataset, 'get_num_episodes'):
        num_episodes = training_dataset.get_num_episodes()
    elif hasattr(training_dataset, 'num_episodes'):
        num_episodes = training_dataset.num_episodes
    
    # Get episode-based settings from config
    logging_episodes = config.exp.training_args.get("logging_episodes", 100)
    save_episodes = config.exp.training_args.get("save_episodes", 500)
    eval_episodes = config.exp.training_args.get("eval_episodes", 500)
    
    trainer = PolicyTrainer(
        policy=policy,
        args=training_args,
        train_dataset=training_dataset,
        optimizers=(optimizer, None),
        data_collator=collate_fn,
        image_transforms=image_transforms,
        use_world_model=policy_config.use_world_model,
        cur_n_obs_img_steps=cur_n_obs_img_steps,
        cur_n_pred_img_steps=cur_n_pred_img_steps,
        training_ds_sample_weights=training_ds_sample_weights,
        sequential_sampler=sequential_sampler if use_memory else None,
        use_memory=use_memory,
        num_episodes=num_episodes,
        logging_episodes=logging_episodes,
        save_episodes=save_episodes,
        eval_episodes=eval_episodes,
        eval_dataset=training_dataset,
    )

    #########################################################   
    # Training
    #########################################################
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(f"Training from checkpoint: {checkpoint}")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    args, overrides = parser.parse_known_args()

    main(args, overrides)
