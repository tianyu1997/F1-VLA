import os
import logging
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch

import transformers
from transformers import set_seed, HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from f1_vla.src.models.configuration_f1 import F1Config
from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.utils.utils import (
    load_ckpt,
    clean_overrides,
    save_training_args, 
    set_policy_config,
    set_camera_config,
)
from f1_vla.src.processors.data_processors.data_config import create_data_config
from f1_vla.src.processors.data_processors.data_loader import create_data, create_mekvm_data, CollateFn
from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainer, PolicyTrainingArguments
from f1_vla.src.processors.train_processors.optimizer_scheduler import create_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger()


def main(args: argparse.Namespace, overrides: list):
    #########################################################
    # Set the policy config and training config
    #########################################################
    logger.info(f"Using transformers version: {transformers.__version__}")
    logger.info(f"Using torch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        
    config = OmegaConf.load(Path(args.config_file))

    # Load separate dataset stats config if it exists
    # This allows keeping large normalization vectors out of main config
    stats_path = Path("f1_vla/config/dataset_stats.yaml")
    if stats_path.exists():
        logger.info(f"Loading dataset stats from {stats_path}")
        stats_cfg = OmegaConf.load(stats_path)
        config = OmegaConf.merge(config, stats_cfg)

    override_cfg = OmegaConf.from_dotlist(clean_overrides(overrides))
    config = OmegaConf.merge(config, override_cfg)
 
    policy_config = F1Config.from_pretrained(f"{config.policy.ckpt_path}")
    policy_config = set_policy_config(policy_config, config.policy)
    policy_config = set_camera_config(policy_config, config.exp)
    
    # Set memory config from exp.use_memory and exp.memory_config
    use_memory = config.exp.get('use_memory', False)
    policy_config.use_memory = use_memory
    if use_memory and hasattr(config.exp, 'memory_config') and config.exp.memory_config:
        from f1_vla.src.models.configuration_f1 import DictWithAttrAccess
        mem_cfg = config.exp.memory_config
        # Use DictWithAttrAccess to match F1Config's expected format
        policy_config.memory_config = DictWithAttrAccess({
            "memory_len": int(mem_cfg.get('memory_len', 4)),
            # k_bptt: number of gradient frames, also window stride
            "k_bptt": int(mem_cfg.get('k_bptt', 4)),
            "init_std": float(mem_cfg.get('init_std', 0.02)),
            "tokenizer_max_length": int(mem_cfg.get('tokenizer_max_length', 512)),
        })
        logger.info(f"Memory enabled: memory_len={policy_config.memory_config.memory_len}, k_bptt={policy_config.memory_config.k_bptt}")
    
    # Set VAE config from exp.vae_config (pixel_loss_weight, etc.)
    vae_cfg = getattr(config.exp, "vae_config", None)
    if vae_cfg:
        if hasattr(vae_cfg, 'pixel_loss_weight'):
            policy_config.pixel_loss_weight = vae_cfg.pixel_loss_weight
        if hasattr(vae_cfg, 'pixel_loss_type'):
            policy_config.pixel_loss_type = vae_cfg.pixel_loss_type
        if hasattr(vae_cfg, 'freeze_encoder'):
            policy_config.vae_freeze_encoder = vae_cfg.freeze_encoder
        if hasattr(vae_cfg, 'test_mode'):
            policy_config.vae_test_mode = vae_cfg.test_mode

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
    # Disable default handler to prevent duplicate logs (since we configured root logger)
    transformers.utils.logging.disable_default_handler()
    # transformers.utils.logging.enable_explicit_format()

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
        
        # Get distributed training info FIRST
        rank = int(os.environ.get('LOCAL_RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Create dataset with distributed support - each rank loads only its portion
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
        
        # Create sequential batch sampler (no longer needs rank/world_size since dataset is already sharded)
        # Window design: window_length = n_obs_img_steps + k_bptt, stride = k_bptt
        # - First n_obs_img_steps frames: history context for memory warmup (loss detached)
        # - Last k_bptt frames: compute gradients
        n_obs = config.dataset.get('n_obs_img_steps', 4)
        k_bptt = config.exp.memory_config.get('k_bptt', 4) if hasattr(config.exp, 'memory_config') else 4
        window_length = n_obs + k_bptt  # e.g., 4 + 4 = 8 frames per window
        
        sequential_sampler = SequentialBatchSampler(
            dataset=training_dataset,
            batch_size=training_args.per_device_train_batch_size,
            chunk_size=window_length,  # Total window length
            stride=k_bptt,             # Window moves by k_bptt each step
            shuffle_episodes=True,
            drop_last=False,
            rank=0,  # Each dataset is already a shard, so sampler treats it as rank 0
            world_size=1,
        )
        logger.info(f"Window config: n_obs={n_obs}, k_bptt={k_bptt}, window_length={window_length}, stride={k_bptt}")
        logger.info(f"Using SEQUENTIAL data loading for memory-based training (rank={rank}, world_size={world_size})")
    elif use_mekvm_format:
        # Use ME_KVM data format (standard random loading)
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
        # Use LeRobot data format
        sequential_sampler = None
        data_config = create_data_config(config.dataset, policy_config, config.exp)
        (
            training_dataset, 
            image_transforms, 
            training_ds_sample_weights, 
            cur_n_obs_img_steps, 
            cur_n_pred_img_steps
        ) = create_data(
            policy_config=policy_config, 
            dataset_config=data_config, 
            training_args=training_args, 
            stage=config.exp.stage,
            max_eval_samples=config.exp.max_eval_samples,
        )
        collate_fn = CollateFn(policy_config.max_state_dim, policy_config.max_action_dim)

    logger.info(f"Training dataset:\n{training_dataset}")
    logger.info(f"len(training_dataset): {len(training_dataset)}")

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
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    #########################################################
    # Create model
    #########################################################
    logger.info("Creating model")
    logger.info(f"Policy config Pretrained path: {policy_config.pretrained_path}")
    kwargs = {"config": policy_config}

    kwargs["pretrained_name_or_path"] = policy_config.pretrained_path
    kwargs["training_args"] = training_args

    # Verify if we are resuming from a checkpoint
    is_resuming = training_args.resume_from_checkpoint is not None or last_checkpoint is not None

    if policy_config.pretrained_path and not args.debug:
        if is_resuming:
            logger.info("Resuming training detected. Initializing model directly (skipping pi0 load).")
            policy = F1_VLA(**kwargs)
            logger.info(f"Skipping base pretrained weights load: {config.exp.load_ckpt if hasattr(config.exp, 'load_ckpt') else 'N/A'}")
        else:
            logger.info("Calling F1_VLA.from_pretrained...")
            policy = F1_VLA.from_pretrained(**kwargs)
            logger.info("F1_VLA.from_pretrained returned.")
            
            policy = load_ckpt(policy, config)
    else:
        policy = F1_VLA(**kwargs)

    optimizer = create_optimizer(policy, training_args)

    #########################################################
    # Create trainer
    #########################################################
    # Get number of episodes for progress display
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
        eval_dataset=training_dataset,  # Use same dataset for eval (can be changed)
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
        
        # NOTE: Removed forced ignore_data_skip=True for SequentialBatchSampler
        # We rely on correct state restoration and HF's skipping mechanism.
        # If sequential sampler is deterministic per epoch, skipping is safe.

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept both --config-file (original) and --config (used by train.sh)
    parser.add_argument("--config-file", "--config", dest="config_file", type=str, required=True,
                        help="Path to training config yaml")
    parser.add_argument('--debug', action='store_true', help='to enable debug mode')
    args, unknown = parser.parse_known_args()
    main(args, overrides=unknown)