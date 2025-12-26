import os
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple

import torch

from f1_vla.src.utils.utils import LargeScaleWeightedRandomSampler
from f1_vla.src.policies.f1_policy import F1_VLA

from lerobot.policies.pretrained import PreTrainedPolicy
from transformers import Trainer, __version__, PretrainedConfig
from transformers.trainer import (
    logger, 
    FSDP_MODEL_NAME, 
    TRAINING_ARGS_NAME, 
    is_peft_available, 
    _get_fsdp_ckpt_kwargs,
    _is_peft_model,
)
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerCallback
from transformers.modeling_utils import load_sharded_checkpoint
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
)

if is_accelerate_available():
    from accelerate.utils import load_fsdp_model


@dataclass
class PolicyTrainingArguments(TrainingArguments):
    train_dir: str | None = None
    eval_dir: str | None = None
    num_eval_episodes: int = 50
    stage: str = "stage3_finetune_vla"
    language_tokenizer_path: str | None = None

    freeze_vision_encoder: bool = False
    freeze_gen_expert: bool = False
    train_act_expert_only: bool = False
    train_gen_expert_only: bool = False
    train_state_proj: bool = True

    gen_out_loss_ratio: float = 0.0

    resize_imgs_with_padding: Tuple[int, int] = (224, 224)

    image_transforms_enabled: bool = True
    image_transforms_max_num_transforms: int = 3
    image_transforms_random_order: bool = True
    image_transforms_type: List[str] = field(
        default_factory=lambda: ["brightness", "contrast", "saturation", "random_crop", "random_rotation"]
    )

    und_expert_lr: float = 0.0
    act_expert_lr: float = 0.0
    gen_expert_lr: float = 0.0
    vision_encoder_lr: float = 0.0
    
    # Episode-based logging and saving
    logging_episodes: int = 10  # Log every N episodes
    save_episodes: int = 100    # Save every N episodes
    eval_episodes: int = 50     # Eval every N episodes (0 to disable)

    def __post_init__(self):
        super().__post_init__()
        random.seed(self.seed)


class EpisodeProgressCallback(TrainerCallback):
    """Custom progress callback that displays progress by epoch/episode with real-time loss/acc.
    
    Also handles episode-based logging, saving, and evaluation.
    """
    
    def __init__(self, num_episodes: int = 0, total_steps: int = 0, batch_size: int = 4, num_epochs: int = 1,
                 logging_episodes: int = 10, save_episodes: int = 100, eval_episodes: int = 50):
        self.num_episodes = num_episodes
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_episodes = logging_episodes
        self.save_episodes = save_episodes
        self.eval_episodes = eval_episodes
        
        # Number of episode batches (groups of batch_size episodes)
        self.num_episode_batches = (num_episodes + batch_size - 1) // batch_size
        self.pbar = None
        self.current_epoch = 0
        self.current_episode_batch = 0
        self.seen_episodes = set()  # Track all seen episode indices
        self.epoch_episode_count = 0  # Episodes seen in current epoch
        self.total_episode_count = 0  # Total episodes across all epochs
        self.last_log_episode = 0
        self.last_save_episode = 0
        self.last_eval_episode = 0
        
        # Real-time metrics (updated every step from compute_loss)
        self.current_loss = 0.0
        self.current_wm_loss = 0.0
        self.current_wm_acc = 0.0
        self.current_action_loss = 0.0
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress bar at training start."""
        from tqdm import tqdm
        if state.is_local_process_zero:
            # Show epoch and episode progress
            desc = f"Epoch 1/{self.num_epochs} [Ep: 0/{self.num_episodes}]"
            # Use num_episodes as total (per epoch)
            self.pbar = tqdm(
                total=self.num_episodes,
                desc=desc,
                dynamic_ncols=True,
                leave=True,
                unit="ep",
            )
            self.pbar.set_postfix({
                'loss': 0,
                'wm_loss': 0,
                'wm_acc': '0.00%',
            })
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset episode counter at epoch start."""
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self.seen_episodes.clear()
        self.epoch_episode_count = 0
        if self.pbar is not None and state.is_local_process_zero:
            self.pbar.reset()
            self.pbar.set_description(f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Ep: 0/{self.num_episodes}]")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bar after each step."""
        if self.pbar is not None and state.is_local_process_zero:
            # Update display with real-time metrics (no step update, only episode-based)
            self.pbar.set_postfix({
                'loss': f"{self.current_loss:.4f}",
                'wm_loss': f"{self.current_wm_loss:.4f}",
                'wm_acc': f"{self.current_wm_acc:.2%}",
            })
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update metrics display when logging (for logged metrics)."""
        if self.pbar is not None and state.is_local_process_zero and logs is not None:
            postfix = {}
            if 'loss' in logs:
                postfix['loss'] = f"{logs['loss']:.4f}"
            if 'wm_out_loss' in logs:
                postfix['wm_loss'] = f"{logs['wm_out_loss']:.4f}"
            if 'wm_acc_mean' in logs:
                postfix['wm_acc'] = f"{logs['wm_acc_mean']:.2%}"
            if 'action_loss' in logs:
                postfix['act_loss'] = f"{logs['action_loss']:.4f}"
            if postfix:
                self.pbar.set_postfix(postfix)
    
    def update_metrics(self, loss: float, wm_loss: float = 0.0, wm_acc: float = 0.0, action_loss: float = 0.0):
        """Update real-time metrics from compute_loss."""
        self.current_loss = loss
        self.current_wm_loss = wm_loss
        self.current_wm_acc = wm_acc
        self.current_action_loss = action_loss
    
    def update_episode(self, episode_indices: list):
        """Update episode counter with all episode indices in the current batch."""
        # Track all unique episodes seen in this epoch
        new_episodes_count = 0
        for ep_idx in episode_indices:
            if ep_idx not in self.seen_episodes:
                self.seen_episodes.add(ep_idx)
                new_episodes_count += 1
        
        if new_episodes_count > 0:
            self.epoch_episode_count = len(self.seen_episodes)
            self.total_episode_count += new_episodes_count
            
            if self.pbar is not None:
                # Update progress bar by number of new episodes
                self.pbar.update(new_episodes_count)
                self.pbar.set_description(
                    f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Ep: {self.epoch_episode_count}/{self.num_episodes}]"
                )
        
        return new_episodes_count
    
    def should_log(self) -> bool:
        """Check if should log based on episode count."""
        if self.logging_episodes <= 0:
            return False
        episodes_since_log = self.total_episode_count - self.last_log_episode
        return episodes_since_log >= self.logging_episodes
    
    def should_save(self) -> bool:
        """Check if should save based on episode count."""
        if self.save_episodes <= 0:
            return False
        episodes_since_save = self.total_episode_count - self.last_save_episode
        return episodes_since_save >= self.save_episodes
    
    def should_eval(self) -> bool:
        """Check if should eval based on episode count."""
        if self.eval_episodes <= 0:
            return False
        episodes_since_eval = self.total_episode_count - self.last_eval_episode
        return episodes_since_eval >= self.eval_episodes
    
    def mark_logged(self):
        self.last_log_episode = self.total_episode_count
    
    def mark_saved(self):
        self.last_save_episode = self.total_episode_count
    
    def mark_evaled(self):
        self.last_eval_episode = self.total_episode_count
    
    def on_train_end(self, args, state, control, **kwargs):
        """Close progress bar at training end."""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None


class PolicyTrainerCallback(TrainerCallback):
    policy: None
    image_transforms: None
    def __init__(self, policy, image_transforms):
        self.policy = policy
        self.image_transforms = image_transforms

    def on_train_begin(self, args, state, control, **kwargs):
        """ move the normalize_inputs and normalize_targets to the device """
        if self.image_transforms is not None:
            self.image_transforms.to(args.device)

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset memory at the start of each epoch for BPTT."""
        if hasattr(self.policy, 'model') and hasattr(self.policy.model, 'memory_manager'):
            if self.policy.model.memory_manager is not None:
                self.policy.model.memory_manager.on_epoch_start()


class PolicyTrainer(Trainer):
    def __init__(
        self, 
        policy: Union[PreTrainedPolicy, F1_VLA], 
        image_transforms=None, 
        use_world_model=True,
        cur_n_obs_img_steps=None, 
        cur_n_pred_img_steps=None, 
        training_ds_sample_weights=None,
        sequential_sampler=None,  # For memory-based sequential training
        use_memory=False,
        num_episodes=0,  # Total number of episodes for progress display
        logging_episodes=10,  # Log every N episodes
        save_episodes=100,  # Save every N episodes  
        eval_episodes=50,  # Eval every N episodes
        eval_dataset=None,  # Dataset for evaluation
        *args, 
        **kwargs
    ):
        self.policy = policy
        self.image_transforms = image_transforms
        self.use_world_model = use_world_model
        self.use_memory = use_memory
        self.sequential_sampler = sequential_sampler
        self.num_episodes = num_episodes
        self.eval_dataset = eval_dataset
        # TODO: make this configurable
        self.pred_img_keys = ["observation.images.image0_history"]
        assert len(self.pred_img_keys) == 1, "Only one image key is supported for now"

        self.cur_n_obs_img_steps = cur_n_obs_img_steps
        self.cur_n_pred_img_steps = cur_n_pred_img_steps

        # Create callbacks
        move_callbacks = [PolicyTrainerCallback(policy=policy, image_transforms=image_transforms)]
        
        # Add episode progress callback if using memory
        training_args = kwargs.get('args')
        total_steps = training_args.max_steps if training_args else 0
        batch_size = training_args.per_device_train_batch_size if training_args else 4
        num_epochs = int(training_args.num_train_epochs) if training_args else 1
        self.episode_progress_callback = EpisodeProgressCallback(
            num_episodes=num_episodes,
            total_steps=total_steps,
            batch_size=batch_size,
            num_epochs=num_epochs,
            logging_episodes=logging_episodes,
            save_episodes=save_episodes,
            eval_episodes=eval_episodes,
        )
        move_callbacks.append(self.episode_progress_callback)

        self.training_ds_sample_weights = training_ds_sample_weights

        self.worker_idx = int(os.environ.get("MLP_ROLE_INDEX", 0))
        self.local_rank_idx = int(os.environ.get('LOCAL_RANK', -1))

        super().__init__(model=policy, callbacks=move_callbacks, *args, **kwargs)
    
    
    def get_train_dataloader(self):
        """Override to use sequential sampler for memory-based training."""
        if self.sequential_sampler is not None:
            from torch.utils.data import DataLoader
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.sequential_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=self.args.dataloader_persistent_workers if self.args.dataloader_num_workers > 0 else False,
                prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
            )
        return super().get_train_dataloader()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Update episode progress if available
        if hasattr(self, 'episode_progress_callback') and 'episode_idx' in inputs:
            # Get all episode indices in the current batch
            episode_indices = inputs['episode_idx'].cpu().tolist()
            self.episode_progress_callback.update_episode(episode_indices)
        
        # apply image transforms to the inputs of understanding expert
        if self.image_transforms is not None:
            for key, value in inputs.items():
                if "history" in key or "mask" in key:
                    continue
                if key.startswith("observation.images"):
                    inputs[key] = self.image_transforms(value)

        outputs = self.policy.forward_with_world_model(
            inputs, 
            cur_n_obs_img_steps=self.cur_n_obs_img_steps, 
            cur_n_pred_img_steps=self.cur_n_pred_img_steps,
            train_gen_expert_only=self.args.train_gen_expert_only,
            gen_out_loss_ratio=self.args.gen_out_loss_ratio,
        )

        loss = outputs["loss"]
        
        # Update real-time metrics for progress bar (every step)
        if hasattr(self, 'episode_progress_callback') and self.state.is_local_process_zero:
            wm_loss = outputs.get("wm_loss", torch.tensor(0)).cpu().item()
            wm_acc = outputs.get("wm_acc_mean", torch.tensor(0)).cpu().item()
            action_loss = outputs.get("action_loss", torch.tensor(0)).cpu().item()
            self.episode_progress_callback.update_metrics(
                loss=loss.cpu().item(),
                wm_loss=wm_loss,
                wm_acc=wm_acc,
                action_loss=action_loss
            )

        # Episode-based logging (instead of step-based)
        if self.state.is_local_process_zero and self.state.is_world_process_zero:
            if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_log():
                action_lr_log = {
                    "action_learning_rate": self.optimizer.param_groups[-1]["lr"],
                }
                action_log = {
                    "action_loss": outputs.get("action_loss", torch.tensor(0)).cpu().item(),
                }
                episode_log = {
                    "episode": self.episode_progress_callback.total_episode_count,
                    "epoch": self.episode_progress_callback.current_epoch + 1,
                }
                if self.policy.use_world_model:
                    wm_log = {
                        "wm_out_loss": outputs.get("wm_loss", torch.tensor(0)).cpu().item(),
                        "wm_acc_mean": outputs.get("wm_acc_mean", torch.tensor(0)).cpu().item(),
                        "wm_acc_tail": outputs.get("wm_acc_tail", torch.tensor(0)).cpu().item(),
                        "wm_learning_rate": self.optimizer.param_groups[4]["lr"],
                    }
                    vit_log = {
                        "vit_learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    if self.policy.model.train_gen_expert_only:
                        loss_dict = {**episode_log, **wm_log, **vit_log}
                    else:
                        loss_dict = {**episode_log, **wm_log, **vit_log, **action_lr_log, **action_log}
                else:
                    loss_dict = {**episode_log, **action_lr_log, **action_log}

                self.log(loss_dict)
                self.episode_progress_callback.mark_logged()
        
        # Episode-based saving
        if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_save():
            if self.state.is_world_process_zero:
                episode_count = self.episode_progress_callback.total_episode_count
                save_dir = os.path.join(self.args.output_dir, f"checkpoint-episode-{episode_count}")
                self._save(save_dir)
                logger.info(f"Saved checkpoint at episode {episode_count}")
            self.episode_progress_callback.mark_saved()
        
        # Episode-based evaluation with video generation
        if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_eval():
            if self.state.is_world_process_zero and self.eval_dataset is not None:
                episode_count = self.episode_progress_callback.total_episode_count
                self._run_eval_with_video(episode_count)
            self.episode_progress_callback.mark_evaled()

        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def _run_eval_with_video(self, episode_count: int, num_samples: int = 4):
        """Run evaluation and generate ground truth vs prediction comparison video."""
        import random
        from PIL import Image
        import torchvision.transforms.functional as TF
        
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            logger.warning("No eval dataset provided, skipping video generation")
            return
        
        # Create output directory
        eval_dir = os.path.join(self.args.output_dir, "eval_videos")
        os.makedirs(eval_dir, exist_ok=True)
        
        # Sample random indices
        indices = random.sample(range(len(self.eval_dataset)), min(num_samples, len(self.eval_dataset)))
        
        self.policy.eval()
        frames_list = []  # List of (gt_frames, pred_frames) tuples
        
        for idx in indices:
            sample = self.eval_dataset[idx]
            
            # Prepare batch (add batch dimension)
            batch = {}
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.unsqueeze(0).to(self.args.device)
                else:
                    batch[key] = value
            
            # Apply image transforms if needed
            if self.image_transforms is not None:
                for key, value in batch.items():
                    if "history" in key or "mask" in key:
                        continue
                    if key.startswith("observation.images"):
                        batch[key] = self.image_transforms(value)
            
            # Get model predictions
            try:
                outputs = self.policy.forward_with_world_model(
                    batch,
                    cur_n_obs_img_steps=self.cur_n_obs_img_steps,
                    cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                    train_gen_expert_only=True,
                    gen_out_loss_ratio=1.0,
                    return_images=True,  # Request image outputs for visualization
                )
                
                # Get ground truth and predicted images
                # Assuming wm_pred_img and wm_gt_img are in outputs
                if 'wm_pred_img' in outputs and 'wm_gt_img' in outputs:
                    pred_img = outputs['wm_pred_img']  # [B, T, C, H, W] or [B, C, H, W]
                    gt_img = outputs['wm_gt_img']
                    frames_list.append((gt_img.cpu(), pred_img.cpu()))
            except Exception as e:
                logger.warning(f"Error during eval forward: {e}")
                continue
        
        if not frames_list:
            logger.warning("No frames generated for video")
            return
        
        # Create comparison video
        video_path = os.path.join(eval_dir, f"eval_episode_{episode_count}.mp4")
        self._create_comparison_video(frames_list, video_path)
        logger.info(f"Saved evaluation video to {video_path}")
        
        self.policy.train()
    
    def _create_comparison_video(self, frames_list, output_path: str, fps: int = 5):
        """Create side-by-side comparison video of ground truth and predictions."""
        try:
            import cv2
            import numpy as np
            
            all_frames = []
            
            for gt_tensor, pred_tensor in frames_list:
                # Handle different tensor shapes
                if gt_tensor.dim() == 5:  # [B, T, C, H, W]
                    gt_tensor = gt_tensor[0]  # [T, C, H, W]
                    pred_tensor = pred_tensor[0]
                elif gt_tensor.dim() == 4:  # [B, C, H, W]
                    gt_tensor = gt_tensor.unsqueeze(0)  # [1, C, H, W]
                    pred_tensor = pred_tensor.unsqueeze(0)
                
                for t in range(gt_tensor.shape[0]):
                    gt_frame = gt_tensor[t]  # [C, H, W]
                    pred_frame = pred_tensor[t]
                    
                    # Convert to numpy and denormalize
                    gt_np = gt_frame.permute(1, 2, 0).numpy()  # [H, W, C]
                    pred_np = pred_frame.permute(1, 2, 0).numpy()
                    
                    # Normalize to 0-255 range
                    gt_np = ((gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 255).astype(np.uint8)
                    pred_np = ((pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8) * 255).astype(np.uint8)
                    
                    # Add labels
                    gt_labeled = cv2.putText(gt_np.copy(), "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    pred_labeled = cv2.putText(pred_np.copy(), "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Concatenate side by side
                    combined = np.concatenate([gt_labeled, pred_labeled], axis=1)
                    all_frames.append(combined)
            
            if not all_frames:
                return
            
            # Write video
            h, w = all_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            for frame in all_frames:
                # Convert RGB to BGR for OpenCV
                if frame.shape[-1] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            
            writer.release()
            
        except ImportError:
            logger.warning("cv2 not available, skipping video generation. Install with: pip install opencv-python")
        except Exception as e:
            logger.warning(f"Error creating video: {e}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.accelerator.unwrap_model(self.model)._save_pretrained(Path(output_dir))
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):

        if model is None:
            model = self.model

        config_file = os.path.join(resume_from_checkpoint, CONFIG_NAME)
        adapter_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_WEIGHTS_NAME)
        adapter_safe_weights_file = os.path.join(resume_from_checkpoint, ADAPTER_SAFE_WEIGHTS_NAME)
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)
        is_fsdp_ckpt = os.path.isdir(resume_from_checkpoint) and (
            # this checks the FSDP state dict when `SHARDED_STATE_DICT` is used
            any(
                FSDP_MODEL_NAME in folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
            )
            # this checks the FSDP state dict when `FULL_STATE_DICT` is used
            or os.path.isfile(os.path.join(resume_from_checkpoint, f"{FSDP_MODEL_NAME}.bin"))
        )
        # if multiple adapters exist, they get saved in sub directories
        adapter_subdirs = (
            [
                folder_name
                for folder_name in os.listdir(resume_from_checkpoint)
                if os.path.isdir(os.path.join(resume_from_checkpoint, folder_name))
                and (
                    os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_WEIGHTS_NAME))
                    or os.path.isfile(os.path.join(resume_from_checkpoint, folder_name, ADAPTER_SAFE_WEIGHTS_NAME))
                )
            ]
            if os.path.isdir(resume_from_checkpoint)
            else []
        )

        if is_fsdp_ckpt and not self.is_fsdp_enabled:
            raise ValueError(f"Checkpoint found at {resume_from_checkpoint} is only supported when using PyTorch FSDP")

        if not (
            any(
                os.path.isfile(f)
                for f in [
                    weights_file,
                    safe_weights_file,
                    weights_index_file,
                    safe_weights_index_file,
                    adapter_weights_file,
                    adapter_safe_weights_file,
                ]
            )
            or is_fsdp_ckpt
            or adapter_subdirs
        ):
            raise ValueError(f"Can't find a valid checkpoint at {resume_from_checkpoint}")

        logger.info(f"Loading model from {resume_from_checkpoint}.")

        if os.path.isfile(config_file):
            config = PretrainedConfig.from_json_file(config_file)
            checkpoint_version = config.transformers_version
            if checkpoint_version is not None and checkpoint_version != __version__:
                logger.warning(
                    f"You are resuming training from a checkpoint trained with {checkpoint_version} of "
                    f"Transformers but your current version is {__version__}. This is not recommended and could "
                    "yield to errors or unwanted behaviors."
                )

        if os.path.isfile(weights_file) or os.path.isfile(safe_weights_file) or is_fsdp_ckpt:
            weights_only_kwarg = {"weights_only": True}
            # If the model is on the GPU, it still works!
            if is_sagemaker_mp_enabled():
                if os.path.isfile(os.path.join(resume_from_checkpoint, "user_content.pt")):
                    # If the 'user_content.pt' file exists, load with the new smp api.
                    # Checkpoint must have been saved with the new smp api.
                    smp.resume_from_checkpoint(
                        path=resume_from_checkpoint, tag=WEIGHTS_NAME, partial=False, load_optimizer=False
                    )
                else:
                    # If the 'user_content.pt' file does NOT exist, load with the old smp api.
                    # Checkpoint must have been saved with the old smp api.
                    if hasattr(self.args, "fp16") and self.args.fp16 is True:
                        logger.warning(
                            "Enabling FP16 and loading from smp < 1.10 checkpoint together is not suppported."
                        )
                    state_dict = torch.load(
                        weights_file,
                        map_location="cpu",
                        **weights_only_kwarg,
                    )
                    # Required for smp to not auto-translate state_dict from hf to smp (is already smp).
                    state_dict["_smp_is_partial"] = False
                    load_result = model.load_state_dict(state_dict, strict=True)
                    # release memory
                    del state_dict
            elif self.is_fsdp_enabled:
                load_fsdp_model(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    model,
                    resume_from_checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                # We load the model state dict on the CPU to avoid an OOM error.
                if self.args.save_safetensors and os.path.isfile(safe_weights_file):
                    model = PreTrainedPolicy._load_as_safetensor(model, safe_weights_file, "cpu", False)
                    logger.info(f"\033[31mLoading model from {safe_weights_file} complete !!\033[0m")
                else:
                    raise NotImplementedError("Not implemented")

        # Load adapters following PR # 24096
        elif _is_peft_model(model):
            # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
            # TODO: in the future support only specific min PEFT versions
            if (hasattr(model, "active_adapter") or hasattr(model, "active_adapters")) and hasattr(
                model, "load_adapter"
            ):
                if os.path.exists(resume_from_checkpoint):
                    # For BC for older PEFT versions
                    if hasattr(model, "active_adapters"):
                        active_adapters = model.active_adapters
                        if len(active_adapters) > 1:
                            logger.warning("Multiple active adapters detected will only consider the first adapter")
                        active_adapter = active_adapters[0]
                    else:
                        active_adapter = model.active_adapter

                    if adapter_subdirs:
                        for subdir_name in adapter_subdirs:
                            peft_id = os.path.join(resume_from_checkpoint, subdir_name)
                            model.load_adapter(peft_id, subdir_name, is_trainable=(subdir_name == active_adapter))
                        model.set_adapter(active_adapter)
                    else:
                        model.load_adapter(resume_from_checkpoint, active_adapter, is_trainable=True)
                else:
                    logger.warning(
                        "The intermediate checkpoints of PEFT may not be saved correctly, "
                        f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                        "Check some examples here: https://github.com/huggingface/peft/issues/96"
                    )
            else:
                logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
        else:
            # We load the sharded checkpoint
            load_result = load_sharded_checkpoint(
                model, resume_from_checkpoint, strict=is_sagemaker_mp_enabled(), prefer_safe=self.args.save_safetensors
            )
            if not is_sagemaker_mp_enabled():
                self._issue_warnings_after_load(load_result)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return LargeScaleWeightedRandomSampler(self.training_ds_sample_weights, len(self.train_dataset))
