import os
import json
import random
import numpy as np
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
    freeze_paligemma: bool = False  # Freeze entire PaliGemma (vision + language model)
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
    Implements state persistence for proper checkpoint resume.
    """
    
    def __init__(self, policy=None, num_episodes: int = 0, total_steps: int = 0, batch_size: int = 4, num_epochs: int = 1,
                 logging_episodes: int = 10, save_episodes: int = 100, eval_episodes: int = 50):
        self.policy = policy
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
        self.current_epoch = -1  # Initialized to -1 so first on_epoch_begin starts at 0 (displayed as E1)
        self.current_episode_batch = 0
        self.seen_episodes = set()  # Track all seen episode indices
        self.epoch_episode_count = 0  # Episodes seen in current epoch
        self.total_episode_count = 0  # Total episodes across all epochs
        self.last_log_episode = 0
        self.last_save_episode = 0
        self.last_eval_episode = 0
        
        # Flag to track if we've been restored from checkpoint
        self._restored_from_checkpoint = False
        self._is_first_epoch_step = False  # Track first epoch step after resume
        
        # Real-time metrics (updated every step from compute_loss)
        self.current_loss = 0.0
        self.current_wm_loss = 0.0
        self.current_wm_acc = 0.0
        self.current_action_loss = 0.0
    
    def state(self) -> dict:
        """Return state to be saved in trainer_state.json for checkpoint resume."""
        return {
            "current_epoch": self.current_epoch,
            "total_episode_count": self.total_episode_count,
            "epoch_episode_count": self.epoch_episode_count,
            "last_log_episode": self.last_log_episode,
            "last_save_episode": self.last_save_episode,
            "last_eval_episode": self.last_eval_episode,
            "seen_episodes": list(self.seen_episodes),
        }
    
    def load_state(self, state: dict):
        """Load state from trainer_state.json when resuming from checkpoint."""
        # Load total count which is the ground truth
        self.total_episode_count = state.get("total_episode_count", 0)
        self.last_log_episode = state.get("last_log_episode", 0)
        self.last_save_episode = state.get("last_save_episode", 0)
        self.last_eval_episode = state.get("last_eval_episode", 0)
        
        # Calculate current epoch based on dataset size
        if self.num_episodes > 0:
            self.current_epoch = self.total_episode_count // self.num_episodes
            # We do NOT restore epoch_episode_count or seen_episodes complexly
            # We simply start counting "seen in this session's epoch" from zero
            # If HF skips batches correctly, we will just see the remainder of the epoch
            self.epoch_episode_count = 0 
        else:
            self.current_epoch = state.get("current_epoch", 0)
            self.epoch_episode_count = 0

        self.seen_episodes = set()
        
        self._restored_from_checkpoint = True
        logger.info(f"[EpisodeProgressCallback] Restored state: total_episodes={self.total_episode_count} -> calculated epoch={self.current_epoch}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize progress bar at training start."""
        from tqdm import tqdm
        self._is_first_epoch_step = True  # Next on_epoch_begin is the first one after start/resume
        
        if state.is_local_process_zero:
            # If fresh start (not restored), estimate from state.epoch
            if not self._restored_from_checkpoint and state.epoch:
                 self.total_episode_count = int(state.epoch * self.num_episodes)
                 self.current_epoch = int(state.epoch)
            
            # Recalculate epoch one last time to be sure
            if self.num_episodes > 0:
                self.current_epoch = self.total_episode_count // self.num_episodes
            
            # Description
            desc = f"E{self.current_epoch + 1}"
            
            # Initial progress within current epoch
            initial_progress = 0 # We always start bar from 0 for the "remainder" or "new epoch"
            
            self.pbar = tqdm(
                total=self.num_episodes,
                initial=initial_progress,
                desc=desc,
                dynamic_ncols=True,
                leave=True,
                unit="ep",
                ncols=80,
                bar_format='{desc}|{bar:20}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]'
            )
            self.pbar.set_postfix_str('loss=0 acc=0%')
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Reset episode counter at epoch start."""
        # Update current epoch based on absolute total count
        if self.num_episodes > 0:
            self.current_epoch = self.total_episode_count // self.num_episodes
        
        self.seen_episodes.clear()
        self.epoch_episode_count = 0
        
        if self.pbar is not None and state.is_local_process_zero:
            self.pbar.reset()
            self.pbar.set_description(f"E{self.current_epoch + 1}")
        
        # Clear memory bank at epoch boundary
        if hasattr(self, 'policy') and hasattr(self.policy, 'model'):
            model = self.policy.model
            if hasattr(model, 'memory_bank'):
                old_size = len(model.memory_bank._memory_bank)
                model.memory_bank._memory_bank.clear()
                logger.info(f"[EpisodeProgressCallback] Cleared {old_size} memory states at epoch {self.current_epoch} start")
        
        # Force GPU memory cleanup
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(f"[EpisodeProgressCallback] Cleared CUDA cache at epoch {self.current_epoch} start")
    
    def on_step_end(self, args, state, control, **kwargs):
        """Update progress bar after each step."""
        if self.pbar is not None and state.is_local_process_zero:
            # Update display with real-time metrics (compact format)
            self.pbar.set_postfix_str(f'L={self.current_wm_loss:.3f} A={self.current_wm_acc:.1%}')
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Update metrics display when logging (for logged metrics)."""
        if self.pbar is not None and state.is_local_process_zero and logs is not None:
            # Compact postfix format
            loss = logs.get('wm_out_loss', logs.get('loss', 0))
            acc = logs.get('wm_acc_mean', 0)
            self.pbar.set_postfix_str(f'L={loss:.3f} A={acc:.1%}')

        # Lightweight GPU memory snapshot for visibility (once per log event)
        if state.is_local_process_zero and torch.cuda.is_available():
            mem_info = []
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / 1e9
                reserv = torch.cuda.memory_reserved(i) / 1e9
                mem_info.append(f"{i}:A{alloc:.2f}G/R{reserv:.2f}G")
            logger.info(f"[GPU Memory] {' | '.join(mem_info)}")
    
    def update_metrics(self, loss: float, wm_loss: float = 0.0, wm_acc: float = 0.0, action_loss: float = 0.0):
        """Update real-time metrics from compute_loss (every micro-batch)."""
        # Initialize EMA variables if they don't exist
        if not hasattr(self, 'ema_loss'):
            self.ema_loss = loss
            self.ema_wm_loss = wm_loss
            self.ema_wm_acc = wm_acc
            self.ema_action_loss = action_loss
        
        # Exponential moving average (alpha=0.01 for smooth updates over ~100 steps)
        alpha = 0.02
        self.ema_loss = (1 - alpha) * self.ema_loss + alpha * loss
        self.ema_wm_loss = (1 - alpha) * self.ema_wm_loss + alpha * wm_loss
        self.ema_wm_acc = (1 - alpha) * self.ema_wm_acc + alpha * wm_acc
        self.ema_action_loss = (1 - alpha) * self.ema_action_loss + alpha * action_loss
         
        self.current_loss = self.ema_loss
        self.current_wm_loss = self.ema_wm_loss
        self.current_wm_acc = self.ema_wm_acc
        self.current_action_loss = self.ema_action_loss
    
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
                self.pbar.set_description(f"E{self.current_epoch + 1}/{self.num_epochs}")
        
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
        should = episodes_since_eval >= self.eval_episodes
        if should:
            logger.info(f"[EpisodeProgressCallback] should_eval=True: total={self.total_episode_count}, "
                       f"last_eval={self.last_eval_episode}, since_eval={episodes_since_eval}, threshold={self.eval_episodes}")
        return should
    
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
        """Called at Trainer epoch boundary.
        
        Note: We explicitly clear memory to prevent leaks across epochs.
        """
        if hasattr(self.policy, 'model'):
            # Clear memory bank
            if hasattr(self.policy.model, 'memory_bank') and self.policy.model.memory_bank is not None:
                if hasattr(self.policy.model.memory_bank, 'clear_memory_bank'):
                    self.policy.model.memory_bank.clear_memory_bank()
                elif hasattr(self.policy.model.memory_bank, '_memory_bank'):
                    self.policy.model.memory_bank._memory_bank.clear()
                logger.info(f"[PolicyTrainerCallback] Cleared memory bank at epoch {state.epoch} start")
            
            # Clear memory manager step counts (for BPTT)
            if hasattr(self.policy.model, 'memory_manager') and self.policy.model.memory_manager is not None:
                if hasattr(self.policy.model.memory_manager, '_step_counts'):
                    self.policy.model.memory_manager._step_counts.clear()
                logger.info(f"[PolicyTrainerCallback] Cleared memory manager step counts at epoch {state.epoch} start")

        # Force GC and CUDA cache clear
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        logger.info(f"[PolicyTrainer] eval_dataset received: {eval_dataset is not None}, type: {type(eval_dataset)}")
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
            policy=policy,
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

        # Store eval_dataset before calling super().__init__() since Trainer may override it
        self._custom_eval_dataset = eval_dataset
        
        super().__init__(model=policy, callbacks=move_callbacks, *args, **kwargs)
        
        # Restore our custom eval_dataset after Trainer.__init__()
        self.eval_dataset = self._custom_eval_dataset
        logger.info(f"[PolicyTrainer] eval_dataset after init: {self.eval_dataset is not None}")

        # Ensure gradient clipping is active even if the caller does not set it explicitly.
        # HuggingFace Trainer applies clip_grad_norm_ when max_grad_norm > 0.
        if getattr(self.args, "max_grad_norm", None) in (None, 0):
            self.args.max_grad_norm = 1.0
            logger.info("[PolicyTrainer] max_grad_norm not set; defaulting to 1.0 for gradient clipping")

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        super().save_model(output_dir, _internal_call)
        
        # Save callback state explicitly
        if self.episode_progress_callback and self.is_world_process_zero():
             try:
                 state = self.episode_progress_callback.state()
                 state_path = os.path.join(output_dir, "episode_progress.json")
                 with open(state_path, "w") as f:
                     json.dump(state, f)
                 logger.info(f"Saved episode progress to {state_path}")
             except Exception as e:
                 logger.warning(f"Failed to save episode progress: {e}")

    def _save_full_checkpoint(self, output_dir: str):
        """Save complete checkpoint including optimizer, scheduler, and RNG state."""
        # 1. Save model, args, trainer_state (via standard _save)
        self._save(output_dir)
        
        # 2. Save optimizer and scheduler (only on main process)
        if self.is_world_process_zero() and not self.args.save_only_model:
            if self.optimizer is not None:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if self.lr_scheduler is not None:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        
        # 3. Save RNG state (only on main process for now, mirroring HF behavior)
        if self.is_world_process_zero() and not self.args.save_only_model:
            rng_states = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                if self.args.local_rank != -1:  # Distributed
                    rng_states["cuda"] = torch.cuda.get_rng_state()
                else:
                    rng_states["cuda"] = torch.cuda.get_rng_state_all()
            
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # Try to load callback state
        if self.episode_progress_callback:
            state_path = os.path.join(resume_from_checkpoint, "episode_progress.json")
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r") as f:
                        state = json.load(f)
                    self.episode_progress_callback.load_state(state)
                    logger.info(f"Loaded episode progress from {state_path}")
                except Exception as e:
                    logger.warning(f"Failed to load episode progress from {state_path}: {e}")
            else:
                logger.info(f"No episode progress file found at {state_path}, will estimate from global_step")
                
                # Check if we can infer from checkpoint directory name
                try:
                    dir_name = os.path.basename(resume_from_checkpoint.rstrip('/'))
                    if dir_name.startswith("checkpoint-episode-"):
                        ep_count = int(dir_name.split("-")[-1])
                        self.episode_progress_callback.total_episode_count = ep_count
                        # Also update last markers to prevent immediate re-save/re-eval
                        self.episode_progress_callback.last_save_episode = ep_count
                        self.episode_progress_callback.last_log_episode = ep_count
                        self.episode_progress_callback.last_eval_episode = ep_count
                        
                        # Mark as restored so on_train_begin doesn't overwrite with estimation
                        self.episode_progress_callback._restored_from_checkpoint = True
                        
                        # Populate seen_episodes with dummies to match the count within current epoch
                        # We guess epoch_episode_count based on ep_count and num_episodes
                        # This avoids double-counting the current episode immediately after resume
                        if self.episode_progress_callback.num_episodes > 0:
                             epoch_ep_count = ep_count % self.episode_progress_callback.num_episodes
                             if epoch_ep_count == 0 and ep_count > 0:
                                 # Edge case: we are exactly at end of epoch multple
                                 epoch_ep_count = self.episode_progress_callback.num_episodes
                             
                             self.episode_progress_callback.epoch_episode_count = epoch_ep_count
                             for i in range(epoch_ep_count):
                                 self.episode_progress_callback.seen_episodes.add(-(i + 1))
                                 
                        logger.info(f"Inferred episode count {ep_count} from checkpoint name '{dir_name}'")
                except Exception as e:
                    logger.warning(f"Failed to infer episode count from checkpoint name: {e}")
    
    
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

        # Chunked BPTT path: sampler已按 window_length 打包连续帧（batch 维度=时间维度）
        # window_length = n_obs_img_steps + k_bptt
        chunk_size = 1
        if hasattr(self.policy.config, 'memory_config') and self.policy.config.memory_config:
            k_bptt = getattr(self.policy.config.memory_config, 'k_bptt', 4) or 4
            n_obs = getattr(self, 'cur_n_obs_img_steps', 4)
            chunk_size = n_obs + k_bptt  # window_length
        if (self.use_memory and self.sequential_sampler is not None and chunk_size > 1
                and isinstance(inputs, dict) and len(inputs) > 0):
            batch_dim = list(inputs.values())[0].shape[0] if isinstance(list(inputs.values())[0], torch.Tensor) else 0
            if batch_dim >= chunk_size:
                return self._compute_loss_chunked(model, inputs, chunk_size, return_outputs)
        
        # apply image transforms to the inputs of understanding expert
        # Use MANUAL video transforms for consistency across time dimension
        if (self.image_transforms is not None and 
            hasattr(self.args, 'image_transforms_enabled') and 
            self.args.image_transforms_enabled and 
            hasattr(self.args, 'image_transforms_type')):
            # Group keys by camera to handle temporal consistency
            camera_groups = {}
            for key in inputs.keys():
                if key.startswith("observation.images.") and "mask" not in key:
                    # Extract base name: remove_history suffix
                    base_name = key.replace("_history", "")
                    if base_name not in camera_groups:
                        camera_groups[base_name] = []
                    camera_groups[base_name].append(key)
            
            # Apply transforms per camera group
            batch_size = list(inputs.values())[0].shape[0] if inputs else 0
            
            for base_name, keys in camera_groups.items():
                # Pre-generate augmentation parameters for the whole batch for this camera
                # This ensures consistent crop/brightness across history and current frame
                batch_crops = []
                batch_brights = []
                
                do_crop = 'random_crop' in self.args.image_transforms_type
                do_bright = 'brightness' in self.args.image_transforms_type
                
                # Check dimensions from the first tensor in the group
                ref_tensor = inputs[keys[0]]
                H, W = ref_tensor.shape[-2:]
                tgt_size = 224
                
                for b in range(batch_size):
                    # Crop params
                    if do_crop and H > tgt_size and W > tgt_size:
                        top = torch.randint(0, H - tgt_size + 1, (1,)).item()
                        left = torch.randint(0, W - tgt_size + 1, (1,)).item()
                        batch_crops.append((top, left, tgt_size, tgt_size))
                    else:
                        # Center crop or Resize fallback if RandomCrop disabled or image small
                        # Default to Center Crop for validation consistency if no random crop
                        if H > tgt_size and W > tgt_size:
                            top = (H - tgt_size) // 2
                            left = (W - tgt_size) // 2
                            batch_crops.append((top, left, tgt_size, tgt_size))
                        else:
                            batch_crops.append(None) # No crop needed/possible
                            
                    # Brightness params
                    if do_bright:
                        # factor in [0.8, 1.2]
                        batch_brights.append(0.8 + 0.4 * torch.rand(1).item())
                    else:
                        batch_brights.append(1.0)
                
                # Apply parameters to all keys for this camera
                for key in keys:
                    src_tensor = inputs[key] # (B, ..., H, W)
                    # Create output tensor with target size 224
                    # Preserve all dimensions except last two
                    out_shape = list(src_tensor.shape)
                    out_shape[-2] = tgt_size
                    out_shape[-1] = tgt_size
                    out_tensor = torch.empty(out_shape, device=src_tensor.device, dtype=src_tensor.dtype)
                    
                    for b in range(batch_size):
                        # Extract image (or video)
                        img = src_tensor[b] # (..., H, W)
                        
                        # Apply Brightness
                        if batch_brights[b] != 1.0:
                            img = img * batch_brights[b]
                            # Clamp? Usually fine for NN, but clean data implies 0-1
                            # img = torch.clamp(img, 0., 1.) 
                        
                        # Apply Crop
                        if batch_crops[b] is not None:
                            t, l, h, w = batch_crops[b]
                            img = img[..., t:t+h, l:l+w]
                        
                        out_tensor[b] = img
                    
                    # Update inputs
                    inputs[key] = out_tensor

        outputs = self.policy.forward_with_world_model(
            inputs, 
            cur_n_obs_img_steps=self.cur_n_obs_img_steps, 
            cur_n_pred_img_steps=self.cur_n_pred_img_steps,
            train_gen_expert_only=self.args.train_gen_expert_only,
            gen_out_loss_ratio=self.args.gen_out_loss_ratio,
        )

        loss = outputs["loss"]
        
        # Update real-time metrics for progress bar (every micro-batch for real-time display)
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
                    }
                    # Add learning rate logging if optimizer has enough param groups
                    if len(self.optimizer.param_groups) > 4:
                        wm_log["wm_learning_rate"] = self.optimizer.param_groups[4]["lr"]
                        vit_log = {"vit_learning_rate": self.optimizer.param_groups[0]["lr"]}
                    else:
                        vit_log = {"learning_rate": self.optimizer.param_groups[0]["lr"]}
                    
                    # Add teacher-student specific logging
                    ts_log = {}
                    if "gt_loss" in outputs:
                        ts_log["gt_loss"] = outputs["gt_loss"].cpu().item() if hasattr(outputs["gt_loss"], "cpu") else outputs["gt_loss"]
                    if "memory_loss" in outputs:
                        ts_log["memory_loss"] = outputs["memory_loss"].cpu().item() if hasattr(outputs["memory_loss"], "cpu") else outputs["memory_loss"]
                    if "teacher_wm_acc" in outputs:
                        ts_log["teacher_wm_acc"] = outputs["teacher_wm_acc"].cpu().item() if hasattr(outputs["teacher_wm_acc"], "cpu") else outputs["teacher_wm_acc"]
                    
                    # Check if policy is teacher-student or has train_gen_expert_only attribute
                    is_gen_expert_only = (hasattr(self.policy, 'model') and 
                                         hasattr(self.policy.model, 'train_gen_expert_only') and 
                                         self.policy.model.train_gen_expert_only)
                    
                    if is_gen_expert_only or ts_log:  # Teacher-student mode or gen_expert_only
                        loss_dict = {**episode_log, **wm_log, **vit_log, **ts_log}
                    else:
                        loss_dict = {**episode_log, **wm_log, **vit_log, **action_lr_log, **action_log}
                else:
                    loss_dict = {**episode_log, **action_lr_log, **action_log}

                self.log(loss_dict)
                self.episode_progress_callback.mark_logged()
        
        # Episode-based saving and eval (shared with _compute_loss_chunked)
        self._handle_episode_save_and_eval()

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_chunked(self, model, inputs, chunk_size: int, return_outputs: bool = False):
        """在单个 training step 内展开 window_length 帧，实现跨时间步 BPTT。

        Window 设计 (与 SequentialBatchSampler 配合):
        - window_length = n_obs_img_steps + k_bptt (例如 4 + 4 = 8)
        - 前 n_obs_img_steps 帧: 历史上下文，用于 memory warmup，loss 被 detach
        - 后 k_bptt 帧: 计算梯度的有效帧
        - stride = k_bptt: 相邻 window 重叠 n_obs_img_steps 帧
        
        这样设计确保:
        1. 每个计算梯度的帧都有完整的 n_obs_img_steps 历史上下文
        2. memory 从上下文帧 warmup 后再用于梯度帧
        3. 相邻 window 的 memory 状态正确衔接
        
        NOTE: 当前实现假设 per_device_train_batch_size=1，即 batch 维度仅包含
              单个 episode 的连续帧。若 batch_size>1，sampler 会混合多个 episode。
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        time_len = list(inputs.values())[0].shape[0] if inputs else 0
        effective_steps = min(time_len, chunk_size)
        
        # 验证：所有帧属于同一个 episode（仅支持 batch_size=1）
        if "episode_idx" in inputs and isinstance(inputs["episode_idx"], torch.Tensor):
            unique_eps = inputs["episode_idx"].unique()
            if len(unique_eps) > 1:
                logger.warning(
                    f"[_compute_loss_chunked] Multiple episodes in batch ({len(unique_eps)}). "
                    "Chunked BPTT assumes batch_size=1. Falling back to standard compute_loss."
                )
                return self._fallback_compute_loss(model, inputs, return_outputs)

        memory_manager = self.policy.model.memory_manager
        memory_bank = self.policy.model.memory_bank

        # 从 config 读取参数
        n_obs_img_steps = getattr(self, 'cur_n_obs_img_steps', 4)
        k_bptt = 4  # 默认值
        if hasattr(self.policy.config, 'memory_config') and self.policy.config.memory_config:
            k_bptt = getattr(self.policy.config.memory_config, 'k_bptt', 4) or 4
        
        # Episode 起始 frame_idx: 数据集从 n_obs_img_steps - 1 开始（确保有足够历史）
        episode_start_frame = n_obs_img_steps - 1
        
        # 初始化 memory（从 bank 取，或使用 init_memory）
        frame0 = {k: (v[0:1] if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        memory_kv, memory_token, _ = memory_manager.process_batch(
            frame0, device, dtype, 
            episode_start_frame=episode_start_frame
        )
        
        # Window 划分:
        # - context_frames: 前 n_obs_img_steps 帧，用于 memory warmup，不计算梯度
        # - gradient_frames: 后 k_bptt 帧，计算梯度
        context_frames = min(n_obs_img_steps, effective_steps)
        gradient_start = context_frames  # 梯度帧从这个位置开始
        
        losses = []
        final_wm_loss = None
        final_wm_acc = None
        final_action_loss = None
        
        # 检查是否是 episode 的起始 window
        is_episode_start = frame0["frame_idx"][0].item() == episode_start_frame

        for t in range(effective_steps):
            batch_t = {k: (v[t:t+1] if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            
            out_t = self.policy.forward_with_world_model(
                batch_t,
                cur_n_obs_img_steps=self.cur_n_obs_img_steps,
                cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                train_gen_expert_only=self.args.train_gen_expert_only,
                gen_out_loss_ratio=self.args.gen_out_loss_ratio,
                memory_kv=memory_kv,
                memory_token=memory_token,
                return_memory_info=True,
                skip_memory_store=True,
            )
            
            # Loss 梯度策略：
            # - 前 n_obs_img_steps 帧 (t < gradient_start): 上下文帧，loss detach
            # - 后 k_bptt 帧 (t >= gradient_start): 梯度帧，保留 loss 梯度
            # 
            # init_memory 梯度说明：
            # init_memory 的梯度通过 memory 链传递（而非直接通过 loss）：
            #   loss_t -> memory_t -> ... -> memory_0 -> init_memory
            # 因此只要 memory 链保持梯度（is_episode_start 时），后续帧的 loss
            # 就能将梯度传回 init_memory，无需对上下文帧的 loss 保留梯度。
            if t >= gradient_start:
                losses.append(out_t["loss"])
            else:
                # 上下文帧：只累积 loss 值用于监控，不参与梯度计算
                losses.append(out_t["loss"].detach())
            
            # 记录最后一步的指标
            if t == effective_steps - 1:
                final_wm_loss = out_t.get("wm_loss")
                final_wm_acc = out_t.get("wm_acc_mean")
                final_action_loss = out_t.get("action_loss")

            # 更新 memory
            memory_info = out_t.get("memory_info")
            if memory_info is not None:
                # Memory 梯度策略：
                # - 上下文帧 (t < gradient_start): detach memory 以截断梯度链
                # - 梯度帧 (t >= gradient_start): 保留 memory 梯度
                # - 特例：episode 起始 window 的所有帧都保留梯度（让 init_memory 能学习）
                if t >= gradient_start or is_episode_start:
                    # 保留梯度：让 init_memory/memory_token 能学习
                    memory_kv = memory_bank.update_memory(memory_kv, memory_info)
                else:
                    # 上下文帧：detach memory 以截断梯度链
                    memory_kv = memory_bank.update_memory(memory_kv, memory_info)
                    memory_kv = [(k.detach(), v.detach()) for k, v in memory_kv]

        loss = torch.stack(losses).mean()
        outputs = {
            "loss": loss,
            "wm_loss": final_wm_loss if final_wm_loss is not None else torch.tensor(0.0, device=device),
            "wm_acc_mean": final_wm_acc if final_wm_acc is not None else torch.tensor(0.0, device=device),
            "action_loss": final_action_loss if final_action_loss is not None else torch.tensor(0.0, device=device),
        }

        # 窗口结束：detach 后写回 bank，并重置 step_count，确保下一窗口重新累计
        ds_idx = inputs.get("dataset_idx", frame0["dataset_idx"])
        ep_idx = inputs.get("episode_idx", frame0["episode_idx"])
        frame_idx = inputs.get("frame_idx", frame0["frame_idx"])
        tail_batch = {
            "dataset_idx": ds_idx[effective_steps-1:effective_steps],
            "episode_idx": ep_idx[effective_steps-1:effective_steps],
            "frame_idx": frame_idx[effective_steps-1:effective_steps],
        }
        memory_manager.store_updated_memory(tail_batch, memory_kv, detach=True)
        for b in range(len(tail_batch["dataset_idx"])):
            key = (tail_batch["dataset_idx"][b].item(), tail_batch["episode_idx"][b].item())
            memory_manager._step_counts[key] = 0

        # 进度条实时指标
        if hasattr(self, 'episode_progress_callback') and self.state.is_local_process_zero:
            self.episode_progress_callback.update_metrics(
                loss=loss.detach().cpu().item(),
                wm_loss=outputs["wm_loss"].detach().cpu().item(),
                wm_acc=outputs["wm_acc_mean"].detach().cpu().item(),
                action_loss=outputs["action_loss"].detach().cpu().item(),
            )

        # Episode-based logging for chunked path
        if self.state.is_local_process_zero and self.state.is_world_process_zero:
            if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_log():
                episode_log = {
                    "episode": self.episode_progress_callback.total_episode_count,
                    "epoch": self.episode_progress_callback.current_epoch + 1,
                }
                wm_log = {
                    "wm_out_loss": outputs.get("wm_loss", torch.tensor(0)).cpu().item() if hasattr(outputs.get("wm_loss", 0), "cpu") else outputs.get("wm_loss", 0),
                    "wm_acc_mean": outputs.get("wm_acc_mean", torch.tensor(0)).cpu().item() if hasattr(outputs.get("wm_acc_mean", 0), "cpu") else outputs.get("wm_acc_mean", 0),
                    "wm_acc_tail": outputs.get("wm_acc_tail", torch.tensor(0)).cpu().item() if hasattr(outputs.get("wm_acc_tail", 0), "cpu") else outputs.get("wm_acc_tail", 0),
                }
                if len(self.optimizer.param_groups) > 4:
                    wm_log["wm_learning_rate"] = self.optimizer.param_groups[4]["lr"]
                    vit_log = {"vit_learning_rate": self.optimizer.param_groups[0]["lr"]}
                else:
                    vit_log = {"learning_rate": self.optimizer.param_groups[0]["lr"]}
                
                loss_dict = {**episode_log, **wm_log, **vit_log}
                self.log(loss_dict)
                self.episode_progress_callback.mark_logged()

        # Episode-based saving/eval (shared with compute_loss)
        self._handle_episode_save_and_eval()

        return (loss, outputs) if return_outputs else loss
    
    def _handle_episode_save_and_eval(self):
        """Handle episode-based saving and evaluation.
        
        This is called at the end of both compute_loss and _compute_loss_chunked.
        Logging is handled separately due to different output formats.
        """
        # Episode-based saving
        if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_save():
            if self.state.is_world_process_zero:
                episode_count = self.episode_progress_callback.total_episode_count
                save_dir = os.path.join(self.args.output_dir, f"checkpoint-episode-{episode_count}")
                self._save_full_checkpoint(save_dir)
                logger.info(f"Saved checkpoint at episode {episode_count}")
            self.episode_progress_callback.mark_saved()
        
        # Episode-based evaluation with video generation
        if hasattr(self, 'episode_progress_callback') and self.episode_progress_callback.should_eval():
            logger.info(f"[Eval] Triggering evaluation at episode {self.episode_progress_callback.total_episode_count}")
            if self.state.is_world_process_zero and self.eval_dataset is not None:
                episode_count = self.episode_progress_callback.total_episode_count
                try:
                    self._run_eval_with_video(episode_count)
                except Exception as e:
                    logger.warning(f"[Eval] Error during video generation: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.eval_dataset is None:
                logger.warning("[Eval] eval_dataset is None, skipping video generation")
            self.episode_progress_callback.mark_evaled()

    def _fallback_compute_loss(self, model, inputs, return_outputs: bool = False):
        """当 batch 包含多个 episode 时的回退路径，逐帧处理。"""
        device = next(model.parameters()).device
        batch_size = list(inputs.values())[0].shape[0]
        
        losses = []
        wm_losses = []
        wm_accs = []
        action_losses = []
        
        for t in range(batch_size):
            batch_t = {k: (v[t:t+1] if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            out_t = self.policy.forward_with_world_model(
                batch_t,
                cur_n_obs_img_steps=self.cur_n_obs_img_steps,
                cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                train_gen_expert_only=self.args.train_gen_expert_only,
                gen_out_loss_ratio=self.args.gen_out_loss_ratio,
            )
            losses.append(out_t["loss"])
            wm_losses.append(out_t.get("wm_loss", torch.tensor(0.0, device=device)))
            wm_accs.append(out_t.get("wm_acc_mean", torch.tensor(0.0, device=device)))
            action_losses.append(out_t.get("action_loss", torch.tensor(0.0, device=device)))
        
        loss = torch.stack(losses).mean()
        outputs = {
            "loss": loss,
            "wm_loss": torch.stack(wm_losses).mean(),
            "wm_acc_mean": torch.stack(wm_accs).mean(),
            "action_loss": torch.stack(action_losses).mean(),
        }
        return (loss, outputs) if return_outputs else loss
    
    @torch.no_grad()
    def _run_eval_with_video(self, episode_count: int, num_samples: int = 4):
        """Run evaluation and generate ground truth vs prediction comparison video for a full episode."""
        import random
        from PIL import Image
        import torchvision.transforms.functional as TF
        
        logger.info(f"[Eval] Starting video generation for episode {episode_count}")
        
        if self.eval_dataset is None or len(self.eval_dataset) == 0:
            logger.warning("No eval dataset provided, skipping video generation")
            return
        
        logger.info(f"[Eval] eval_dataset size: {len(self.eval_dataset)}")
        
        # Create output directory
        eval_dir = os.path.join(self.args.output_dir, "eval_videos")
        os.makedirs(eval_dir, exist_ok=True)
        logger.info(f"[Eval] Output directory: {eval_dir}")
        
        # Sample ALL frames from ONE complete episode
        total_samples = len(self.eval_dataset)
        
        # Pick a random starting point and find which episode it belongs to
        random_idx = random.randint(0, total_samples - 1)
        sample = self.eval_dataset[random_idx]
        target_episode = sample['episode_idx'].item() if isinstance(sample['episode_idx'], torch.Tensor) else sample['episode_idx']
        
        # Search backwards to find episode start
        start_idx = random_idx
        while start_idx > 0:
            prev_sample = self.eval_dataset[start_idx - 1]
            prev_episode = prev_sample['episode_idx'].item() if isinstance(prev_sample['episode_idx'], torch.Tensor) else prev_sample['episode_idx']
            if prev_episode != target_episode:
                break
            start_idx -= 1
        
        # Search forwards to find episode end
        end_idx = random_idx
        while end_idx < total_samples - 1:
            next_sample = self.eval_dataset[end_idx + 1]
            next_episode = next_sample['episode_idx'].item() if isinstance(next_sample['episode_idx'], torch.Tensor) else next_sample['episode_idx']
            if next_episode != target_episode:
                break
            end_idx += 1
        
        # Get all indices for this episode
        indices = list(range(start_idx, end_idx + 1))
        logger.info(f"[Eval] Sampling full episode {target_episode}: {len(indices)} frames (indices {start_idx} to {end_idx})")
        
        self.policy.eval()
        frames_list = []  # List of (gt_frames, pred_frames) tuples
        
        for idx in indices:
            sample = self.eval_dataset[idx]
            
            # Use data_collator to properly batch the sample (handles tokenization etc.)
            if self.data_collator is not None:
                batch = self.data_collator([sample])
            else:
                # Fallback: manual batching
                batch = {}
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.unsqueeze(0)
                    else:
                        batch[key] = [value]
            
            # Move to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.args.device)
            
            # NOTE: Do NOT apply image_transforms during eval video generation!
            # Random brightness/contrast/saturation transforms would cause flickering
            # between frames since each frame gets different random augmentation.
            # The model should be robust to un-augmented images during evaluation.
            
            # Get model predictions
            try:
                # logger.info(f"[Eval] Running forward pass for sample {idx}")
                outputs = self.policy.forward_with_world_model(
                    batch,
                    cur_n_obs_img_steps=self.cur_n_obs_img_steps,
                    cur_n_pred_img_steps=self.cur_n_pred_img_steps,
                    train_gen_expert_only=True,
                    gen_out_loss_ratio=1.0,
                    return_images=True,  # Request image outputs for visualization
                )
                
                # logger.info(f"[Eval] Forward pass output keys: {outputs.keys()}")
                
                # Get ground truth and predicted images
                # Assuming wm_pred_img and wm_gt_img are in outputs
                if 'wm_pred_img' in outputs and 'wm_gt_img' in outputs:
                    pred_img = outputs['wm_pred_img']  # [B, T, C, H, W] or [B, C, H, W]
                    gt_img = outputs['wm_gt_img']
                    # logger.info(f"[Eval] Got images: gt_shape={gt_img.shape}, pred_shape={pred_img.shape}")
                    frames_list.append((gt_img.cpu(), pred_img.cpu()))
                else:
                    logger.warning(f"[Eval] Missing image keys. Available: {outputs.keys()}")
            except Exception as e:
                logger.warning(f"Error during eval forward: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not frames_list:
            logger.warning("No frames generated for video")
            return
        
        # Create comparison video with episode info in filename
        video_path = os.path.join(eval_dir, f"eval_step_{episode_count}_ep{target_episode}.mp4")
        self._create_comparison_video(frames_list, video_path, indices=indices)
        logger.info(f"Saved evaluation video to {video_path}")
        
        self.policy.train()
    
    def _create_comparison_video(self, frames_list, output_path: str, fps: int = 5, indices=None):
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
                    
                    # Convert to numpy - data is already in 0-1 range
                    gt_np = gt_frame.permute(1, 2, 0).numpy()  # [H, W, C]
                    pred_np = pred_frame.permute(1, 2, 0).numpy()
                    
                    # Scale from 0-1 to 0-255 directly (don't use per-frame min/max normalization)
                    gt_np = (gt_np * 255).clip(0, 255).astype(np.uint8)
                    pred_np = (pred_np * 255).clip(0, 255).astype(np.uint8)
                    
                    # Add labels
                    gt_labeled = cv2.putText(gt_np.copy(), "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    pred_labeled = cv2.putText(pred_np.copy(), "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Concatenate side by side
                    combined = np.concatenate([gt_labeled, pred_labeled], axis=1)
                    all_frames.append(combined)
            
            if not all_frames:
                return
            
            # Save frames as temporary images, then use ffmpeg to create H.264 video
            h, w = all_frames[0].shape[:2]
            mp4_output_path = output_path if output_path.endswith('.mp4') else output_path.replace('.avi', '.mp4')
            
            # Create temp directory for frames
            import tempfile
            import subprocess
            temp_dir = tempfile.mkdtemp()
            
            try:
                # Save all frames as temporary PNG files
                for i, frame in enumerate(all_frames):
                    if frame.shape[-1] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    cv2.imwrite(os.path.join(temp_dir, f"frame_{i:04d}.png"), frame_bgr)
                
                # Use ffmpeg to create H.264 encoded MP4 (VS Code compatible)
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    '-crf', '23',
                    mp4_output_path
                ]
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"ffmpeg error: {result.stderr}")
                    # Fallback to OpenCV mp4v
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(mp4_output_path, fourcc, fps, (w, h))
                    for frame in all_frames:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else frame
                        writer.write(frame_bgr)
                    writer.release()
            finally:
                # Clean up temp directory
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Also save some sample frames as PNG for easy viewing
            frame_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            num_frames_to_save = min(5, len(all_frames))
            for i in range(num_frames_to_save):
                # Include sample index info if available
                idx_info = f"_sample{indices[i]}" if indices and i < len(indices) else ""
                frame_path = os.path.join(frame_dir, f"{base_name}_frame_{i:03d}{idx_info}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(all_frames[i], cv2.COLOR_RGB2BGR))
            
            logger.info(f"Video saved to {mp4_output_path} ({len(all_frames)} frames)")
            logger.info(f"Saved {num_frames_to_save} sample frames as PNG")
            
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
        
        # Save trainer_state.json for resume support
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        logger.info(f"Saved trainer_state.json to {output_dir}")

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
