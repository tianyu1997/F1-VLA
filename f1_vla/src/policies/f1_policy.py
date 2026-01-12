import os
import logging
import packaging
from pathlib import Path
from collections import deque
from typing import List, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE

import safetensors
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.constants import ACTION, OBS_STATE
from lerobot.policies.pi0.modeling_pi0 import resize_with_pad, pad_vector, create_sinusoidal_pos_embedding

from transformers import AutoTokenizer
from transformers.utils import logging

from f1_vla.src.models.modeling_f1 import F1FlowMatching
from f1_vla.src.models.wm.vqvae import VQVAE
from f1_vla.src.models.configuration_f1 import F1Config


logger = logging.get_logger(__name__)


class F1_VLA(nn.Module):
    config_class = F1Config
    cache_action_steps = 5

    def __init__(
        self,
        config: F1Config,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.use_world_model = config.use_world_model
        
        # Determine target device for VAE loading
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_device = device

        self.language_tokenizer = AutoTokenizer.from_pretrained(config.language_tokenizer_path)

        pn = config.gen_expert_config.pn
        patch_nums = tuple(map(int, pn.replace('-', '_').split('_')))

        # VAE trainability is controlled by VQVAE's own flags.
        # Do NOT forcibly freeze all params here; it breaks configurations that enable decoder training.
        vae_test_mode = getattr(config, "vae_test_mode", getattr(config.gen_expert_config.vae, "test_mode", True))
        vae_freeze_encoder = getattr(config, "vae_freeze_encoder", getattr(config.gen_expert_config.vae, "freeze_encoder", True))

        self.vae = VQVAE(
            vocab_size=config.gen_expert_config.vae.vocab_size, 
            z_channels=config.gen_expert_config.vae.z_channels, 
            ch=config.gen_expert_config.vae.ch, 
            dropout=getattr(config.gen_expert_config.vae, 'dropout', 0.0),
            test_mode=vae_test_mode,
            freeze_encoder=vae_freeze_encoder,
            share_quant_resi=config.gen_expert_config.vae.share_quant_resi, 
            v_patch_nums=patch_nums
        )
        if os.path.exists(config.gen_expert_config.vae.vae_ckpt):
            # Load VAE weights directly to target device
            vae_ckpt = torch.load(config.gen_expert_config.vae.vae_ckpt, map_location=device, weights_only=False)
            self.vae.load_state_dict(vae_ckpt, strict=True)
            del vae_ckpt
        # Move VAE to target device
        self.vae = self.vae.to(device)

        self.use_only_3rd_hist_image = True
        self.last_l = patch_nums[-1] * patch_nums[-1]

        # Add label smoothing to prevent overfitting on discrete VAE tokens
        # FIXED: Reduced from 0.1 to 0.02 for 4096-class VAE tokens (less aggressive regularization)
        self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=0)

        self.model = F1FlowMatching(config, patch_nums, self.vae, **kwargs)

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.cache_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    # ==================== Multi-Actor Interface ====================
    
    @property
    def active_actor(self) -> str:
        """Get the currently active actor name."""
        return self.model.active_actor
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        elif hasattr(self.model, "paligemma"):
             # If F1FlowMatching wraps paligemma directly
             if hasattr(self.model.paligemma, "gradient_checkpointing_enable"):
                self.model.paligemma.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        elif hasattr(self.model, "paligemma"):
            if hasattr(self.model.paligemma, "gradient_checkpointing_disable"):
                self.model.paligemma.gradient_checkpointing_disable()
    
    @active_actor.setter
    def active_actor(self, actor_name: str):
        """Set the active actor by name."""
        self.model.active_actor = actor_name
    
    def get_actor(self, actor_name: str):
        """Get a specific actor expert by name."""
        return self.model.get_actor(actor_name)
    
    def add_actor(self, actor_name: str, random_init: bool = True):
        """Add a new actor expert dynamically.
        
        Args:
            actor_name: Name for the new actor (e.g., "explorer")
            random_init: If True, randomly initialize weights. If False, copy from active actor.
        """
        self.model.add_actor(actor_name, random_init=random_init)
    
    def list_actors(self) -> list:
        """Return list of available actor names."""
        return self.model.list_actors()
    
    def set_trainable_actors(self, actor_names: list):
        """Set which actors should have trainable parameters.
        
        Args:
            actor_names: List of actor names to train. Other actors will be frozen.
        """
        self.model.set_trainable_actors(actor_names)
    
    def save_actor(self, actor_name: str, save_path: str):
        """Save a specific actor's weights.
        
        Args:
            actor_name: Name of the actor to save
            save_path: Path to save the actor weights
        """
        actor = self.get_actor(actor_name)
        torch.save(actor.state_dict(), save_path)
        logger.info(f"Saved actor '{actor_name}' to {save_path}")
    
    def load_actor(self, actor_name: str, load_path: str, strict: bool = True):
        """Load weights for a specific actor.
        
        Args:
            actor_name: Name of the actor to load weights into
            load_path: Path to the saved actor weights
            strict: Whether to strictly enforce state dict key matching
        """
        actor = self.get_actor(actor_name)
        state_dict = torch.load(load_path, map_location='cpu', weights_only=True)
        actor.load_state_dict(state_dict, strict=strict)
        logger.info(f"Loaded actor '{actor_name}' from {load_path}")
    
    def forward_with_actor(
        self,
        batch: dict[str, Tensor],
        actor_name: str = None,
        return_action_stats: bool = False,
        noise: Tensor | None = None,
        time: Tensor | None = None,
        **kwargs
    ) -> dict[str, Tensor]:
        """Forward pass with a specific actor.
        
        This method temporarily switches to the specified actor, performs
        the forward pass, and returns the results including optional 
        action statistics for RL training.
        
        Args:
            batch: Input batch with observations. Can be:
                   - Full F1-VLA batch format with "observation.images.image0", etc.
                   - Simplified format with "state_emb" or "observation.state" for RL
            actor_name: Name of the actor to use (if None, uses active actor)
            return_action_stats: If True, return additional stats for RL
            noise: Optional noise for flow matching
            time: Optional time for flow matching
            **kwargs: Additional arguments passed to underlying forward
            
        Returns:
            dict with:
                - 'action': Action output (B, action_dim)
                - 'state_emb': State embedding for value head (if return_action_stats)
        """
        # Save current active actor
        original_actor = self.active_actor
        
        # Switch to specified actor if provided
        if actor_name is not None and actor_name != original_actor:
            self.active_actor = actor_name
        
        try:
            output_dict = {}
            
            # Check if we have full image data or just state embeddings (RL mode)
            has_images = "observation.images.image0" in batch
            
            if has_images:
                # Full forward path with images
                images, img_masks = self.prepare_mix_images(batch)
                state = self.prepare_state(batch)
                lang_tokens, lang_masks = self.prepare_language(batch)
                
                # Get state embedding
                state_emb = self.model.state_proj(state)  # (B, hidden_dim)
                output_dict['state_emb'] = state_emb
                
                B = state.shape[0]
                device = state.device
                
                # Prepare time embedding
                if time is None:
                    time = torch.ones(B, device=device)
                
                # Time projection using sinusoidal embedding (same as modeling_f1.py)
                # NOTE: action MLPs are in float32, paligemma is in bfloat16
                from lerobot.policies.pi0.modeling_pi0 import create_sinusoidal_pos_embedding
                time_emb = create_sinusoidal_pos_embedding(
                    time, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
                )
                time_emb = time_emb.float()  # Keep in float32 for action MLPs
                
                # Prepare action input
                action_dim = self.config.max_action_dim
                chunk_size = self.config.chunk_size
                
                if noise is None:
                    action_input = torch.zeros(B, chunk_size, action_dim, device=device, dtype=torch.float32)
                else:
                    action_input = noise.float()
                
                # Project action input (action_in_proj is float32)
                action_emb = self.model.action_in_proj(action_input)
                
                # Fuse timestep + action information using an MLP (same as modeling_f1.py)
                time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
                action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
                action_time_emb = self.model.action_time_mlp_in(action_time_emb)
                action_time_emb = F.silu(action_time_emb)  # swish == silu
                action_time_emb = self.model.action_time_mlp_out(action_time_emb)
                
                # Get prefix embeddings from understanding expert
                und_embs, und_pad_masks, und_att_masks = self.model.embed_prefix(
                    images, img_masks, lang_tokens, lang_masks
                )
                
                # Ensure dtypes match und_embs (paligemma is bfloat16)
                target_dtype = und_embs.dtype
                action_time_emb = action_time_emb.to(dtype=target_dtype)
                state_emb = state_emb.to(dtype=target_dtype)
                
                # Add state and action_time_emb to create suffix
                suffix_embs = torch.cat([
                    state_emb.unsqueeze(1),
                    action_time_emb,
                ], dim=1)
                
                suffix_len = suffix_embs.shape[1]
                suffix_pad_masks = torch.ones(B, suffix_len, dtype=und_pad_masks.dtype, device=device)
                suffix_att_masks = torch.ones(B, suffix_len, dtype=und_att_masks.dtype, device=device)
                
                pad_masks = torch.cat([und_pad_masks, suffix_pad_masks], dim=1)
                att_masks = torch.cat([und_att_masks, suffix_att_masks], dim=1)
                
                from f1_vla.src.models.modeling_f1 import make_att_2d_masks
                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
                position_ids = torch.cumsum(pad_masks, dim=1) - 1
                
                # Forward through the model
                (_, _, act_out), _ = self.model.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[und_embs, None, suffix_embs],
                    use_cache=False,
                    fill_kv_cache=False,
                )
                
                # Output action processing (act_out already processed by model)
                action_features = act_out[:, 1:, :]  # Skip state token
                # Convert to float32 for action_out_proj (which is float32)
                action_features = action_features.to(dtype=torch.float32)
                action_output = self.model.action_out_proj(action_features)
                action_mean = action_output[:, 0, :]
                
            else:
                # Simplified RL mode - use state embedding directly
                # Get state from observation - raise error if missing
                if "state_emb" in batch:
                    state_emb = batch["state_emb"]
                elif "observation.state" in batch:
                    state = batch["observation.state"]
                    if state.dim() == 1:
                        state = state.unsqueeze(0)
                    state_emb = self.model.state_proj(state)
                else:
                    raise ValueError(
                        f"forward_with_actor (RL mode) requires 'state_emb' or 'observation.state' in batch. "
                        f"Got keys: {list(batch.keys())}"
                    )
                
                output_dict['state_emb'] = state_emb
                
                B = state_emb.shape[0]
                device = state_emb.device
                dtype = state_emb.dtype
                
                # For RL mode, use a simple MLP path through action projection
                # Simplified: just use state + noise -> action
                
                # Time embedding using sinusoidal encoding (same as embed_suffix)
                if time is None:
                    time = torch.ones(B, device=device)
                
                from lerobot.policies.pi0.modeling_pi0 import create_sinusoidal_pos_embedding
                time_emb = create_sinusoidal_pos_embedding(
                    time, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
                )
                time_emb = time_emb.type(dtype=dtype)
                
                # Project action input (zeros for deterministic forward)
                # For RL, we use chunk_size=1 for single-step actions
                action_dim = self.config.max_action_dim
                rl_chunk_size = 1  # Single step for RL
                
                if noise is None:
                    action_input = torch.zeros(B, rl_chunk_size, action_dim, device=device, dtype=dtype)
                else:
                    action_input = noise.to(device=device, dtype=dtype)
                    if action_input.dim() == 2:
                        action_input = action_input.unsqueeze(1)  # (B, action_dim) -> (B, 1, action_dim)
                
                action_emb = self.model.action_in_proj(action_input)
                
                # Fuse time and action
                time_emb_expanded = time_emb[:, None, :].expand_as(action_emb)
                action_time_emb = torch.cat([action_emb, time_emb_expanded], dim=2)
                
                # Pass through MLPs
                action_time_emb = self.model.action_time_mlp_in(action_time_emb)
                action_time_emb = F.silu(action_time_emb)
                action_time_emb = self.model.action_time_mlp_out(action_time_emb)
                
                # Project to action space and squeeze chunk dim
                action_mean = self.model.action_out_proj(action_time_emb)  # (B, 1, action_dim)
                action_mean = action_mean.squeeze(1)  # (B, action_dim)
            
            # Truncate to actual action dim
            actual_action_dim = 7
            if action_mean.shape[-1] > actual_action_dim:
                action_mean = action_mean[..., :actual_action_dim]
            
            output_dict['action'] = action_mean
            
            return output_dict
            
        finally:
            # Restore original actor
            if actor_name is not None and actor_name != original_actor:
                self.active_actor = original_actor

    # ==================== End Multi-Actor Interface ====================

    @torch.no_grad
    def select_action_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        top_k: int = 900,
        top_p: float = 0.95,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Tensor:
        self.eval()

        if len(self._action_queue) == 0:
            images, image_masks = self.prepare_mix_images(batch)
            world_model_images = self.prepare_mix_history_images(batch)
            state = self.prepare_state(batch)
            lang_tokens, lang_masks = self.prepare_language(batch)

            B, T, C, H, W = world_model_images.shape
            world_model_images = world_model_images.reshape(B * T, C, H, W)

            world_model_indices_list = self.model.vae.img_to_idxBl(world_model_images)
            world_model_input_embs = self.model.vae.quantize.idxBl_to_var_input(world_model_indices_list)
            world_model_input_embs = world_model_input_embs.reshape(B, T, *world_model_input_embs.shape[1:])

            action_output = self.model.sample_actions_with_world_model(
                images=images, 
                image_masks=image_masks, 
                lang_tokens=lang_tokens, 
                lang_masks=lang_masks, 
                state=state, 
                world_model_input_embs=world_model_input_embs, 
                predict_action_only=False, 
                noise=noise,
                top_k=top_k, top_p=top_p, num_samples=num_samples, rng=rng,
            )
            actions = action_output.actions

            # Unpad actions
            original_action_dim = 7
            actions = actions[:, :, :original_action_dim]

            return actions

    def _get_memory_state(self, batch: dict[str, Tensor]) -> tuple:
        """
        Get memory state for the batch if memory is enabled.
        
        Returns:
            (memory_kv, memory_token, should_detach) if memory enabled
            (None, None, False) if memory disabled
            
        Gradient flow design:
        - frame_idx == 0: Uses init_memory (nn.Parameter), keeps gradients for learning
        - frame_idx > 0: Uses stored memory from memory bank (already detached in store_memory)
        
        The should_detach flag controls BPTT truncation - when True, gradients won't flow
        back to previous time steps through the memory.
        
        Note: Memory from bank is already detached (done in store_memory), so only
        init_memory contributes gradients. This is the correct BPTT behavior.
        """
        if not self.config.use_memory or self.model.memory_manager is None:
            return None, None, False
        
        device = batch["observation.state"].device
        dtype = next(self.model.parameters()).dtype
        
        memory_kv, memory_token, should_detach_list = self.model.memory_manager.process_batch(batch, device, dtype)
        
        # Detach if ANY sample needs detach (conservative for BPTT correctness)
        # This controls whether we detach AFTER forward, before storing back
        should_detach = any(should_detach_list)
        
        # Note: We do NOT detach memory_kv here anymore because:
        # 1. Memory from bank is already detached (in store_memory)
        # 2. init_memory (used at frame_idx==0) should keep gradients for learning
        # 3. The "backward through graph twice" error was caused by store_memory
        #    not properly detaching, which is now fixed
        
        return memory_kv, memory_token, should_detach
    
    def _update_memory_state(
        self, 
        batch: dict[str, Tensor], 
        updated_memory, 
        should_detach: bool
    ) -> None:
        """Store updated memory state after forward pass."""
        if self.config.use_memory and self.model.memory_manager is not None:
            self.model.memory_manager.store_updated_memory(
                batch, updated_memory, detach=should_detach
            )

    def forward_with_world_model(
        self, 
        batch: dict[str, Tensor], 
        noise: Tensor | None = None, 
        time: Tensor | None = None, 
        cur_n_obs_img_steps: int | None = None, 
        cur_n_pred_img_steps: int | None = None,  
        train_gen_expert_only: bool = False, 
        gen_out_loss_ratio: float = 0.1,
        return_images: bool = False,  # Return gt/pred images for eval visualization
        return_memory_info: bool = False,  # Return memory_info for custom BPTT handling
        skip_memory_store: bool = False,   # Skip default store_updated_memory (for chunked BPTT)
        memory_kv: list[tuple[Tensor, Tensor]] | None = None,
        memory_token: Tensor | None = None,
    ) -> dict[str, Tensor]:

        #########################################################
        # prepare the inputs
        #########################################################

        images, img_masks = self.prepare_mix_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        action_is_pad = batch.get("action_is_pad")

        world_model_images = self.prepare_mix_history_images(batch)
        B, T, C, H, W = world_model_images.shape
        world_model_images = world_model_images.reshape(B * T, C, H, W)
        
        # DEBUG: Check world_model_images before VAE encoding
        if torch.isnan(world_model_images).any() or torch.isinf(world_model_images).any():
            logger.error(f"[F1_VLA] world_model_images has nan/inf BEFORE VAE! "
                        f"shape={world_model_images.shape}, nan={torch.isnan(world_model_images).sum()}")
        
        world_model_image_indices = self.model.vae.img_to_idxBl(world_model_images)
        
        # DEBUG: Check indices after VAE encoding
        for i, idx in enumerate(world_model_image_indices):
            if torch.isnan(idx.float()).any() or torch.isinf(idx.float()).any():
                logger.error(f"[F1_VLA] world_model_image_indices[{i}] has nan/inf! shape={idx.shape}")
            if (idx < 0).any() or (idx >= self.config.gen_expert_config.vae.vocab_size).any():
                logger.error(f"[F1_VLA] world_model_image_indices[{i}] out of range! min={idx.min()}, max={idx.max()}")
        
        # prepare the output of world model
        gt_world_model_indices = torch.cat(world_model_image_indices, dim=1).reshape(B, T, -1)[:, cur_n_obs_img_steps: cur_n_obs_img_steps + cur_n_pred_img_steps].contiguous()
        # prepare the input of world model
        world_model_embs = self.model.vae.quantize.idxBl_to_var_input(world_model_image_indices)
        
        # DEBUG: Check embeddings after quantization
        if torch.isnan(world_model_embs).any() or torch.isinf(world_model_embs).any():
            logger.error(f"[F1_VLA] world_model_embs has nan/inf! shape={world_model_embs.shape}, "
                        f"nan={torch.isnan(world_model_embs).sum()}")
        
        world_model_embs = world_model_embs.reshape(B, T, *world_model_embs.shape[1:])
        world_model_input_embs = world_model_embs[:, :cur_n_obs_img_steps]
        world_model_output_embs = world_model_embs[:, cur_n_obs_img_steps:cur_n_obs_img_steps + cur_n_pred_img_steps]
        if len(world_model_output_embs.shape) == 4:
            world_model_output_embs = world_model_output_embs.reshape(B, -1, world_model_output_embs.shape[3])

        #########################################################
        # Get memory state if enabled
        #########################################################
        # 允许外部传入 memory 以支持 chunked BPTT
        if memory_kv is None and memory_token is None:
            # 完全由内部获取 memory
            memory_kv, memory_token, should_detach = self._get_memory_state(batch)
        elif memory_kv is not None and memory_token is not None:
            # 外部传入了完整的 memory 状态
            should_detach = False
        else:
            # 部分传入的情况，补全缺失的部分
            if self.config.use_memory and self.model.memory_manager is not None:
                device = batch["observation.state"].device
                dtype = next(self.model.parameters()).dtype
                if memory_kv is None:
                    memory_kv, _, _ = self.model.memory_manager.process_batch(batch, device, dtype)
                if memory_token is None:
                    batch_size = batch["observation.state"].shape[0]
                    memory_token = self.model.memory_bank.get_memory_token(batch_size, device, dtype)
            should_detach = False

        #########################################################
        # Forward and compute the loss
        #########################################################
        # Temporarily set train_gen_expert_only flag on model
        original_train_gen_expert_only = getattr(self.model, 'train_gen_expert_only', False)
        self.model.train_gen_expert_only = train_gen_expert_only
        
        # DEBUG: Check all model inputs for NaN/Inf
        def _check_tensor(name, t):
            if t is not None and isinstance(t, torch.Tensor):
                if torch.isnan(t).any() or torch.isinf(t).any():
                    logger.error(f"[F1_VLA] {name} has nan/inf! shape={t.shape}")
                    return True
            return False
        
        has_nan_input = False
        has_nan_input |= _check_tensor("images", images)
        has_nan_input |= _check_tensor("state", state)
        has_nan_input |= _check_tensor("world_model_input_embs", world_model_input_embs)
        has_nan_input |= _check_tensor("world_model_output_embs", world_model_output_embs)
        has_nan_input |= _check_tensor("actions", actions)
        if memory_kv is not None:
            for i, (k, v) in enumerate(memory_kv):
                has_nan_input |= _check_tensor(f"memory_kv[{i}].k", k)
                has_nan_input |= _check_tensor(f"memory_kv[{i}].v", v)
        has_nan_input |= _check_tensor("memory_token", memory_token)
        
        if has_nan_input:
            logger.error("[F1_VLA] Model inputs have NaN/Inf! Skipping forward.")
        
        try:
            action_losses, gen_logits, memory_info, past_key_values = self.model.forward_with_world_model(
                images, img_masks, lang_tokens, lang_masks, state, 
                world_model_input_embs, world_model_output_embs, actions, noise, time, 
                memory_kv=memory_kv, memory_token=memory_token
            )
        finally:
            # Restore original flag
            self.model.train_gen_expert_only = original_train_gen_expert_only

        gen_token_len = gen_logits.shape[1]
        gt_world_model_indices = gt_world_model_indices.reshape(B, -1)[:, :gen_token_len]
        
        # Debug: check for invalid values before CrossEntropyLoss
        vocab_size = self.config.gen_expert_config.vae.vocab_size
        if torch.isnan(gen_logits).any() or torch.isinf(gen_logits).any():
            logger.error(f"[F1_VLA] gen_logits has nan/inf! shape={gen_logits.shape}, "
                        f"nan_count={torch.isnan(gen_logits).sum()}, inf_count={torch.isinf(gen_logits).sum()}")
        if (gt_world_model_indices < 0).any() or (gt_world_model_indices >= vocab_size).any():
            logger.error(f"[F1_VLA] gt_world_model_indices out of range! "
                        f"min={gt_world_model_indices.min()}, max={gt_world_model_indices.max()}, "
                        f"vocab_size={vocab_size}")
        
        gen_loss_ce = self.gen_loss_fct(gen_logits.reshape(-1, gen_logits.shape[-1]), gt_world_model_indices.reshape(-1)).view(B, -1)
        
        # Pixel-level reconstruction loss
        pixel_loss_weight = getattr(self.config, 'pixel_loss_weight', 0.0)
        if pixel_loss_weight > 0:
            # Get predicted indices from logits
            pred_indices = gen_logits.argmax(dim=-1)  # [B, gen_token_len]
            
            # Decode images
            with torch.set_grad_enabled(True):  # Enable gradients for VAE decoder
                gt_images = self._decode_indices_to_images(gt_world_model_indices, B, cur_n_pred_img_steps)
                pred_images = self._decode_indices_to_images(pred_indices, B, cur_n_pred_img_steps)
            
            # Compute pixel loss (MSE or L1)
            pixel_loss_type = getattr(self.config, 'pixel_loss_type', 'mse')  # 'mse' or 'l1'
            if pixel_loss_type == 'l1':
                pixel_loss = F.l1_loss(pred_images, gt_images)
            else:
                pixel_loss = F.mse_loss(pred_images, gt_images)
        else:
            pixel_loss = torch.tensor(0.0, device=gen_loss_ce.device)
        
        # Combine losses
        gen_loss_combined = gen_loss_ce + pixel_loss_weight * pixel_loss
        
        # Episode-internal loss warmup: weight down loss for early frames
        # Memory is inaccurate at episode start, so early frame predictions should contribute less
        frame_indices = batch.get("frame_idx")
        warmup_frames = getattr(self.config, 'loss_warmup_frames', 8)  # Default 8 frames warmup
        warmup_min_weight = getattr(self.config, 'loss_warmup_min_weight', 0.1)  # Minimum weight for frame 0
        
        if frame_indices is not None and warmup_frames > 0:
            # Linear warmup from warmup_min_weight to 1.0 over warmup_frames
            # weight = warmup_min_weight + (1 - warmup_min_weight) * min(frame_idx / warmup_frames, 1.0)
            frame_indices_float = frame_indices.float()  # Don't overwrite original frame_indices
            loss_weights = warmup_min_weight + (1.0 - warmup_min_weight) * torch.clamp(frame_indices_float / warmup_frames, max=1.0)
            loss_weights = loss_weights.view(B, 1).expand_as(gen_loss_ce)  # [B, gen_token_len]
            gen_loss_ce_weighted = (gen_loss_ce * loss_weights).mean()
            avg_loss_weight = loss_weights[:, 0].mean()  # Average weight across batch
        else:
            gen_loss_ce_weighted = gen_loss_ce.mean()
            avg_loss_weight = torch.tensor(1.0, device=gen_loss_ce.device)
        
        # Final gen_loss with pixel loss
        gen_loss = gen_loss_ce_weighted + pixel_loss_weight * pixel_loss

        # Update memory with GRU and store to memory bank
        if self.config.use_memory and self.model.memory_manager is not None and not skip_memory_store:
            dataset_indices = batch.get("dataset_idx")
            episode_indices = batch.get("episode_idx")
            frame_indices = batch.get("frame_idx")
            
            if dataset_indices is not None and episode_indices is not None and frame_indices is not None:
                # Update memory content using GRU if memory_info is available
                if memory_info is not None and memory_kv is not None:
                    updated_memory = self.model.memory_bank.update_memory(memory_kv, memory_info)
                    # store_updated_memory will handle both storing and step count update
                    self.model.memory_manager.store_updated_memory(batch, updated_memory, detach=should_detach)

        loss_dict = {}
        # Store past_key_values for memory distillation (student needs gradients)
        loss_dict["past_key_values"] = past_key_values
        loss_dict["wm_acc_mean"] = (gen_logits.argmax(dim=-1) == gt_world_model_indices).float().mean()
        loss_dict["loss_weight"] = avg_loss_weight  # Monitor warmup weight
        loss_dict["wm_loss_ce"] = gen_loss_ce_weighted  # Cross-entropy loss
        loss_dict["wm_loss_pixel"] = pixel_loss  # Pixel reconstruction loss
        last_resolution_token_len = self.model.num_resolutions * self.model.num_resolutions
        loss_dict["wm_acc_tail"] = (gen_logits[:, -last_resolution_token_len:].argmax(dim=-1) == gt_world_model_indices[:, -last_resolution_token_len:]).float().mean()

        if train_gen_expert_only:
            loss_dict["loss"] = gen_loss
            loss_dict["wm_loss"] = gen_loss
            
            # Generate images for eval visualization if requested (for gen_expert_only mode)
            if return_images:
                # Get predicted indices from logits
                pred_indices = gen_logits.argmax(dim=-1)  # [B, gen_token_len]
                
                # Decode ground truth images
                gt_images = self._decode_indices_to_images(gt_world_model_indices, B, cur_n_pred_img_steps)
                # Decode predicted images
                pred_images = self._decode_indices_to_images(pred_indices, B, cur_n_pred_img_steps)
                
                loss_dict["wm_gt_img"] = gt_images
                loss_dict["wm_pred_img"] = pred_images
            
            # Add memory_info for chunked BPTT
            if return_memory_info:
                loss_dict["memory_info"] = memory_info
            
            return loss_dict

        loss_dict["action_losses_after_forward"] = action_losses.clone()

        if action_is_pad is not None:
            in_episode_bound = ~action_is_pad
            if action_losses.shape == in_episode_bound.shape:
                action_losses = action_losses * in_episode_bound
            else:
                action_losses = action_losses * in_episode_bound.unsqueeze(-1)
            loss_dict["action_losses_after_in_ep_bound"] = action_losses.clone()

        # Remove padding
        action_losses = action_losses[:, :, : self.config.max_action_dim]
        loss_dict["action_losses_after_rm_padding"] = action_losses.clone()
        loss_dict["action_loss"] = action_losses.mean().clone()
        loss_dict["wm_loss"] = gen_loss.clone()

        loss_dict["loss"] = loss_dict["action_loss"] + gen_out_loss_ratio * loss_dict["wm_loss"]

        #########################################################
        # Generate images for eval visualization if requested
        #########################################################
        if return_images:
            # Get predicted indices from logits
            pred_indices = gen_logits.argmax(dim=-1)  # [B, gen_token_len]
            
            # For GT images: use ORIGINAL images from batch, NOT VAE reconstructions
            # This avoids flickering caused by VAE reconstruction errors
            if hasattr(self.config, 'camera_config') and self.config.camera_config:
                wm_target_key = self.config.camera_config.get('world_model_target_key',
                    "observation.images.image0_target")
            else:
                wm_target_key = "observation.images.image0_target"
            
            # Get original target images [B, n_pred, C, H, W] in range [0, 1]
            gt_images_original = batch.get(wm_target_key)
            
            if gt_images_original is not None:
                # Resize if needed (from 256x256 to match VAE output)
                # Already in [0, 1] range from dataset
                gt_images = gt_images_original
            else:
                # Fallback: decode from indices if original not available
                gt_images = self._decode_indices_to_images(gt_world_model_indices, B, cur_n_pred_img_steps)
            
            # Decode predicted images
            pred_images = self._decode_indices_to_images(pred_indices, B, cur_n_pred_img_steps)
            
            loss_dict["wm_gt_img"] = gt_images
            loss_dict["wm_pred_img"] = pred_images

        if return_memory_info:
            loss_dict["memory_info"] = memory_info
        return loss_dict
    
    def _decode_indices_to_images(self, indices: Tensor, batch_size: int, num_frames: int) -> Tensor:
        """Decode VAE indices back to images.
        
        Args:
            indices: [B, num_tokens] VAE token indices where num_tokens = tokens_per_frame * num_frames
            batch_size: Batch size
            num_frames: Number of frames to decode
            
        Returns:
            images: [B, T, C, H, W] decoded images (256x256)
        """
        try:
            # VAE uses multi-scale patches: v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            # tokens_per_scale = [1, 4, 9, 16, 25, 36, 64, 100, 169, 256] = 680 total per frame
            v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            tokens_per_scale = [pn**2 for pn in v_patch_nums]
            tokens_per_frame = sum(tokens_per_scale)  # 680
            
            # Reshape indices for each frame: [B, T*680] -> [B*T, 680]
            indices = indices.view(batch_size * num_frames, tokens_per_frame)
            
            # Split into multi-scale list for idxBl_to_img
            # idxBl_to_img expects: List[Tensor] where each tensor is [B, l] and l = pn^2
            ms_idx_Bl = []
            start_idx = 0
            for num_tokens in tokens_per_scale:
                ms_idx_Bl.append(indices[:, start_idx:start_idx + num_tokens])
                start_idx += num_tokens
            
            # Decode using VAE
            with torch.no_grad():
                # idxBl_to_img returns [B*T, C, H, W] when same_shape=True, last_one=True
                images = self.model.vae.idxBl_to_img(ms_idx_Bl, same_shape=True, last_one=True)
            
            # Reshape to [B, T, C, H, W]
            # Output is 256x256 (16x16 patches * 16 = 256)
            if images.dim() == 4:  # [B*T, C, H, W]
                images = images.view(batch_size, num_frames, *images.shape[1:])
            
            # Convert from [-1, 1] to [0, 1] range
            images = (images + 1) / 2
            
            return images
        except Exception as e:
            logger.warning(f"Error decoding indices to images: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(batch_size, num_frames, 3, 256, 256, device=indices.device)

    def prepare_mix_images(self, batch):
        images = []
        image_masks = []

        # Use camera config if available, otherwise fall back to default keys
        if hasattr(self.config, 'camera_config') and self.config.camera_config:
            img_keys = self.config.camera_config.get('understanding_image_keys', 
                ["observation.images.image0", "observation.images.image1"])
        else:
            img_keys = [
                "observation.images.image0",
                "observation.images.image1",
                "observation.images.image2",
            ]

        for key in img_keys:
            if key not in batch:
                img = torch.zeros_like(batch["observation.images.image0"])
                mask = torch.zeros_like(batch["observation.images.image0_mask"])
                # Take only the last frame (current observation) for understanding expert
                if len(img.shape) == 5:
                    img = img[:, -1]  # (b, t, c, h, w) -> (b, c, h, w)
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
                images.append(img)
                image_masks.append(mask)
                continue
            img = batch[key]
            # Take only the last frame (current observation) for understanding expert
            if len(img.shape) == 5:
                img = img[:, -1]  # (b, t, c, h, w) -> (b, c, h, w)
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            img = img * 2.0 - 1.0

            images.append(img)
            image_masks.append(batch[f"{key}_mask"])
        
        # delete the empty images
        for i in range(len(images) - 1, -1, -1):
            if images[i].sum() == 0:
                images.pop(i)
                image_masks.pop(i)

        return images, image_masks

    def prepare_mix_history_images(self, batch):
        """Prepare history images + target images for world model training."""
        # Use camera config if available
        if hasattr(self.config, 'camera_config') and self.config.camera_config:
            wm_input_key = self.config.camera_config.get('world_model_input_key', 
                "observation.images.image0_history")
            wm_target_key = self.config.camera_config.get('world_model_target_key',
                "observation.images.image0_target")
        else:
            wm_input_key = "observation.images.image0_history"
            wm_target_key = "observation.images.image0_target"
        
        # Get history images (n_obs_img_steps frames)
        hist_img = batch[wm_input_key]  # (B, n_obs, C, H, W)
        
        # Get target images (n_pred_img_steps frames)
        target_img = batch[wm_target_key]  # (B, n_pred, C, H, W)
        
        # Concatenate history and target along time dimension
        # Result: (B, n_obs + n_pred, C, H, W)
        combined = torch.cat([hist_img, target_img], dim=1)

        # Normalize [0, 1] -> [-1, 1] for VAE
        combined = combined * 2.0 - 1.0

        return combined

    def _format_history_text(self, batch) -> List[str]:
        """
        Format action and state history as text for PaliGemma.
        
        Returns list of formatted history strings, one per batch item.
        Format: "History: [state t-3: s0,s1,...] [action t-3: a0,a1,...] ..."
        """
        history_texts = []
        batch_size = batch[OBS_STATE].shape[0]
        
        # Check if history is available
        has_state_history = "observation.state_history" in batch
        has_action_history = "action_history" in batch
        
        if not (has_state_history or has_action_history):
            return [""] * batch_size
        
        for b in range(batch_size):
            parts = []
            
            # State history: (n_obs_img_steps, state_dim)
            if has_state_history:
                state_hist = batch["observation.state_history"][b]  # (n_steps, dim)
                n_steps = state_hist.shape[0]
                state_strs = []
                for t in range(n_steps):
                    # Format state values with 2 decimal places
                    vals = state_hist[t].tolist()
                    # Truncate to first 8 dims for brevity
                    vals_str = ",".join([f"{v:.2f}" for v in vals[:8]])
                    if len(vals) > 8:
                        vals_str += "..."
                    state_strs.append(f"s{t-n_steps+1}:[{vals_str}]")
                parts.append("state:" + " ".join(state_strs))
            
            # Action history: (n_obs_img_steps, action_dim) - now aligned with state_history
            if has_action_history:
                action_hist = batch["action_history"][b]  # (n_steps, dim)
                if action_hist.numel() > 0:
                    n_steps = action_hist.shape[0]
                    action_strs = []
                    for t in range(n_steps):
                        vals = action_hist[t].tolist()
                        # Truncate to first 8 dims for brevity
                        vals_str = ",".join([f"{v:.2f}" for v in vals[:8]])
                        if len(vals) > 8:
                            vals_str += "..."
                        # Use same indexing as state: t-n_steps+1 means a{t-3}, a{t-2}, a{t-1}, a{0}
                        action_strs.append(f"a{t-n_steps+1}:[{vals_str}]")
                    parts.append("action:" + " ".join(action_strs))
            
            if parts:
                history_texts.append("History: " + " | ".join(parts) + " ")
            else:
                history_texts.append("")
        
        return history_texts

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input, optionally including action/state history"""
        device = batch[OBS_STATE].device
        tasks = batch["task"]
        
        # Determine max_length: use extended length if memory is enabled
        if self.config.use_memory:
            max_length = self.config.memory_config.tokenizer_max_length
            # Get history text
            history_texts = self._format_history_text(batch)
            # Combine: task + history
            tasks = [
                f"{task} {hist}" if not task.endswith("\n") else f"{task[:-1]} {hist}\n"
                for task, hist in zip(tasks, history_texts)
            ]
        else:
            max_length = self.config.tokenizer_max_length

        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=max_length,
            return_tensors="pt",
            truncation=True,
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: F1Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = F1Config.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)

        if model_id.endswith(".json"):
            model_id = "/".join(model_id.split("/")[:-1])

        if os.path.isdir(model_id):
            logger.info(f"Loading weights from local directory: {model_id}")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model, model_file: str, map_location: str, strict: bool):
        if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
            load_model_as_safetensor(model, model_file, strict=strict)
            if map_location != "cpu":
                logger.warning(
                    "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                    " This means that the model is loaded on 'cpu' first and then copied to the device."
                    " This leads to a slower loading time."
                    " Please update safetensors to version 0.4.3 or above for improved performance."
                )
                model.to(map_location)
        else:
            # Use safe_open with explicit device="cpu" to avoid device mapping issues in distributed training
            from safetensors import safe_open
            state_dict = {}
            with safe_open(model_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
            model.load_state_dict(state_dict, strict=strict)
        return model

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config.save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))
