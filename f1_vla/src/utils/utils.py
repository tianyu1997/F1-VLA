import os
import logging
from pathlib import Path
from omegaconf import OmegaConf

import numpy as np
from typing import Union

import torch
from torch.utils.data import Sampler

from f1_vla.src.policies.f1_policy import F1_VLA

logger = logging.getLogger(__name__)


def save_training_args(training_args, policy_config, config):
    os.makedirs(training_args.output_dir, exist_ok=True)
    policy_config.save_pretrained(Path(training_args.output_dir))

    if not os.path.exists(Path(training_args.output_dir) / "config.yaml"):
        OmegaConf.save(config, Path(training_args.output_dir) / "config.yaml")


def clean_overrides(override_args):
    cleaned_args = []
    for arg in override_args:
        if arg.startswith("--"):
            cleaned_args.append(arg[2:])
        else:
            cleaned_args.append(arg)
    return cleaned_args


def load_ckpt(policy, config):
    # Safely check for load_ckpt key (OmegaConf may raise exception)
    try:
        load_ckpt_path = config.exp.load_ckpt if hasattr(config.exp, 'load_ckpt') else None
    except:
        load_ckpt_path = None
    
    if load_ckpt_path is not None:
        import os
        from safetensors import safe_open
        
        ckpt_path = config.exp.load_ckpt
        # If it's a directory, look for model.safetensors inside
        if os.path.isdir(ckpt_path):
            ckpt_path = os.path.join(ckpt_path, "model.safetensors")
        
        logger.info(f"Loading pretrained checkpoint from: {ckpt_path}")
        
        # Check key compatibility before loading
        model_keys = set(policy.state_dict().keys())
        ckpt_keys = set()
        with safe_open(ckpt_path, framework="pt") as f:
            ckpt_keys = set(f.keys())
        
        matched = model_keys & ckpt_keys
        missing = model_keys - ckpt_keys
        extra = ckpt_keys - model_keys
        
        logger.info(f"  Model keys: {len(model_keys)}, Checkpoint keys: {len(ckpt_keys)}")
        logger.info(f"  Matched: {len(matched)}, Missing in ckpt: {len(missing)}, Extra in ckpt: {len(extra)}")
        
        if missing:
            logger.info(f"  NOTE: {len(missing)} keys not in checkpoint (VAE/PaliGemma loaded separately)")
        
        # Check for memory_info_proj shape mismatch (due to GRU fix)
        memory_proj_keys = [k for k in model_keys if 'memory_info_proj' in k]
        skip_keys = []
        if memory_proj_keys:
            model_state = policy.state_dict()
            with safe_open(ckpt_path, framework="pt") as f:
                for key in memory_proj_keys:
                    if key in ckpt_keys:
                        model_shape = model_state[key].shape
                        ckpt_tensor = f.get_tensor(key)
                        ckpt_shape = ckpt_tensor.shape
                        if model_shape != ckpt_shape:
                            skip_keys.append(key)
                            logger.warning(f"  SKIP {key}: shape mismatch (model={model_shape} vs ckpt={ckpt_shape})")
                            logger.warning(f"    -> Will randomly initialize this layer (GRU mechanism updated)")
        
        # Load checkpoint, skipping incompatible keys
        if skip_keys:
            import torch
            state_dict = {}
            with safe_open(ckpt_path, framework="pt") as f:
                for key in f.keys():
                    if key not in skip_keys:
                        state_dict[key] = f.get_tensor(key)
            
            missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded checkpoint with {len(skip_keys)} skipped keys (shape mismatch)")
            logger.info(f"  Skipped: {skip_keys}")
        else:
            # Load with strict=False since VAE and PaliGemma are loaded separately
            F1_VLA._load_as_safetensor(policy, ckpt_path, "cpu", strict=False)
        
        logger.info(f"Successfully loaded {len(matched) - len(skip_keys)} weights from pretrained checkpoint!")
    else:
        logger.info("No pretrained checkpoint specified, training from scratch")
        
    return policy


def set_policy_config(policy_config, src_config):
    """
    Set the policy config from the config file
    Args:
        policy_config: The policy config to set which is used to initialize the policy
        src_config: The policy config from the local config file
    """
    policy_config.pretrained_path = src_config.path
    policy_config.language_tokenizer_path = src_config.language_tokenizer_path

    policy_config.use_world_model = src_config.use_world_model

    if policy_config.use_world_model:
        policy_config.gen_expert_config.pn = src_config.pn
        policy_config.gen_expert_config.temporal_conv_kernel_size = src_config.temporal_conv_kernel_size
        policy_config.gen_expert_config.temporal_conv_stride = src_config.temporal_conv_stride
        policy_config.gen_expert_config.num_resolutions = src_config.num_resolutions
        policy_config.gen_expert_config.vae.vae_ckpt = src_config.vae_ckpt

    policy_config.resize_imgs_with_padding = eval(src_config.resize_imgs_with_padding)

    policy_config.attention_implementation = src_config.attention_implementation
    policy_config.chunk_size = src_config.chunk_size

    # Episode-internal loss warmup configuration
    if hasattr(src_config, 'loss_warmup_frames'):
        policy_config.loss_warmup_frames = src_config.loss_warmup_frames
    if hasattr(src_config, 'loss_warmup_min_weight'):
        policy_config.loss_warmup_min_weight = src_config.loss_warmup_min_weight
    
    # Pixel-level reconstruction loss configuration
    if hasattr(src_config, 'vae_config'):
        vae_cfg = src_config.vae_config
        if hasattr(vae_cfg, 'pixel_loss_weight'):
            policy_config.pixel_loss_weight = vae_cfg.pixel_loss_weight
        if hasattr(vae_cfg, 'pixel_loss_type'):
            policy_config.pixel_loss_type = vae_cfg.pixel_loss_type
        if hasattr(vae_cfg, 'freeze_encoder'):
            policy_config.vae_freeze_encoder = vae_cfg.freeze_encoder

    return policy_config


def set_camera_config(policy_config, exp_config):
    """
    Set camera configuration from exp config.
    
    Config format:
        und_camera_keys: list of camera keys for understanding expert (PaliGemma)
        wm_camera_key: camera key for world model prediction
    
    Args:
        policy_config: The policy config to update
        exp_config: The exp config from yaml
    """
    from f1_vla.src.models.configuration_f1 import DictWithAttrAccess
    
    if hasattr(exp_config, 'camera_config') and exp_config.camera_config:
        cam_cfg = exp_config.camera_config
        
        # Get camera keys from config
        und_camera_keys = cam_cfg.get("und_camera_keys", ["head_rgb", "wrist_rgb"])
        wm_camera_key = cam_cfg.get("wm_camera_key", "head_rgb")
        
        # Convert to native Python types (omegaconf ListConfig -> list)
        if hasattr(und_camera_keys, '_content'):  # OmegaConf ListConfig
            und_camera_keys = list(und_camera_keys)
        und_camera_keys = [str(k) for k in und_camera_keys]  # Ensure all are strings
        wm_camera_key = str(wm_camera_key)
        
        # Find wm_camera index in und_camera_keys (for image key naming)
        wm_camera_idx = und_camera_keys.index(wm_camera_key) if wm_camera_key in und_camera_keys else 0
        
        # Auto-generate observation image keys
        understanding_image_keys = [f"observation.images.image{i}" for i in range(len(und_camera_keys))]
        # World model always uses image0 naming (dataset convention)
        world_model_input_key = "observation.images.image0_history"
        world_model_target_key = "observation.images.image0_target"
        
        policy_config.camera_config = DictWithAttrAccess({
            "und_camera_keys": und_camera_keys,
            "wm_camera_key": wm_camera_key,
            "wm_camera_idx": int(wm_camera_idx),
            "understanding_image_keys": understanding_image_keys,
            "world_model_input_key": world_model_input_key,
            "world_model_target_key": world_model_target_key,
        })
    return policy_config


class LargeScaleWeightedRandomSampler(Sampler):
    def __init__(
        self, 
        weights: Union[torch.Tensor, list, np.ndarray], 
        num_samples: int, 
        replacement: bool = True, 
        max_block: int = 2**24 - 1
    ):
        if isinstance(weights, list):
            weights = torch.tensor(weights)
        elif isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.max_block = max_block

    def __iter__(self):
        return iter(self._sample_indices().tolist())

    def _sample_indices(self) -> torch.Tensor:
        weights = self.weights
        total_weight = weights.sum()
        indices = []
        n = len(weights)
        num_blocks = (n + self.max_block - 1) // self.max_block

        for i in range(num_blocks):
            start = i * self.max_block
            end = min((i + 1) * self.max_block, n)
            block_weights = weights[start:end].float()
            block_weight_sum = block_weights.sum()

            if block_weight_sum == 0:
                continue

            block_prob = block_weight_sum / total_weight
            block_sample_count = int(round(self.num_samples * block_prob.item()))
            sampled = torch.multinomial(block_weights, block_sample_count, self.replacement)
            indices.append(sampled + start)

        return torch.cat(indices)[:self.num_samples]  # truncate in case of rounding error

    def __len__(self):
        return self.num_samples


def convert_ds_stats_to_dict(ds_stats):
    for k, v in ds_stats.items():
        for _k, _v in v.items():
            if isinstance(_v, np.ndarray):
                ds_stats[k][_k] = _v.tolist()
    return ds_stats
