# coding=utf-8
# Copyright 2024 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import CONFIG_MAPPING, AutoConfig

logger = logging.get_logger(__name__)

class DictWithAttrAccess(dict):
    """A dictionary that supports both dict-style and attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class F1Config(PretrainedConfig):
    model_type = "f1"
    sub_configs = {"und_expert_config": AutoConfig, "gen_expert_config": AutoConfig, "act_expert_config": AutoConfig}

    def __init__(
        self,
        und_expert_config=None,
        gen_expert_config=None,
        act_expert_config=None,
        proj_width=1024,
        chunk_size=50,
        max_action_dim=32,
        max_state_dim=32,
        tokenizer_max_length=48,
        use_cache=True,
        use_world_model=True,
        attention_implementation="eager",
        resize_imgs_with_padding="(224, 224)",
        # Memory configuration
        use_memory=False,  # Memory switch: if False, behavior unchanged
        memory_config=None,  # Memory module configuration
        # Multi-camera configuration
        camera_config=None,  # Camera settings for multi-camera support
        # Episode-internal loss warmup
        loss_warmup_frames=8,  # Linear warmup over first N frames
        loss_warmup_min_weight=0.1,  # Minimum loss weight for frame 0
        # Multi-actor configuration
        actor_config=None,  # Multi-actor settings for explorer training
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_world_model = use_world_model
        self.is_encoder_decoder = False

        self.proj_width = proj_width
        self.chunk_size = chunk_size

        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim

        self.tokenizer_max_length = tokenizer_max_length
        self.use_cache = use_cache
        self.attention_implementation = attention_implementation
        self.resize_imgs_with_padding = resize_imgs_with_padding

        # Episode-internal loss warmup
        self.loss_warmup_frames = loss_warmup_frames
        self.loss_warmup_min_weight = loss_warmup_min_weight

        # Multi-actor configuration
        # Supports multiple action experts (e.g., "actor" for policy, "explorer" for RL exploration)
        if actor_config is None:
            actor_config = {}
        self.actor_config = DictWithAttrAccess({
            # List of actor names to initialize. "actor" is the default policy actor.
            "actor_names": list(actor_config.get("actor_names", ["actor"])),
            # Which actor to use for training/inference. Can be overridden at runtime.
            "active_actor": str(actor_config.get("active_actor", "actor")),
            # Checkpoints for each actor (dict: actor_name -> ckpt_path or None for random init)
            "actor_checkpoints": dict(actor_config.get("actor_checkpoints", {})),
        })

        self.num_steps = 10

        # Camera configuration for multi-camera support
        if camera_config is None:
            camera_config = {}
        und_camera_keys = camera_config.get("und_camera_keys", ["head_rgb", "wrist_rgb"])
        wm_camera_key = camera_config.get("wm_camera_key", "head_rgb")
        
        # Convert to native Python types (omegaconf ListConfig -> list)
        if hasattr(und_camera_keys, '_content'):  # OmegaConf ListConfig
            und_camera_keys = list(und_camera_keys)
        wm_camera_idx = und_camera_keys.index(wm_camera_key) if wm_camera_key in und_camera_keys else 0
        
        # Auto-generate observation image keys from camera list
        # World model always uses image0 naming (dataset convention)
        self.camera_config = DictWithAttrAccess({
            "und_camera_keys": list(und_camera_keys),  # Ensure plain list
            "wm_camera_key": str(wm_camera_key),
            "wm_camera_idx": int(wm_camera_idx),
            "understanding_image_keys": [f"observation.images.image{i}" for i in range(len(und_camera_keys))],
            "world_model_input_key": "observation.images.image0_history",
            "world_model_target_key": "observation.images.image0_target",
        })

        # Memory configuration
        self.use_memory = use_memory
        if memory_config is None:
            memory_config = {}
        self.memory_config = DictWithAttrAccess({
            "memory_len": int(memory_config.get("memory_len", 4)),  # Number of memory slots
            "bptt_steps": int(memory_config.get("bptt_steps", 8)),  # BPTT truncation length
            "init_std": float(memory_config.get("init_std", 0.02)),  # Init std for memory params
            "tokenizer_max_length": int(memory_config.get("tokenizer_max_length", 512)),  # Extended tokenizer length for history
        })

        self.und_expert_config = und_expert_config
        if isinstance(self.und_expert_config, dict):
            und_expert_config["model_type"] = (
                und_expert_config["model_type"] if "model_type" in und_expert_config else "paligemma"
            )
            self.und_expert_config = CONFIG_MAPPING[und_expert_config["model_type"]](**und_expert_config)
        elif und_expert_config is None:
            self.und_expert_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152, # pi0
                    # "vocab_size": 257216, # paligemma
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )

        if self.use_world_model:    
            self.gen_expert_config = gen_expert_config
            if isinstance(self.gen_expert_config, dict):
                gen_expert_config["model_type"] = (
                    gen_expert_config["model_type"] if "model_type" in gen_expert_config else "paligemma"
                )
                self.gen_expert_config = CONFIG_MAPPING[gen_expert_config["model_type"]](**gen_expert_config)
            elif gen_expert_config is None:
                self.gen_expert_config = CONFIG_MAPPING["gemma"](
                        attention_bias=False,
                        attention_dropout=0.0,
                        bos_token_id=2,
                        eos_token_id=1,
                        head_dim=256,
                        hidden_act="gelu_pytorch_tanh",
                        hidden_activation="gelu_pytorch_tanh",
                        hidden_size=1024,
                        initializer_range=0.02,
                        intermediate_size=4096,
                        max_position_embeddings=8192,
                        model_type="gemma",
                        num_attention_heads=8,
                        num_hidden_layers=18,
                        num_key_value_heads=1,
                        pad_token_id=0,
                        rms_norm_eps=1e-06,
                        rope_theta=10000.0,
                        torch_dtype="float32",
                        transformers_version="4.48.1",
                        use_cache=True,
                        vocab_size=257152,
                )
            # Convert vae dict to object with attribute access if it exists and is a dict
            if hasattr(self.gen_expert_config, 'vae') and isinstance(self.gen_expert_config.vae, dict):
                vae_dict = self.gen_expert_config.vae
                self.gen_expert_config.vae = DictWithAttrAccess(vae_dict)
            elif not hasattr(self.gen_expert_config, 'vae'):
                # Only set default vae_dict if vae doesn't exist
                vae_dict = {
                    "vae_ckpt": None,
                    "vocab_size": 4096,
                    "z_channels": 32,
                    "ch": 160,
                    "test_mode": True,
                    "share_quant_resi": 4,
                }
                self.gen_expert_config.vae = DictWithAttrAccess(vae_dict)

        self.act_expert_config = act_expert_config
        if isinstance(self.act_expert_config, dict):
            act_expert_config["model_type"] = (
                act_expert_config["model_type"] if "model_type" in act_expert_config else "paligemma"
            )
            self.act_expert_config = CONFIG_MAPPING[act_expert_config["model_type"]](**act_expert_config)
        elif act_expert_config is None:
            self.act_expert_config = CONFIG_MAPPING["gemma"](
                    attention_bias=False,
                    attention_dropout=0.0,
                    bos_token_id=2,
                    eos_token_id=1,
                    head_dim=256,
                    hidden_act="gelu_pytorch_tanh",
                    hidden_activation="gelu_pytorch_tanh",
                    hidden_size=1024,
                    initializer_range=0.02,
                    intermediate_size=4096,
                    max_position_embeddings=8192,
                    model_type="gemma",
                    num_attention_heads=8,
                    num_hidden_layers=18,
                    num_key_value_heads=1,
                    pad_token_id=0,
                    rms_norm_eps=1e-06,
                    rope_theta=10000.0,
                    torch_dtype="float32",
                    transformers_version="4.48.1",
                    use_cache=True,
                    vocab_size=257152,
            )
        super().__init__(**kwargs)

    @property
    def ignore_index(self):
        warnings.warn(
            "The `ignore_index` attribute is deprecated and will be removed in v4.47.",
            FutureWarning,
        )
        return self._ignore_index

    @ignore_index.setter
    def ignore_index(self, value):
        self._ignore_index = value

    def to_dict(self):
        output = super().to_dict()
        output.pop("_ignore_index", None)
        return output
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if pretrained_model_name_or_path.endswith(".json"):
            config = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            config = super().from_pretrained(f"{pretrained_model_name_or_path}/config.json", **kwargs)
        return config