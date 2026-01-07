import sys
import os
sys.path.append(os.getcwd())

import torch
from pathlib import Path
from omegaconf import OmegaConf
from f1_vla.src.models.configuration_f1 import F1Config, DictWithAttrAccess
from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.utils.utils import set_policy_config, set_camera_config

def main():
    print("Loading config...")
    config_path = "f1_vla/config/test_fast.yaml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        return

    config = OmegaConf.load(Path(config_path))
    
    print("Loading F1Config...")
    print(f"Loading from: {config.policy.ckpt_path}")
    policy_config = F1Config.from_pretrained(f"{config.policy.ckpt_path}")
    policy_config = set_policy_config(policy_config, config.policy)
    policy_config = set_camera_config(policy_config, config.exp)
    
    if config.exp.get('use_memory', False):
        mem_cfg = config.exp.memory_config
        policy_config.use_memory = True
        policy_config.memory_config = DictWithAttrAccess({
            "memory_len": int(mem_cfg.get('memory_len', 4)),
            "bptt_steps": int(mem_cfg.get('bptt_steps', 8)),
            "init_std": float(mem_cfg.get('init_std', 0.02)),
            "tokenizer_max_length": int(mem_cfg.get('tokenizer_max_length', 512)),
        })
        print("Memory config set.")

    print("Initializing Model...")
    model = F1_VLA(policy_config)
    print("Model initialized.")

    # Test from_pretrained
    pretrained_path = getattr(policy_config, "pretrained_path", "/mnt/data2/ty/F1-VLA/pi0")
    print(f"Testing from_pretrained: {pretrained_path}")
    if os.path.exists(pretrained_path):
        try:
            model_pretrained = F1_VLA.from_pretrained(pretrained_path, config=policy_config)
            print("from_pretrained successful!")
            del model_pretrained
        except Exception as e:
            print(f"from_pretrained failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Pretrained path not found: {pretrained_path}")

    if hasattr(model, 'model') and hasattr(model.model, 'paligemma_with_expert'):
        paligemma = model.model.paligemma_with_expert
        print(f"Has paligemma_with_expert: {type(paligemma)}")
        print(f"Supports GC: {getattr(paligemma, 'supports_gradient_checkpointing', 'Unknown')}")
        
        # Check if enabling works
        try:
            if hasattr(paligemma, 'gradient_checkpointing_enable'):
                paligemma.gradient_checkpointing_enable()
                print("Gradient Checkpointing enabled on paligemma_with_expert.")
            else:
                 print("paligemma_with_expert has no gradient_checkpointing_enable method")
        except Exception as e:
            print(f"Failed to enable GC: {e}")

    print("Loading Data (Sequential)...")
    from f1_vla.src.processors.data_processors.sequential_dataset import create_sequential_mekvm_data
    
    # Mock training args
    args_dict = OmegaConf.to_container(config.exp.training_args, resolve=True)
    class MockArgs:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    training_args = MockArgs(**args_dict)

    try:
        (
            train_dataset,
            image_transforms,
            weights,
            obs_steps,
            pred_steps
        ) = create_sequential_mekvm_data(
            policy_config=policy_config,
            dataset_config=config.dataset,
            training_args=training_args,
            stage=config.exp.stage,
            rank=0,
            world_size=1
        )
        print(f"Data loaded: {len(train_dataset)} frames/sequences")
        
        from f1_vla.src.processors.data_processors.sequential_dataset import SequentialBatchSampler
        sampler = SequentialBatchSampler(train_dataset, batch_size=4, drop_last=False, rank=0, world_size=1)
        print("Sampler created.")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        import traceback
        traceback.print_exc()

    print("Setup Check Passed!")

if __name__ == "__main__":
    main()
