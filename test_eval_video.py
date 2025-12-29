#!/usr/bin/env python3
"""Test script to verify video generation functionality."""

import os
import sys
import torch
import random

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from omegaconf import OmegaConf
from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.processors.data_processors.data_loader import create_mekvm_data

def test_video_generation():
    print("=" * 60)
    print("Testing Video Generation")
    print("=" * 60)
    
    # Load config
    config_path = "f1_vla/config/memory_wm_only_resume.yaml"
    config = OmegaConf.load(config_path)
    
    print(f"\n[1] Loading config from {config_path}")
    print(f"    eval_episodes: {config.exp.training_args.get('eval_episodes', 'not set')}")
    
    # Load checkpoint - use the latest available
    checkpoint_path = config.exp.training_args.get('resume_from_checkpoint', 
                                                    'outputs/memory_wm_only_v2/checkpoint-episode-5000')
    # Check if the checkpoint exists, if not find the latest one
    if not os.path.exists(checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-episode-')]
            if checkpoints:
                # Sort by episode number and pick the latest
                checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"    Using latest checkpoint: {checkpoint_path}")
    print(f"\n[2] Loading model from {checkpoint_path}")
    
    # Load model config from checkpoint directory
    from f1_vla.src.models.configuration_f1 import F1Config
    policy_config = F1Config.from_pretrained(checkpoint_path)  # Load config from checkpoint
    policy_config.use_world_model = config.policy.use_world_model
    policy_config.use_memory = config.policy.get('use_memory', False)
    
    # Create mock training args for model initialization
    from dataclasses import dataclass, field
    @dataclass
    class MockTrainingArgs:
        seed: int = 42
        train_gen_expert_only: bool = True
        train_act_expert_only: bool = False
        train_state_proj: bool = True
        freeze_vision_encoder: bool = True
        freeze_gen_expert: bool = False
        und_expert_lr: float = 5e-5
        act_expert_lr: float = 0.0
        vision_encoder_lr: float = 0.0
        gen_expert_lr: float = 5e-5
    
    mock_args = MockTrainingArgs()
    policy = F1_VLA(config=policy_config, training_args=mock_args)
    
    # Load checkpoint weights
    import safetensors.torch
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    if os.path.exists(model_file):
        state_dict = safetensors.torch.load_file(model_file)
        policy.load_state_dict(state_dict, strict=False)
        print(f"    Loaded weights from {model_file}")
    else:
        print(f"    Warning: {model_file} not found, using random weights")
    policy = policy.to("cuda")
    policy.eval()
    print(f"    Model loaded successfully")
    
    # Load dataset (small subset)
    print(f"\n[3] Loading dataset")
    
    @dataclass
    class DatasetMockArgs:
        seed: int = 42
        image_transforms_enabled: bool = False
        image_transforms_type: list = field(default_factory=list)
        image_transforms_max_num_transforms: int = 0
        image_transforms_random_order: bool = False
    
    # Limit samples
    dataset_config = OmegaConf.to_container(config.dataset, resolve=True)
    dataset_config['max_train_samples'] = 100
    dataset_config = OmegaConf.create(dataset_config)
    
    (
        dataset,
        image_transforms,
        training_ds_sample_weights,
        cur_n_obs_img_steps,
        cur_n_pred_img_steps
    ) = create_mekvm_data(
        policy_config=policy_config,
        dataset_config=dataset_config,
        training_args=DatasetMockArgs(),
        stage=config.exp.stage,
    )
    
    print(f"    Dataset size: {len(dataset)}")
    print(f"    cur_n_obs_img_steps: {cur_n_obs_img_steps}")
    print(f"    cur_n_pred_img_steps: {cur_n_pred_img_steps}")
    
    # Test the video generation components
    print(f"\n[4] Testing forward_with_world_model with return_images=True")
    
    # Use collator to create proper batch with mask fields
    from f1_vla.src.processors.data_processors.me_kvm_dataset import MEKVMCollateFn
    collator = MEKVMCollateFn(
        max_state_dim=policy_config.max_state_dim,
        max_action_dim=policy_config.max_action_dim,
    )
    
    # Get a few samples and collate them
    samples = [dataset[i] for i in range(2)]
    batch = collator(samples)
    
    # Move to GPU
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to("cuda")
    
    print(f"    Batch keys: {list(batch.keys())[:8]}...")
    
    try:
        with torch.no_grad():
            outputs = policy.forward_with_world_model(
                batch,
                cur_n_obs_img_steps=cur_n_obs_img_steps,
                cur_n_pred_img_steps=cur_n_pred_img_steps,
                train_gen_expert_only=True,
                gen_out_loss_ratio=1.0,
                return_images=True,
            )
        
        print(f"    Output keys: {list(outputs.keys())}")
        
        if 'wm_gt_img' in outputs and 'wm_pred_img' in outputs:
            gt_img = outputs['wm_gt_img']
            pred_img = outputs['wm_pred_img']
            print(f"    ✓ Got images!")
            print(f"      GT shape: {gt_img.shape}")
            print(f"      Pred shape: {pred_img.shape}")
            
            # Test video creation with multiple samples
            print(f"\n[5] Testing video creation with multiple samples")
            import cv2
            import numpy as np
            
            eval_dir = "outputs/test_eval_videos"
            os.makedirs(eval_dir, exist_ok=True)
            
            # Collect frames from multiple batches
            all_frames = []
            num_test_batches = 5
            
            for batch_idx in range(num_test_batches):
                # Get different samples
                start_idx = batch_idx * 2
                samples = [dataset[i] for i in range(start_idx, start_idx + 2)]
                batch = collator(samples)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to("cuda")
                
                with torch.no_grad():
                    outputs = policy.forward_with_world_model(
                        batch,
                        cur_n_obs_img_steps=cur_n_obs_img_steps,
                        cur_n_pred_img_steps=cur_n_pred_img_steps,
                        train_gen_expert_only=True,
                        gen_out_loss_ratio=1.0,
                        return_images=True,
                    )
                
                gt_img = outputs['wm_gt_img'].cpu()
                pred_img = outputs['wm_pred_img'].cpu()
                
                # Process frames from this batch
                for b in range(gt_img.shape[0]):
                    gt_tensor = gt_img[b]  # [T, C, H, W] or [C, H, W]
                    pred_tensor = pred_img[b]
                    
                    if gt_tensor.dim() == 3:  # [C, H, W]
                        gt_tensor = gt_tensor.unsqueeze(0)
                        pred_tensor = pred_tensor.unsqueeze(0)
                    
                    for t in range(gt_tensor.shape[0]):
                        gt_frame = gt_tensor[t]
                        pred_frame = pred_tensor[t]
                        
                        gt_np = gt_frame.permute(1, 2, 0).numpy()
                        pred_np = pred_frame.permute(1, 2, 0).numpy()
                        
                        # Normalize to 0-255
                        gt_np = ((gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 255).astype(np.uint8)
                        pred_np = ((pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8) * 255).astype(np.uint8)
                        
                        # Add labels
                        gt_labeled = cv2.putText(gt_np.copy(), "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        pred_labeled = cv2.putText(pred_np.copy(), "Pred", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        combined = np.concatenate([gt_labeled, pred_labeled], axis=1)
                        all_frames.append(combined)
                
                print(f"    Batch {batch_idx + 1}/{num_test_batches}: collected {len(all_frames)} frames")
            
            if all_frames:
                # Save as AVI with XVID codec (more compatible)
                video_path = os.path.join(eval_dir, "test_eval_new.avi")
                h, w = all_frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(video_path, fourcc, 5, (w, h))
                
                for frame in all_frames:
                    if frame.shape[-1] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    writer.write(frame_bgr)
                
                writer.release()
                print(f"    ✓ Video saved to {video_path}")
                print(f"      Frames: {len(all_frames)}, Size: {w}x{h}")
                file_size = os.path.getsize(video_path)
                print(f"      File size: {file_size} bytes")
                
                # Also save individual frames as images for verification
                frame_dir = os.path.join(eval_dir, "frames")
                os.makedirs(frame_dir, exist_ok=True)
                for i, frame in enumerate(all_frames[:5]):  # Save first 5 frames
                    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"    ✓ Saved {min(5, len(all_frames))} sample frames to {frame_dir}/")
            else:
                print(f"    ✗ No frames generated")
        else:
            print(f"    ✗ Missing image keys!")
            print(f"      Available: {list(outputs.keys())}")
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_video_generation()
