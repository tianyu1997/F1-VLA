# F1-VLA Complete Training Guide

Complete guide for training F1-VLA models including World Model pretraining, Teacher-Student distillation, and Explorer RL training.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [World Model Training](#3-world-model-training)
4. [Teacher-Student Distillation](#4-teacher-student-distillation)
5. [Explorer RL Training](#5-explorer-rl-training)
6. [Monitoring & Debugging](#6-monitoring--debugging)
7. [Checkpoint Management](#7-checkpoint-management)
8. [Advanced Configuration](#8-advanced-configuration)
9. [Best Practices](#9-best-practices)

---

## 1. Environment Setup

### 1.1 Hardware Requirements

| Training Mode | GPUs | VRAM per GPU | Total VRAM |
|--------------|------|--------------|------------|
| World Model (distributed) | 4 | 40GB+ | 160GB |
| Teacher-Student | 2 | 48GB+ | 96GB |
| Explorer Phase 1 | 1 | 24GB+ | 24GB |
| Explorer Phase 2 | 1 | 32GB+ | 32GB |

**Recommended**: NVIDIA A6000 (48GB) or A100 (40GB/80GB)

### 1.2 Software Installation

```bash
# Create conda environment
conda create -n f1_vla python=3.10
conda activate f1_vla

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

# Install F1-VLA package
cd f1_vla
pip install -e .

# Fix numpy version (important for compatibility)
pip install numpy==1.26.4

# Optional: Install FFmpeg for faster video loading
# Ubuntu/Debian
sudo apt-get install ffmpeg
# macOS
brew install ffmpeg
```

### 1.3 Verify Installation

```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check F1-VLA
python -c "from f1_vla.src.models.modeling_f1 import F1ForConditionalGeneration; print('F1-VLA OK')"

# Check Memory module
python -c "from f1_vla.src.models.memory import KVMemoryBank; print('Memory OK')"
```

---

## 2. Data Preparation

### 2.1 Data Format

F1-VLA supports two data formats:

#### LeRobot Format (Standard)
```
dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   └── tasks.jsonl
└── videos/
    └── chunk-000/
        ├── observation.images.head_000000.mp4
        └── observation.images.wrist_000000.mp4
```

#### ME-KVM Format (Extended)
```
data/
├── clean/
│   ├── episode_000/
│   │   ├── head/
│   │   │   └── frame_0000.jpg
│   │   ├── wrist/
│   │   │   └── frame_0000.jpg
│   │   ├── actions.npy
│   │   └── states.npy
│   └── ...
└── noisy/
    └── ...
```

### 2.2 Data Configuration

```yaml
# In your config.yaml
dataset:
  # For ME-KVM format
  use_mekvm_format: True
  mekvm_data_dirs:
    - /path/to/data/clean
    - /path/to/data/noisy  # Optional
  mekvm_task_descriptions:
    - "Task description for clean data"
    - "Task description for noisy data"
  mekvm_weights:
    - 0.7  # 70% clean
    - 0.3  # 30% noisy
  
  # Observation settings
  n_obs_img_steps: 4      # History length (L)
  n_pred_img_steps: 1     # Prediction horizon
  chunk_size: 4           # Action chunk size
  
  # Image settings
  image_size:
    height: 224
    width: 224
```

### 2.3 Download Pretrained Models

```bash
# Create model directories
mkdir -p F1_pretrain paligemma-3b-pt-224 var pi0

# Download F1_pretrain (HuggingFace)
huggingface-cli download InternRobotics/F1-VLA --local-dir F1_pretrain

# Download PaliGemma
huggingface-cli download google/paligemma-3b-pt-224 --local-dir paligemma-3b-pt-224

# Download VAE
wget https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth -P var/

# Download Pi0
huggingface-cli download lerobot/pi0_base --local-dir pi0
```

---

## 3. World Model Training

### 3.1 Configuration

Create or modify `f1_vla/config/memory_from_f1pretrain.yaml`:

```yaml
dataset:
  use_mekvm_format: True
  mekvm_data_dirs:
    - /path/to/your/data
  n_obs_img_steps: 4
  n_pred_img_steps: 1

exp:
  stage: stage1_pretrain_wm
  load_ckpt: /path/to/F1_pretrain
  
  # Memory Configuration
  use_memory: True
  memory_config:
    memory_len: 32        # Memory slots per layer
    bptt_steps: 4         # BPTT truncation
    init_std: 0.02        # Initialization std
  
  # VAE Configuration (freeze for WM training)
  vae_config:
    test_mode: True       # Freeze VAE completely
    freeze_encoder: True
    pixel_loss_weight: 0.0
  
  training_args:
    output_dir: outputs/world_model
    num_train_epochs: 1000
    max_steps: 360000
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    learning_rate: 3.0e-5
    warmup_steps: 3000
    bf16: True
    
    # Episode-based saving (prevents OOM)
    save_episodes: 240    # Save every 240 episodes (~1 epoch)
    logging_episodes: 50  # Log every 50 episodes
```

### 3.2 Launch Training

```bash
# Method 1: Auto-detect free GPUs
./train.sh -a -c f1_vla/config/memory_from_f1pretrain.yaml

# Method 2: Specify GPUs manually
./train.sh -g 0,1,2,3 -c f1_vla/config/memory_from_f1pretrain.yaml

# Method 3: Resume from checkpoint
./train.sh -g 0,1,2,3 -c f1_vla/config/memory_from_f1pretrain.yaml \
    -r outputs/world_model/checkpoint-episode-50000
```

### 3.3 Training Script Options

```bash
./train.sh [OPTIONS]

Options:
  -c, --config FILE     Config file path
  -g, --gpus IDS        GPU IDs (comma-separated, e.g., "0,1,2,3")
  -a, --auto            Auto-detect free GPUs (memory < 2GB)
  -m, --max-gpus N      Max GPUs in auto mode (default: 4)
  -r, --resume PATH     Resume from checkpoint
  -p, --port PORT       Master port (default: 29500)
  -h, --help            Show help
```

### 3.4 Expected Output

```
==========================================
F1-VLA Training (torchrun)
==========================================
Config: f1_vla/config/memory_from_f1pretrain.yaml
GPUs: 0,1,2,3 (4 GPUs)
Mode: Manual
Log file: logs/train_20260106_150439.log
==========================================

Training started with PID: 12345
Monitor:  tail -f logs/latest_log.log
Stop:     kill 12345
```

### 3.5 Training Progress Monitoring

```
[Episode 50/1440] Loss: 2.1234, WM_Loss: 2.0567, Memory: 45.2GB
[Episode 100/1440] Loss: 1.9876, WM_Loss: 1.9234, Memory: 45.1GB
...
[Epoch 1 Complete] Avg Loss: 1.8765, Time: 2.5h
Checkpoint saved: outputs/world_model/checkpoint-episode-1440
```

---

## 4. Teacher-Student Distillation

### 4.1 Overview

Teacher-Student training transfers knowledge from a multi-camera model (teacher) to a single-camera model (student) with memory distillation.

```
┌──────────────────────────────────────────────────────────────┐
│                 Teacher-Student Architecture                  │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   Teacher (GPU 0)              Student (GPU 1)               │
│   ┌─────────────┐              ┌─────────────┐               │
│   │ Head + Wrist│              │ Wrist Only  │               │
│   │   Cameras   │              │   Camera    │               │
│   └──────┬──────┘              └──────┬──────┘               │
│          │                            │                      │
│          ▼                            ▼                      │
│   ┌─────────────┐              ┌─────────────┐               │
│   │   Frozen    │──────────────│  Trainable  │               │
│   │   Model     │   KV Distill │    Model    │               │
│   └─────────────┘              └─────────────┘               │
│                                                              │
│   Loss = GT_loss + λ × MSE(KV_student, KV_teacher)          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 Configuration

```yaml
# f1_vla/config/teacher_student_config.yaml

exp:
  stage: stage_teacher_student
  
  # Model checkpoints
  teacher_ckpt: /path/to/trained/world_model/checkpoint
  student_ckpt: /path/to/trained/world_model/checkpoint  # Can start from same
  
  # Memory configuration (ENABLED)
  use_memory: True
  memory_config:
    memory_len: 32
    bptt_steps: 4
    init_std: 0.02
  
  # Teacher-Student specific config
  teacher_student_config:
    use_teacher_student: False   # Not standard mode
    use_student_only: False      # Not control group
    use_split_gpu: True          # RECOMMENDED: Separate GPUs
    
    # Device assignment (indices within CUDA_VISIBLE_DEVICES)
    teacher_device: "cuda:0"     # First visible GPU
    student_device: "cuda:1"     # Second visible GPU
    
    # Memory distillation
    memory_loss_weight: 0.1      # λ for memory distillation
    use_memory_distillation: True
  
  training_args:
    output_dir: outputs/teacher_student
    per_device_train_batch_size: 1
    learning_rate: 3.0e-5
    bf16: True
```

### 4.3 Launch Training

```bash
# Two GPUs required (teacher + student)
./train.sh -g 5,6 -c f1_vla/config/teacher_student_config.yaml -r ""

# The -r "" flag starts fresh (no resume)
```

### 4.4 Memory Distillation Details

The memory distillation loss encourages the student to learn similar KV cache states as the teacher:

```python
# Pseudo-code for memory distillation
teacher_kv = teacher.get_past_key_values()[-memory_len:]  # Last N memory states
student_kv = student.get_past_key_values()[-memory_len:]

# Detach teacher (no gradient)
teacher_kv = teacher_kv.detach()

# MSE loss on KV states
memory_loss = F.mse_loss(student_kv, teacher_kv)

# Total loss
total_loss = gt_loss + memory_loss_weight * memory_loss
```

### 4.5 Typical Training Output

```
[Teacher-Student Split GPU Mode]
  Teacher Device: cuda:0 (GPU 5)
  Student Device: cuda:1 (GPU 6)
  Memory Distillation: Enabled (weight=0.1)

[Episode 50] 
  GT Loss: 2.1234
  Memory Loss: 1.9876
  Total Loss: 2.3222 (= 2.1234 + 0.1 × 1.9876)
```

---

## 5. Explorer RL Training

### 5.1 Overview

Explorer training uses reinforcement learning to train an actor that explores states where the World Model makes errors, improving model robustness.

### 5.2 Phase 1: Train Explorer (Freeze WM)

In Phase 1, the World Model is frozen and only the Explorer actor is trained using PPO.

```yaml
# f1_vla/config/explorer_train_config.yaml

phase1:
  enabled: true
  
  ppo:
    learning_rate: 3.0e-4
    gamma: 0.99           # Discount factor
    gae_lambda: 0.95      # GAE parameter
    clip_epsilon: 0.2     # PPO clip range
    value_coef: 0.5       # Value loss weight
    entropy_coef: 0.01    # Entropy bonus
    max_grad_norm: 0.5    # Gradient clipping
  
  training:
    total_timesteps: 100000
    num_envs: 4           # Parallel environments
    steps_per_rollout: 256
    num_epochs: 4         # PPO epochs per update
    batch_size: 64
```

### 5.3 Phase 2: Adversarial Training

In Phase 2, the Explorer and World Model are trained adversarially:
- Explorer tries to find states that maximize WM error
- World Model tries to minimize prediction error on those states

```yaml
phase2:
  enabled: true
  
  adversarial:
    # World Model training
    wm_learning_rate: 1.0e-4
    wm_updates_per_iter: 10
    
    # Explorer training
    explorer_learning_rate: 1.0e-4
    explorer_updates_per_iter: 1
    
    # Warmup (WM only)
    warmup_iterations: 100
    
    # Mode collapse prevention
    collapse_threshold: 0.1
  
  training:
    total_iterations: 1000
    steps_per_iteration: 256
```

### 5.4 Reward Function

The exploration reward combines multiple components:

| Component | Weight | Formula | Purpose |
|-----------|--------|---------|---------|
| $r_1$ | α=1.0 | $uncertainty_{t+1}$ | High uncertainty states |
| $r_2$ | β=1.0 | $MSE(pred, gt)$ | Prediction errors |
| $r_3$ | γ=0.5 | $MSE_{t+1} - MSE_{t+2}$ | Improving predictions |
| $r_4$ | ε=0.1 | $unc_{t+1} - unc_{t+2}$ | Reducing uncertainty |
| $r_5$ | δ=0.01 | $-\|a_t\|$ | Action regularization |

```python
R_total = α*r1 + β*r2 + γ*r3 + ε*r4 - δ*r5
```

### 5.5 Launch Explorer Training

```bash
# Phase 1 only
./train_explorer.sh -g 5 -p 1

# Phase 2 only (after Phase 1)
./train_explorer.sh -g 5 -p 2

# Both phases
./train_explorer.sh -g 5 -a
```

### 5.6 Explorer Training Output

```
================================================================================
Explorer RL Training - Phase 1
================================================================================
World Model: FROZEN
Explorer: TRAINABLE (random init)

[Timestep 1000] Reward: 0.234, Policy Loss: 0.045, Value Loss: 0.123
[Timestep 2000] Reward: 0.456, Policy Loss: 0.038, Value Loss: 0.098
...

================================================================================
Explorer RL Training - Phase 2
================================================================================
Mode: ADVERSARIAL

[Iteration 100] 
  Explorer Reward: 0.789
  WM Loss: 1.234
  Mode Collapse Risk: 0.05 (OK)
```

---

## 6. Monitoring & Debugging

### 6.1 Log Monitoring

```bash
# Real-time log viewing
tail -f logs/latest_log.log

# Filter for specific metrics
tail -f logs/latest_log.log | grep "loss"
tail -f logs/latest_log.log | grep "episode"
tail -f logs/latest_log.log | grep "memory"

# View last N lines
tail -100 logs/latest_log.log
```

### 6.2 GPU Monitoring

```bash
# Continuous GPU monitoring
watch -n 1 nvidia-smi

# Check specific GPUs
nvidia-smi -i 0,1,2,3

# Memory usage only
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
```

### 6.3 Training Metrics

Key metrics to monitor:

| Metric | Description | Expected Range |
|--------|-------------|----------------|
| `loss` | Total training loss | Decreasing trend |
| `wm_loss` | World model loss | 0.5 - 2.0 |
| `memory_loss` | Memory distillation loss | 1.0 - 3.0 |
| `learning_rate` | Current LR | Per schedule |
| `grad_norm` | Gradient norm | < max_grad_norm |
| `episode` | Current episode | Increasing |

### 6.4 Common Issues & Solutions

#### OOM (Out of Memory)

```yaml
# Solution 1: Reduce batch size, increase accumulation
per_device_train_batch_size: 1
gradient_accumulation_steps: 16

# Solution 2: Enable episode-based saving
save_episodes: 120  # More frequent saves

# Solution 3: Reduce memory length
memory_config:
  memory_len: 16  # Reduced from 32
```

#### NaN in Loss

```yaml
# Solution: Check memory initialization
memory_config:
  init_std: 0.02  # Not too large

# Solution: Gradient clipping
max_grad_norm: 1.0

# Solution: Lower learning rate
learning_rate: 1.0e-5
```

#### Slow Data Loading

```bash
# Install video acceleration
pip install torchcodec ffmpeg-python

# Increase workers
dataloader_num_workers: 8
dataloader_prefetch_factor: 8
```

### 6.5 Debug Mode

For debugging, use debug config:

```bash
./train.sh -g 0 -c f1_vla/config/debug_test.yaml
```

Debug config settings:
```yaml
exp:
  training_args:
    max_steps: 100           # Quick test
    logging_episodes: 1      # Log every episode
    save_episodes: 10        # Frequent saves
    per_device_train_batch_size: 1
```

---

## 7. Checkpoint Management

### 7.1 Checkpoint Structure

```
outputs/world_model/
├── checkpoint-episode-50000/
│   ├── config.json           # Model configuration
│   ├── model.safetensors     # Model weights
│   ├── optimizer.pt          # Optimizer state
│   ├── scheduler.pt          # LR scheduler state
│   ├── trainer_state.json    # Training state
│   └── rng_state.pth         # Random state
├── checkpoint-episode-100000/
└── ...
```

### 7.2 Resume Training

```bash
# Resume from specific checkpoint
./train.sh -g 0,1,2,3 -c config.yaml \
    -r outputs/world_model/checkpoint-episode-50000

# Resume from latest (auto-detected)
./train.sh -g 0,1,2,3 -c config.yaml \
    -r outputs/world_model/

# Start fresh (no resume)
./train.sh -g 0,1,2,3 -c config.yaml -r ""
```

### 7.3 Checkpoint Cleanup

```bash
# Keep only last N checkpoints
python -c "
import os
import shutil
from pathlib import Path

output_dir = Path('outputs/world_model')
checkpoints = sorted(output_dir.glob('checkpoint-episode-*'), 
                     key=lambda x: int(x.name.split('-')[-1]))

# Keep last 3
for ckpt in checkpoints[:-3]:
    shutil.rmtree(ckpt)
    print(f'Removed: {ckpt}')
"
```

### 7.4 Export Model

```bash
# Convert checkpoint to HuggingFace format
python -c "
from f1_vla.src.policies.f1_policy import F1_VLA

# Load checkpoint
policy = F1_VLA.from_checkpoint('outputs/world_model/checkpoint-episode-100000')

# Save as HuggingFace model
policy.save_pretrained('exported_model/')
"
```

### 7.5 Checkpoint Size Estimation

| Component | Size |
|-----------|------|
| Model weights | ~12GB |
| Optimizer state | ~24GB |
| Total checkpoint | ~40GB |

---

## 8. Advanced Configuration

### 8.1 Multi-Dataset Training

```yaml
dataset:
  use_mekvm_format: True
  mekvm_data_dirs:
    - /data/clean_robotwin
    - /data/clean_libero
    - /data/noisy_robotwin
  mekvm_task_descriptions:
    - "RoboTwin clean manipulation tasks"
    - "LIBERO clean manipulation tasks"
    - "RoboTwin noisy manipulation tasks"
  mekvm_weights:
    - 0.4  # 40% RoboTwin clean
    - 0.4  # 40% LIBERO clean
    - 0.2  # 20% RoboTwin noisy
```

### 8.2 Learning Rate Schedules

```yaml
# Cosine with warmup (recommended)
lr_scheduler_type: cosine
warmup_steps: 3000
learning_rate: 3.0e-5

# Linear decay
lr_scheduler_type: linear
warmup_steps: 1000
learning_rate: 5.0e-5

# Constant with warmup
lr_scheduler_type: constant_with_warmup
warmup_steps: 500
learning_rate: 1.0e-5
```

### 8.3 Mixed Precision Training

```yaml
# BF16 (recommended for A100/A6000)
bf16: True
fp16: False

# FP16 (for older GPUs)
bf16: False
fp16: True

# Full precision (debugging)
bf16: False
fp16: False
```

### 8.4 Distributed Training Options

```bash
# Single node, multi-GPU (default)
./train.sh -g 0,1,2,3 -c config.yaml

# Custom master port (if default conflicts)
./train.sh -g 0,1,2,3 -c config.yaml -p 29501

# Multi-node training (advanced)
torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.1 \
    --master_port=29500 \
    train_hf.py --config config.yaml
```

### 8.5 Memory-Efficient Options

```yaml
# Gradient checkpointing
gradient_checkpointing: True

# 8-bit optimizer
optim: adamw_bnb_8bit

# Reduce memory length for limited VRAM
memory_config:
  memory_len: 16  # Instead of 32
  bptt_steps: 2   # Instead of 4
```

---

## 9. Best Practices

### 9.1 Training Workflow

```
1. Prepare Data
   ├── Convert to ME-KVM format
   ├── Verify data integrity
   └── Split train/val

2. Test Configuration
   ├── Run debug config first
   ├── Check single batch
   └── Verify GPU usage

3. Training
   ├── Start with World Model
   ├── Monitor first 1000 steps
   └── Regular checkpoints

4. Evaluation
   ├── Validate every few epochs
   └── Track key metrics

5. Fine-tuning
   ├── Teacher-Student distillation
   └── Explorer RL training
```

### 9.2 Hyperparameter Tuning

| Parameter | Start Value | Tuning Range | Notes |
|-----------|-------------|--------------|-------|
| `learning_rate` | 3e-5 | 1e-5 to 1e-4 | Lower for fine-tuning |
| `memory_len` | 32 | 8 to 64 | Higher = more memory |
| `bptt_steps` | 4 | 2 to 8 | Higher = slower |
| `memory_loss_weight` | 0.1 | 0.01 to 0.5 | Balance with GT loss |
| `batch_size` | 1 | 1 to 4 | Limited by VRAM |
| `gradient_accumulation` | 8 | 4 to 16 | Effective batch size |

### 9.3 Memory Management

```python
# In training loop, periodically clear cache
if step % 100 == 0:
    torch.cuda.empty_cache()
    
# Monitor memory usage
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Memory bank cleanup at epoch boundaries
if is_epoch_end:
    model.memory_bank.clear_inactive_episodes()
```

### 9.4 Reproducibility

```yaml
# Set seed for reproducibility
seed: 42

# Deterministic algorithms (slower but reproducible)
torch_compile: False
dataloader_drop_last: True

# Environment variables
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 9.5 Debugging Checklist

- [ ] Check GPU memory before training (`nvidia-smi`)
- [ ] Verify data paths exist and are readable
- [ ] Test with single batch first (`max_steps: 10`)
- [ ] Monitor loss in first 100 steps (should decrease)
- [ ] Check gradient norms (should be < max_grad_norm)
- [ ] Verify checkpoint saves work
- [ ] Test resume functionality
- [ ] Check memory bank is updating (log memory size)

### 9.6 Production Training Timeline

| Stage | Duration | Output |
|-------|----------|--------|
| World Model (360K steps) | 3-5 days | Base model |
| Teacher-Student | 1-2 days | Distilled model |
| Explorer Phase 1 | 1 day | Exploration policy |
| Explorer Phase 2 | 1-2 days | Robust model |
| **Total** | **6-10 days** | Final model |

---

## Appendix

### A. Complete Configuration Example

```yaml
# Complete training configuration
dataset:
  use_mekvm_format: True
  mekvm_data_dirs:
    - /mnt/data/clean
  mekvm_task_descriptions:
    - "Robot manipulation task"
  mekvm_weights:
    - 1.0
  n_obs_img_steps: 4
  n_pred_img_steps: 1
  chunk_size: 4
  image_size:
    height: 224
    width: 224

exp:
  stage: stage1_pretrain_wm
  load_ckpt: /path/to/F1_pretrain
  
  use_memory: True
  memory_config:
    memory_len: 32
    bptt_steps: 4
    init_std: 0.02
    tokenizer_max_length: 512
  
  vae_config:
    test_mode: True
    freeze_encoder: True
    pixel_loss_weight: 0.0
  
  training_args:
    output_dir: outputs/training
    run_name: f1_vla_training
    do_train: True
    do_eval: False
    num_train_epochs: 1000
    max_steps: 360000
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    
    optim: adamw_bnb_8bit
    learning_rate: 3.0e-5
    weight_decay: 1.0e-4
    max_grad_norm: 1.0
    adam_beta1: 0.9
    adam_beta2: 0.999
    adam_epsilon: 1.0e-8
    lr_scheduler_type: cosine
    warmup_steps: 3000
    
    bf16: True
    seed: 42
    
    logging_episodes: 50
    save_episodes: 240
    save_total_limit: 5
    
    dataloader_num_workers: 8
    dataloader_pin_memory: False
    dataloader_persistent_workers: True
    dataloader_prefetch_factor: 8
```

### B. Environment Variables

```bash
# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=0  # Set to 1 for debugging

# Tokenizers
export TOKENIZERS_PARALLELISM=false

# PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable W&B (optional)
export WANDB_DISABLED=true
```

### C. Useful Commands

```bash
# Kill training process
kill $(cat logs/train_pid.txt)

# Check disk usage
du -sh outputs/*/

# Count checkpoints
ls -d outputs/*/checkpoint-* | wc -l

# Find latest checkpoint
ls -td outputs/*/checkpoint-* | head -1

# Check training status
ps aux | grep train_hf.py

# View GPU processes
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

### D. Config File Reference

| Config File | Purpose | GPUs |
|-------------|---------|------|
| `memory_from_f1pretrain.yaml` | World Model with Memory | 4 |
| `teacher_student_config.yaml` | Teacher-Student Distillation | 2 |
| `explorer_train_config.yaml` | Explorer RL Training | 1 |
| `debug_test.yaml` | Quick debugging | 1 |
| `memory_wm_clean_only.yaml` | Clean data only training | 4 |

---

*Last updated: January 2026*
