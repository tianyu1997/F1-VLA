<div align="center">

# <img src="assets/logo.png" alt="F1 Logo" width="70" height="45"> F1-VLA

### A Vision-Language-Action Model with Memory-Augmented World Model and Explorer RL

[![Paper](https://img.shields.io/badge/Paper-arXiv-red.svg)](https://arxiv.org/abs/2509.06951)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-blue.svg)](https://aopolin-lv.github.io/F1-VLA)
[![Demo](https://img.shields.io/badge/Demo-YouTube-red.svg)](https://www.youtube.com/watch?v=wz-fOJU3FEM)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Training Modes](#-training-modes)
- [Configuration](#-configuration)
- [Model Components](#-model-components)
- [Evaluation](#-evaluation)
- [Citation](#-citation)

---

## 🎯 Overview

**F1-VLA** is a novel Vision-Language-Action model that integrates **visual foresight generation** into robot decision-making. This extended version includes:

- **KV Memory Bank**: Long-term memory for sequential reasoning
- **Teacher-Student Distillation**: Knowledge transfer with memory distillation
- **Explorer RL Training**: Adversarial exploration for world model improvement

We introduce $\mathcal{F}_1$, a novel paradigm by integrating **visual foresight generation** into the decision-making pipeline. Our model employs a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions through **predictive inverse dynamics modeling**.

<div align="center">
  <video src="https://github.com/user-attachments/assets/7d24ac5f-e8fa-4609-8731-2f36b64a9005"
         controls autoplay muted playsinline loop width="720"></video>
  
  <p><em>🏁 Best viewed with sound on</em></p>
</div>

---

## 🚀 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Predictive Inverse Dynamics** | Visual foresight generation for planning-based control |
| 🏗️ **Mixture-of-Transformer** | Three specialized experts (Understanding, Generation, Action) |
| 💾 **KV Memory Bank** | GRU-based memory for long-horizon reasoning |
| 🎓 **Teacher-Student Distillation** | Multi-camera to single-camera knowledge transfer |
| 🔍 **Explorer RL** | Adversarial training for robust world models |
| ⚡ **Multi-GPU Training** | Distributed training with torchrun |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              F1-VLA Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │
│   │   Vision    │    │  Language   │    │   Memory    │    │   Action    │ │
│   │  Encoder    │───▶│  Backbone   │───▶│    Bank     │───▶│   Expert    │ │
│   │ (SigLIP)    │    │ (PaliGemma) │    │  (KV+GRU)   │    │   (MoE)     │ │
│   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │
│         │                                      │                  │        │
│         │            ┌─────────────┐           │                  │        │
│         └───────────▶│ Generation  │◀──────────┘                  │        │
│                      │   Expert    │                              │        │
│                      │   (VAR)     │                              ▼        │
│                      └─────────────┘                     ┌─────────────┐   │
│                             │                            │   Robot     │   │
│                             ▼                            │   Actions   │   │
│                      ┌─────────────┐                     │   (7-DOF)   │   │
│                      │ World Model │                     └─────────────┘   │
│                      │  Prediction │                                       │
│                      └─────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

```
Stage 1: World Model Pretraining
    └── Train generation expert with KV Memory Bank
    
Stage 2: Teacher-Student Distillation  
    └── Teacher (multi-cam) → Student (single-cam) + Memory distillation

Stage 3: Explorer RL Training
    ├── Phase 1: Freeze WM, train Explorer with PPO
    └── Phase 2: Adversarial training (Explorer vs World Model)
```

---

## 📦 Installation

### Prerequisites

- Python ≥ 3.10
- PyTorch ≥ 2.6.0
- CUDA ≥ 12.4
- GPU Memory ≥ 24GB (48GB recommended for training)

### Setup

```bash
# Clone repository
git clone https://github.com/InternRobotics/F1-VLA.git
cd F1-VLA

# Create conda environment
conda create -n f1_vla python=3.10
conda activate f1_vla

# Install PyTorch with CUDA
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    torchcodec==0.2.1 --index-url https://download.pytorch.org/whl/cu124

# Install F1-VLA
cd f1_vla
pip install -e .

# Fix numpy version
pip install numpy==1.26.4
```

### Download Pretrained Models

| Model | Link | Description |
|-------|------|-------------|
| F1_pretrain | [HuggingFace](https://huggingface.co/InternRobotics/F1-VLA) | Pretrained F1-VLA model |
| PaliGemma-3B | [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224) | Language backbone |
| VAE | [vae_ch160v4096z32.pth](https://huggingface.co/FoundationVision/var) | VAR tokenizer |
| Pi0 Base | [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base) | Action expert |

---

## 🚀 Quick Start

### 1. World Model Training (with Memory)

```bash
# Using train.sh with auto GPU detection
./train.sh -a -c f1_vla/config/memory_from_f1pretrain.yaml

# Or specify GPUs manually
./train.sh -g 0,1,2,3 -c f1_vla/config/memory_from_f1pretrain.yaml
```

### 2. Teacher-Student Distillation

```bash
# Requires 2 GPUs (teacher + student on separate GPUs)
./train.sh -g 5,6 -c f1_vla/config/teacher_student_config.yaml -r ""
```

### 3. Explorer RL Training

```bash
# Phase 1: Train Explorer (freeze WM)
./train_explorer.sh -g 5 -p 1

# Phase 2: Adversarial training
./train_explorer.sh -g 5 -p 2
```

### Monitor Training

```bash
# View live logs
tail -f logs/latest_log.log

# Check GPU usage
watch -n 1 nvidia-smi

# Stop training
kill $(cat logs/train_pid.txt)
```

---

## 🎛️ Training Modes

### Mode 1: World Model Pretraining

Train the world model with KV Memory Bank for sequential prediction.

```yaml
# f1_vla/config/memory_from_f1pretrain.yaml
exp:
  stage: stage1_pretrain_wm
  use_memory: True
  memory_config:
    memory_len: 32      # Memory slots per layer
    bptt_steps: 4       # Gradient backprop steps
    init_std: 0.02      # Memory initialization
```

**GPU Requirements**: 4× 48GB GPUs (A6000/A100)

### Mode 2: Teacher-Student Distillation

Transfer knowledge from multi-camera teacher to single-camera student with memory distillation.

```yaml
# f1_vla/config/teacher_student_config.yaml
exp:
  teacher_student_config:
    use_split_gpu: True           # Separate GPUs for teacher/student
    teacher_device: "cuda:0"      # Teacher GPU
    student_device: "cuda:1"      # Student GPU
    memory_loss_weight: 0.1       # Memory distillation weight
    use_memory_distillation: True # Enable KV memory transfer
```

**Loss Function**: 
$$L_{total} = L_{GT} + \lambda \cdot L_{memory}$$

Where $L_{memory} = MSE(KV_{student}, KV_{teacher})$

**GPU Requirements**: 2× 48GB GPUs

### Mode 3: Explorer RL Training

Two-phase adversarial training for robust exploration.

#### Phase 1: Freeze World Model, Train Explorer

```yaml
phase1:
  enabled: true
  ppo:
    learning_rate: 3.0e-4
    gamma: 0.99           # Discount factor
    gae_lambda: 0.95      # GAE parameter
    clip_epsilon: 0.2     # PPO clip range
```

#### Phase 2: Adversarial Training

```yaml
phase2:
  enabled: true
  adversarial:
    wm_learning_rate: 1.0e-4
    explorer_learning_rate: 1.0e-4
    wm_updates_per_iter: 10
    explorer_updates_per_iter: 1
```

**Reward Function**:
$$R = \alpha \cdot r_{uncertainty} + \beta \cdot r_{MSE} + \gamma \cdot r_{MSE\_improve} + \epsilon \cdot r_{unc\_improve} - \delta \cdot |a|$$

---

## 📁 Project Structure

```
F1-VLA/
├── train.sh                    # Main training script
├── train_hf.py                 # HuggingFace trainer entry point
├── train_explorer.sh           # Explorer training script
│
├── f1_vla/
│   ├── config/                 # Configuration files
│   │   ├── memory_from_f1pretrain.yaml   # World Model training
│   │   ├── teacher_student_config.yaml   # Distillation training
│   │   └── explorer_train_config.yaml    # Explorer RL training
│   │
│   └── src/
│       ├── models/             # Model implementations
│       │   ├── modeling_f1.py          # F1-VLA main model
│       │   ├── configuration_f1.py     # Model configuration
│       │   ├── memory.py               # KV Memory Bank (GRU-based)
│       │   ├── explorer.py             # Explorer actor
│       │   ├── explorer_trainer.py     # Explorer RL trainer (PPO)
│       │   ├── adversarial_trainer.py  # Adversarial WM trainer
│       │   ├── vae_embedding.py        # VAR tokenizer
│       │   └── reward_computation.py   # Exploration reward
│       │
│       ├── policies/           # Policy wrappers
│       │   └── f1_policy.py            # F1_VLA policy class
│       │
│       ├── processors/         # Data and training processors
│       │   ├── data_processors/        # Dataset loaders
│       │   │   └── sequential_dataset.py  # Sequential BPTT dataset
│       │   └── train_processors/       # Training utilities
│       │       └── policy_trainer.py   # Custom trainer
│       │
│       └── utils/              # Utility functions
│
├── outputs/                    # Training outputs
│   ├── memory_from_f1pretrain/         # WM checkpoints
│   ├── teacher_student_wm/             # Distillation checkpoints
│   └── explorer_training/              # Explorer checkpoints
│
└── logs/                       # Training logs
    ├── latest_log.log          # Symlink to latest log
    └── train_pid.txt           # Training process ID
```

---

## ⚙️ Configuration Reference

### Memory Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_memory` | `False` | Enable KV Memory Bank |
| `memory_len` | `32` | Number of memory slots per layer |
| `bptt_steps` | `4` | BPTT truncation steps |
| `init_std` | `0.02` | Memory parameter initialization std |

### Training Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | `3e-5` | Learning rate |
| `per_device_train_batch_size` | `1` | Batch size per GPU |
| `gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `num_train_epochs` | `1000` | Maximum epochs |
| `max_steps` | `360000` | Maximum training steps |
| `bf16` | `True` | Use BF16 mixed precision |
| `save_episodes` | `240` | Save checkpoint every N episodes |
| `logging_episodes` | `50` | Log every N episodes |

### Explorer Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward.alpha` | `1.0` | Uncertainty reward weight |
| `reward.beta` | `1.0` | MSE reward weight |
| `reward.gamma` | `0.5` | MSE improvement weight |
| `reward.epsilon` | `0.1` | Uncertainty improvement weight |
| `reward.delta` | `0.01` | Action penalty weight |
| `phase1.ppo.gamma` | `0.99` | PPO discount factor |
| `phase1.ppo.clip_epsilon` | `0.2` | PPO clip range |

---

## 🔧 Model Components

### 1. KV Memory Bank

GRU-based memory bank for long-horizon sequential reasoning.

```python
from f1_vla.src.models.memory import KVMemoryBank

memory = KVMemoryBank(
    num_layers=26,        # Transformer layers
    num_kv_heads=8,       # KV attention heads
    head_dim=256,         # Head dimension
    hidden_size=2048,     # Model hidden size
    memory_len=32,        # Memory slots
    init_std=0.02,        # Initialization std
)

# Memory flow:
# 1. Get memory token for input
memory_token = memory.get_memory_token(batch_size, device, dtype)

# 2. Get previous memory state (or initial for frame_idx=0)
prev_memory = memory.get_memory(batch_keys, device, dtype)

# 3. Update memory after forward pass
memory.update_memory(batch_keys, new_memory_state)
```

### 2. Explorer Actor

RL-trained actor for exploration and information gain maximization.

```python
from f1_vla.src.models.explorer import ExplorerConfig, initialize_explorer

config = ExplorerConfig(
    random_init=True,                    # Random initialization
    freeze_world_model=True,             # Phase 1: freeze WM
    reward_uncertainty_weight=1.0,       # Uncertainty reward
    reward_mse_weight=1.0,               # MSE reward
)

# Initialize explorer in policy
initialize_explorer(policy, config)
```

### 3. Reward Computation

Multi-component reward for exploration training:

| Component | Formula | Description |
|-----------|---------|-------------|
| $r_1$ | $uncertainty_{t+1}$ | World model uncertainty |
| $r_2$ | $MSE(pred, gt)$ | Prediction error |
| $r_3$ | $MSE_{t+1} - MSE_{t+2}$ | MSE improvement |
| $r_4$ | $unc_{t+1} - unc_{t+2}$ | Uncertainty reduction |
| $r_5$ | $-\|a_t\|$ | Action penalty |

---

## 📊 Performance

### Real-World Robot Experiments

| Task | Platform | $\mathcal{F}_1$ | $\pi_0$ | Improvement |
|:----:|:--------:|:------:|:---:|:-----------:|
| Multi-task | Genie-1 | 82.2% | 65.2% | +17.0% |
| Adaptation | Franka | 66.7% | 53.3% | +13.4% |
| Long-horizon | ARX LIFT II | 40.0% | 0.0% | +40.0% |
| Dynamic Env | ARX LIFT II | 66.7% | 33.3% | +33.4% |

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce `batch_size`, increase `gradient_accumulation_steps`, enable `save_episodes` |
| NaN in memory | Check `init_std` setting, memory auto-resets on NaN detection |
| Slow data loading | Install FFmpeg + TorchCodec for video acceleration |
| CUDA errors | Ensure `CUDA_VISIBLE_DEVICES` matches config device indices |
| Checkpoint not found | Use `-r ""` flag to skip resume, or verify checkpoint path |

### GPU Memory Guidelines

| Training Mode | GPU Config | Minimum VRAM |
|--------------|------------|--------------|
| World Model | 4× GPU distributed | 4× 40GB |
| Teacher-Student | 2× GPU (split mode) | 2× 48GB |
| Explorer Phase 1 | 1× GPU | 1× 24GB |
| Explorer Phase 2 | 1× GPU | 1× 32GB |

---

## 📚 Citation

```bibtex
@article{f1_vla_2025,
  title={F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions},
  author={Qi Lv and Weijie Kong and Hao Li and Jia Zeng and Zherui Qiu and Delin Qu and 
          Haoming Song and Qizhi Chen and Xiang Deng and Michael Yu Wang and 
          Liqiang Nie and Jiangmiao Pang},
  eprint={2509.06951},
  archivePrefix={arXiv},
  year={2025},
  url={https://arxiv.org/abs/2509.06951}
}
```

---

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Lerobot](https://github.com/huggingface/lerobot)
- [Any4lerobot](https://github.com/Tavish9/any4lerobot/)
- [VAR](https://github.com/FoundationVision/VAR)
- [PaliGemma](https://github.com/google-research/big_vision)

---

<div align="center">
  <sub>Built with ❤️ by the F1-VLA Team</sub>
</div>
