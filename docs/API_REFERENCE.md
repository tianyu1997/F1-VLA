# F1-VLA API Reference

Complete API documentation for F1-VLA model components.

---

## Table of Contents

1. [Model Classes](#1-model-classes)
2. [Memory Module](#2-memory-module)
3. [Explorer Module](#3-explorer-module)
4. [Training Components](#4-training-components)
5. [Data Processing](#5-data-processing)
6. [Configuration](#6-configuration)

---

## 1. Model Classes

### F1ForConditionalGeneration

Main model class for F1-VLA.

```python
from f1_vla.src.models.modeling_f1 import F1ForConditionalGeneration

model = F1ForConditionalGeneration(config)
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `forward()` | Forward pass with optional memory | `ModelOutput` |
| `generate()` | Generate tokens autoregressively | `torch.Tensor` |
| `get_action()` | Get action from observation | `torch.Tensor` |

#### Forward Parameters

```python
output = model.forward(
    input_ids: torch.LongTensor,              # Token IDs
    pixel_values: torch.FloatTensor,          # Image inputs
    attention_mask: torch.Tensor = None,      # Attention mask
    position_ids: torch.LongTensor = None,    # Position IDs
    past_key_values: tuple = None,            # KV cache
    inputs_embeds: torch.FloatTensor = None,  # Direct embeddings
    labels: torch.LongTensor = None,          # Labels for loss
    use_cache: bool = None,                   # Enable KV caching
    output_attentions: bool = None,           # Output attention weights
    output_hidden_states: bool = None,        # Output hidden states
    return_dict: bool = None,                 # Return dict or tuple
    
    # Memory-specific
    memory_state: List[Tuple] = None,         # Previous memory
    frame_indices: torch.LongTensor = None,   # Frame indices for memory
    episode_ids: List[Tuple] = None,          # Episode identifiers
)
```

### F1_VLA (Policy Wrapper)

High-level policy class for training and inference.

```python
from f1_vla.src.policies.f1_policy import F1_VLA

policy = F1_VLA(config)
policy = F1_VLA.from_pretrained("path/to/checkpoint")
```

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `from_pretrained()` | Load from checkpoint | `F1_VLA` |
| `save_pretrained()` | Save to directory | `None` |
| `get_action()` | Get action for observation | `torch.Tensor` |
| `train_step()` | Execute training step | `dict` |
| `list_actors()` | List available actors | `List[str]` |
| `set_active_actor()` | Set active actor | `None` |

---

## 2. Memory Module

### KVMemoryBank

GRU-based Key-Value memory bank for long-horizon reasoning.

```python
from f1_vla.src.models.memory import KVMemoryBank

memory = KVMemoryBank(
    num_layers: int = 26,         # Number of transformer layers
    num_kv_heads: int = 8,        # Number of KV attention heads
    head_dim: int = 256,          # Dimension per head
    hidden_size: int = 2048,      # Model hidden size
    memory_len: int = 32,         # Memory slots per layer
    init_std: float = 0.02,       # Initialization std
)
```

#### Methods

```python
# Get memory token for input
memory_token = memory.get_memory_token(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor  # Shape: (batch, 1, hidden_size)

# Get initial memory (for frame_idx=0)
init_memory = memory.get_initial_memory(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]

# Get memory for batch
memory_state = memory.get_memory(
    batch_keys: List[Tuple[int, int]],  # (dataset_idx, episode_idx)
    device: torch.device,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]

# Update memory after forward
memory.update_memory(
    batch_keys: List[Tuple[int, int]],
    new_memory: List[Tuple[torch.Tensor, torch.Tensor]],
    detach: bool = True,  # Detach for BPTT truncation
)

# Clear memory bank
memory.clear_memory_bank()

# Clear specific episodes
memory.clear_episodes(episode_keys: List[Tuple[int, int]])
```

#### Memory State Format

```python
# Memory state is a list of (key, value) tuples, one per layer
# Each tensor has shape: (batch, memory_len, num_kv_heads, head_dim)

memory_state = [
    (key_layer_0, value_layer_0),  # Layer 0
    (key_layer_1, value_layer_1),  # Layer 1
    ...
    (key_layer_N, value_layer_N),  # Layer N
]

# Example shapes with default config:
# key: (batch, 32, 8, 256)
# value: (batch, 32, 8, 256)
```

---

## 3. Explorer Module

### ExplorerConfig

Configuration for Explorer actor.

```python
from f1_vla.src.models.explorer import ExplorerConfig

config = ExplorerConfig(
    # Initialization
    random_init: bool = True,          # Random init vs copy from actor
    actor_checkpoint: str = None,      # Optional checkpoint
    
    # Reward weights
    reward_uncertainty_weight: float = 1.0,     # r1
    reward_mse_weight: float = 1.0,             # r2
    reward_mse_improvement_weight: float = 1.0, # r3
    reward_uncertainty_improvement_weight: float = 0.5,  # r4
    reward_action_penalty_weight: float = 0.01, # r5
    
    # Training flags
    freeze_world_model: bool = True,   # Freeze WM in Phase 1
    freeze_actor: bool = True,         # Freeze policy actor
)
```

### initialize_explorer

Initialize Explorer in policy model.

```python
from f1_vla.src.models.explorer import initialize_explorer

initialize_explorer(
    policy: F1_VLA,
    explorer_config: ExplorerConfig = None,
    device: torch.device = None,
)
```

### ExplorerRLTrainer

PPO-based trainer for Explorer.

```python
from f1_vla.src.models.explorer_trainer import ExplorerRLTrainer

trainer = ExplorerRLTrainer(
    policy: F1_VLA,
    config: dict,                    # Training config
    env: gym.Env = None,             # Environment
    device: torch.device = None,
)
```

#### Methods

```python
# Collect rollouts
rollouts = trainer.collect_rollouts(
    num_steps: int = 256,
) -> RolloutBuffer

# Train on rollouts
metrics = trainer.train_step(
    rollouts: RolloutBuffer,
) -> dict

# Full training loop
trainer.train(
    total_timesteps: int = 100000,
)
```

### AdversarialExplorerTrainer

Adversarial trainer for Phase 2.

```python
from f1_vla.src.models.adversarial_trainer import AdversarialExplorerTrainer

trainer = AdversarialExplorerTrainer(
    policy: F1_VLA,
    config: dict,
    env: gym.Env = None,
)
```

#### Methods

```python
# Single adversarial iteration
metrics = trainer.adversarial_step(
    rollouts: RolloutBuffer,
) -> dict

# Full adversarial training
trainer.train(
    total_iterations: int = 1000,
)
```

---

## 4. Training Components

### PolicyTrainer

Custom HuggingFace Trainer for F1-VLA.

```python
from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainer

trainer = PolicyTrainer(
    model: F1_VLA,
    args: PolicyTrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset = None,
    data_collator: Callable = None,
    optimizers: Tuple = (None, None),
)
```

### PolicyTrainingArguments

Extended training arguments.

```python
from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainingArguments

args = PolicyTrainingArguments(
    # Standard HuggingFace args
    output_dir: str,
    learning_rate: float = 3e-5,
    per_device_train_batch_size: int = 1,
    num_train_epochs: int = 1000,
    
    # Episode-based logging/saving
    logging_episodes: int = 50,       # Log every N episodes
    save_episodes: int = 240,         # Save every N episodes
    eval_episodes: int = 100,         # Eval every N episodes
    
    # Memory-specific
    use_memory: bool = False,
    memory_len: int = 32,
    bptt_steps: int = 4,
)
```

### SequentialRolloutBuffer

Buffer for storing RL rollouts.

```python
from f1_vla.src.models.sequential_rollout_buffer import (
    SequentialRolloutBuffer,
    SequentialRolloutConfig,
)

config = SequentialRolloutConfig(
    max_episodes: int = 100,
    max_steps_total: int = 10000,
    n_obs_img_steps: int = 4,
    action_dim: int = 7,
)

buffer = SequentialRolloutBuffer(config=config)
```

#### Methods

```python
# Add step to buffer
buffer.add_step(
    obs: torch.Tensor,
    action: torch.Tensor,
    reward: float,
    done: bool,
    value: float = None,
    log_prob: float = None,
)

# Start new episode
buffer.start_episode()

# End current episode
buffer.end_episode()

# Get batch for training
batch = buffer.get_batch(batch_size: int = 64)

# Clear buffer
buffer.clear()
```

---

## 5. Data Processing

### SequentialMEKVMDataset

Dataset for sequential BPTT training.

```python
from f1_vla.src.processors.data_processors.sequential_dataset import SequentialMEKVMDataset

dataset = SequentialMEKVMDataset(
    data_dirs: List[str],            # Data directories
    task_descriptions: List[str],    # Task descriptions
    weights: List[float],            # Sampling weights
    n_obs_img_steps: int = 4,        # History length
    image_size: int = 224,           # Image size
    transform: Callable = None,      # Image transforms
)
```

#### Methods

```python
# Get item
item = dataset[idx]  # Returns dict with images, actions, states

# Get length
length = len(dataset)

# Get episode info
info = dataset.get_episode_info(idx)
```

### CollateFn

Data collator for batching.

```python
from f1_vla.src.processors.data_processors.data_loader import CollateFn

collate_fn = CollateFn(
    tokenizer,
    policy_config,
    data_config,
)

# Collate batch
batch = collate_fn(samples: List[dict]) -> dict
```

---

## 6. Configuration

### F1Config

Model configuration class.

```python
from f1_vla.src.models.configuration_f1 import F1Config

config = F1Config(
    # Vision encoder
    vision_config: dict = None,
    
    # Language model
    text_config: dict = None,
    
    # Generation expert
    gen_expert_config: dict = None,
    
    # Action expert
    action_expert_config: dict = None,
    
    # Memory
    use_memory: bool = False,
    memory_config: dict = None,
    
    # VAE
    vae_config: dict = None,
    pixel_loss_weight: float = 0.0,
)
```

### Memory Config

```python
memory_config = {
    "memory_len": 32,           # Memory slots per layer
    "bptt_steps": 4,            # BPTT truncation
    "init_std": 0.02,           # Initialization std
    "tokenizer_max_length": 512,
}
```

### Loading Config from YAML

```python
from omegaconf import OmegaConf

# Load config
config = OmegaConf.load("config.yaml")

# Access nested values
memory_len = config.exp.memory_config.memory_len

# Merge configs
config = OmegaConf.merge(config, override_config)

# Convert to dict
config_dict = OmegaConf.to_container(config, resolve=True)
```

---

## Usage Examples

### Basic Inference

```python
from f1_vla.src.policies.f1_policy import F1_VLA
import torch

# Load model
policy = F1_VLA.from_pretrained("path/to/checkpoint")
policy.eval()
policy.cuda()

# Prepare input
obs = {
    "image_head": torch.randn(1, 4, 3, 224, 224).cuda(),
    "image_wrist": torch.randn(1, 4, 3, 224, 224).cuda(),
    "instruction": "Pick up the red cube",
}

# Get action
with torch.no_grad():
    action = policy.get_action(obs)
    
print(f"Action: {action.shape}")  # (1, 7)
```

### Training with Memory

```python
from f1_vla.src.models.memory import KVMemoryBank

# Initialize memory
memory = KVMemoryBank(
    num_layers=26,
    num_kv_heads=8,
    head_dim=256,
    hidden_size=2048,
    memory_len=32,
)

# Training loop
for batch in dataloader:
    # Get memory for batch
    batch_keys = [(d, e) for d, e in zip(batch["dataset_idx"], batch["episode_idx"])]
    memory_state = memory.get_memory(batch_keys, device, dtype)
    
    # Forward pass
    output = model(
        **batch,
        memory_state=memory_state,
    )
    
    # Update memory
    new_memory = output.memory_state
    memory.update_memory(batch_keys, new_memory, detach=True)
    
    # Backward pass
    loss = output.loss
    loss.backward()
```

### Explorer Training

```python
from f1_vla.src.models.explorer import ExplorerConfig, initialize_explorer
from f1_vla.src.models.explorer_trainer import ExplorerRLTrainer

# Initialize explorer
config = ExplorerConfig(
    random_init=True,
    freeze_world_model=True,
)
initialize_explorer(policy, config)

# Create trainer
trainer = ExplorerRLTrainer(
    policy=policy,
    config=training_config,
    env=env,
)

# Train
trainer.train(total_timesteps=100000)
```

---

## Type Hints

```python
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch import Tensor

# Common types
BatchKey = Tuple[int, int]  # (dataset_idx, episode_idx)
MemoryState = List[Tuple[Tensor, Tensor]]  # List of (key, value) per layer
ActionChunk = Tensor  # Shape: (batch, chunk_size, action_dim)
ImageTensor = Tensor  # Shape: (batch, channels, height, width)
```

---

*Last updated: January 2026*
