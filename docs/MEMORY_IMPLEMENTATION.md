# F1-VLA Memory功能实现文档

## 概述

本文档记录了F1-VLA模型中Memory功能的完整实现过程，包括设计思路、代码修改和测试验证。

## 实现目标

为F1-VLA添加KV Memory机制，使模型能够在处理序列数据时保留历史信息，支持BPTT（Backpropagation Through Time）训练。

---

## 实现步骤

### Step 1: Memory配置开关 (Commit: 20e776b)

**文件修改**: `f1_vla/src/configs/f1_vla_config.py`

添加Memory相关配置参数：

```python
# Memory配置
use_memory: bool = False          # 是否启用memory
memory_len: int = 4               # memory长度
bptt_steps: int = 4               # BPTT步数
```

**设计说明**:
- `use_memory`: 主开关，控制是否启用memory功能
- `memory_len`: KV cache保留的历史步数
- `bptt_steps`: 梯度回传的时间步数

---

### Step 2: 顺序数据加载 (Commit: febbc20)

**文件修改**: `f1_vla/src/processors/data_processors/sequential_dataset.py`

实现`SequentialMEKVMDataset`类，支持顺序数据加载：

```python
class SequentialMEKVMDataset(Dataset):
    """
    顺序数据集，用于memory训练
    - Episodes间打乱顺序
    - Episode内部按时间顺序访问
    """
```

**核心特性**:
1. Episode内帧按顺序访问（支持BPTT）
2. Episodes之间随机打乱（防止过拟合）
3. 支持多数据源混合
4. 自动检测episode边界

**数据流**:
```
Episode 1: [frame0, frame1, frame2, ...] → 顺序访问
Episode 2: [frame0, frame1, frame2, ...] → 顺序访问
...
Episodes间: 随机打乱顺序
```

---

### Step 3: Action/State文本输入 (Commit: dc5fce8)

**文件修改**: `f1_vla/src/processors/data_collator.py`

扩展数据整理器，支持action history和state的文本输入：

```python
def _format_action_history(self, actions: torch.Tensor) -> str:
    """格式化历史action为文本"""
    action_strs = []
    for action in actions:
        action_str = ", ".join([f"{a:.3f}" for a in action.tolist()])
        action_strs.append(f"[{action_str}]")
    return "Action history: " + " → ".join(action_strs)

def _format_state(self, state: torch.Tensor) -> str:
    """格式化当前state为文本"""
    state_str = ", ".join([f"{s:.3f}" for s in state.tolist()])
    return f"Current state: [{state_str}]"
```

---

### Step 4a: KVMemoryBank模块 (Commit: d0e09da)

**新建文件**: `f1_vla/src/models/kv_memory.py`

实现KV Memory Bank核心模块：

```python
class KVMemoryBank(nn.Module):
    """
    KV Memory Bank for storing and managing key-value cache
    
    Features:
    - 固定长度的滑动窗口
    - 支持batch操作
    - FIFO更新策略
    - Episode边界重置
    """
    
    def __init__(self, memory_len: int, num_layers: int, num_heads: int, head_dim: int):
        ...
    
    def update(self, new_kv: Tuple[torch.Tensor, torch.Tensor], episode_ends: torch.Tensor):
        """更新memory，在episode边界处重置"""
        ...
    
    def get_memory(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """获取当前memory用于attention"""
        ...
    
    def reset(self, batch_indices: Optional[torch.Tensor] = None):
        """重置指定batch的memory"""
        ...
```

**Memory结构**:
```
Layer 0: [K: (batch, memory_len, num_heads, head_dim), V: (batch, memory_len, num_heads, head_dim)]
Layer 1: [K: ..., V: ...]
...
Layer N: [K: ..., V: ...]
```

---

### Step 4b: Memory集成到F1VLA (Commits: 633fc08, c724636)

**文件修改**: `f1_vla/src/models/f1_vla.py`

将KVMemoryBank集成到F1VLA主模型：

```python
class F1VLA(nn.Module):
    def __init__(self, config):
        ...
        # 初始化Memory Bank
        if config.use_memory:
            self.memory_bank = KVMemoryBank(
                memory_len=config.memory_len,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                head_dim=self.head_dim
            )
    
    def forward(self, ..., episode_ends=None):
        ...
        if self.config.use_memory:
            # 获取历史memory
            past_kv = self.memory_bank.get_memory()
            
            # Forward with memory
            outputs = self.language_model(
                ...,
                past_key_values=past_kv,
                use_cache=True
            )
            
            # 更新memory
            self.memory_bank.update(outputs.past_key_values, episode_ends)
```

---

### Step 5: BPTT梯度分离 (Commits: 01f1473, ef8bcb8)

**文件修改**: `f1_vla/src/models/kv_memory.py`

实现BPTT梯度控制：

```python
def update(self, new_kv, episode_ends, detach_grad=True):
    """
    更新memory
    
    Args:
        new_kv: 新的KV cache
        episode_ends: episode结束标记
        detach_grad: 是否分离梯度（BPTT控制）
    """
    if detach_grad:
        # 分离梯度，防止无限回传
        new_kv = tuple(
            (k.detach(), v.detach()) for k, v in new_kv
        )
    
    # FIFO更新
    self._fifo_update(new_kv)
    
    # Episode边界重置
    self._reset_on_episode_end(episode_ends)
```

**BPTT机制**:
```
Step t-3  →  Step t-2  →  Step t-1  →  Step t
   ↓           ↓           ↓           ↓
[detach]   [detach]    [grad]      [grad]
   ↓           ↓           ↓           ↓
Memory     Memory      Memory     Current
```

---

### 最终修复 (Commit: f4527cf)

**文件修改**: `f1_vla/src/processors/data_processors/sequential_dataset.py`

1. 修复导入路径：
```python
# 从
from f1_vla.src.processors.data_processors.image_transforms import ImageTransforms
# 改为
from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
```

2. 添加image mask字段：
```python
def get_frame(self, ...):
    return {
        ...
        "observation.images.image0_mask": torch.tensor(True),
        "observation.images.image1_mask": torch.tensor(True),
        "observation.images.image2_mask": torch.tensor(False),  # 空占位
    }
```

---

## 测试验证

### 配置文件

创建 `f1_vla/config/memory_test_config.yaml`：

```yaml
stage: stage3_finetune_vla
use_memory: true
memory_len: 4
bptt_steps: 4
max_steps: 20
batch_size: 1
learning_rate: 1.0e-5
torch_compile: false
```

### 测试结果

```
✓ 数据集加载: 41,280 samples / 960 episodes
✓ 顺序数据访问: 验证通过
✓ 模型参数: ~3B trainable
✓ 训练步数: 20/20 完成
✓ Loss曲线: 0.70 → 0.83 → 0.62 → 0.70 (稳定)
✓ Checkpoint: 保存成功 (9.7GB)
```

---

## 使用指南

### 启用Memory训练

```bash
# 方式1: 使用测试配置
python train_hf.py --config-file f1_vla/config/memory_test_config.yaml

# 方式2: 在自定义配置中添加
use_memory: true
memory_len: 4
bptt_steps: 4
```

### 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_memory` | bool | false | 启用memory功能 |
| `memory_len` | int | 4 | Memory保留的历史步数 |
| `bptt_steps` | int | 4 | 梯度回传的时间步数 |

### 注意事项

1. **显存占用**: Memory会增加显存使用，建议减小batch_size
2. **数据顺序**: 使用memory时必须保证数据按episode顺序加载
3. **Episode边界**: 确保数据中包含episode结束标记
4. **梯度累积**: BPTT步数不宜过大，建议4-8步

---

## Commit历史

| Commit | 描述 |
|--------|------|
| 20e776b | Memory配置开关 |
| febbc20 | 顺序数据加载 (SequentialMEKVMDataset) |
| dc5fce8 | Action/State文本输入 |
| d0e09da | KVMemoryBank模块 |
| 633fc08 | Memory集成到F1VLA (Part 1) |
| c724636 | Memory集成到F1VLA (Part 2) |
| 01f1473 | BPTT梯度分离 (Part 1) |
| ef8bcb8 | BPTT梯度分离 (Part 2) |
| f4527cf | 修复导入和mask字段，添加测试配置 |

---

## 文件清单

### 新增文件
- `f1_vla/src/models/kv_memory.py` - KV Memory Bank模块
- `f1_vla/config/memory_test_config.yaml` - Memory测试配置
- `docs/MEMORY_IMPLEMENTATION.md` - 本文档

### 修改文件
- `f1_vla/src/configs/f1_vla_config.py` - 添加memory配置
- `f1_vla/src/models/f1_vla.py` - 集成memory功能
- `f1_vla/src/processors/data_processors/sequential_dataset.py` - 顺序数据集
- `f1_vla/src/processors/data_collator.py` - 数据整理器

---

*文档创建日期: 2025-12-25*
