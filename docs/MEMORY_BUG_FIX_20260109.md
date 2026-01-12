# Memory Module Bug Fix - 2026-01-09

## Bug Description

在 `f1_policy.py` 的 `forward_with_world_model` 函数中发现了一个严重的bug，导致 memory 模块的 step counter 被重复更新，破坏了 BPTT（Backpropagation Through Time）逻辑。

## Root Cause

1. **重复的 memory 更新代码**：
   - 在 `train_gen_expert_only=True` 分支中有一处 memory 更新代码（原第645-664行）
   - 在 `train_gen_expert_only=False` 分支中有另一处几乎相同的 memory 更新代码（原第693-713行）

2. **重复的 step counter 更新**：
   - 两处都手动调用了 `update_step_count`
   - 但 `store_updated_memory` 函数内部也会调用 `update_step_count`
   - 导致每个 forward pass 中，step counter 被更新了 **2次** 而不是 1次

3. **影响**：
   - Step counter 的错误计数会导致 BPTT truncation 在错误的时间点发生
   - `should_detach()` 函数会返回错误的值
   - 梯度会在错误的地方被 detach，破坏梯度流
   - 这是训练无法收敛的主要原因之一

## Code Changes

### 修复前（原代码）

```python
# 第一处：train_gen_expert_only=True 分支
if train_gen_expert_only:
    # ... loss 计算 ...
    
    # Update memory with GRU and store to memory bank
    if self.config.use_memory and self.model.memory_manager is not None:
        # ... 获取 indices ...
        if memory_info is not None and memory_kv is not None:
            updated_memory = self.model.memory_bank.update_memory(memory_kv, memory_info)
            self._update_memory_state(batch, updated_memory, should_detach)  # 调用1
        
        # Update step count for BPTT tracking
        for b in range(len(dataset_indices)):  # ❌ 重复调用
            self.model.memory_manager.update_step_count(...)

# 第二处：train_gen_expert_only=False 分支
else:
    # ... loss 计算 ...
    
    # Update memory with GRU and store to memory bank (same as above)
    if self.config.use_memory and self.model.memory_manager is not None:
        # ... 获取 indices ...
        if memory_info is not None and memory_kv is not None:
            updated_memory = self.model.memory_bank.update_memory(memory_kv, memory_info)
            self._update_memory_state(batch, updated_memory, should_detach)  # 调用2
        
        # Update step count for BPTT tracking
        for b in range(len(dataset_indices)):  # ❌ 重复调用
            self.model.memory_manager.update_step_count(...)
```

注意：`_update_memory_state` 内部调用 `store_updated_memory`，而 `store_updated_memory` 内部已经调用了 `update_step_count`！

### 修复后（新代码）

```python
# 第一处：train_gen_expert_only=True 分支
if train_gen_expert_only:
    # ... loss 计算 ...
    
    # Update memory with GRU and store to memory bank
    if self.config.use_memory and self.model.memory_manager is not None:
        # ... 获取 indices ...
        if memory_info is not None and memory_kv is not None:
            updated_memory = self.model.memory_bank.update_memory(memory_kv, memory_info)
            # store_updated_memory will handle both storing and step count update
            self.model.memory_manager.store_updated_memory(batch, updated_memory, detach=should_detach)
            # ✅ 直接调用，内部会正确调用 update_step_count 一次
    
    return loss_dict

# 第二处：删除重复的代码
# ❌ 删除了 train_gen_expert_only=False 分支后的重复 memory 更新代码
```

## Verification

### store_updated_memory 函数（memory.py 第588-619行）

```python
def store_updated_memory(
    self,
    batch: Dict[str, Any],
    updated_memory: List[Tuple[torch.Tensor, torch.Tensor]],
    detach: bool = True,
) -> None:
    """Store updated memory and update step counts."""
    dataset_indices = batch["dataset_idx"]
    episode_indices = batch["episode_idx"]
    frame_indices = batch["frame_idx"]
    
    # Store memory
    self.memory_bank.store_memory(
        dataset_indices, episode_indices, updated_memory, detach=detach
    )
    
    # Update step counts
    for b in range(len(dataset_indices)):
        self.update_step_count(  # ✅ 在这里正确调用一次
            dataset_indices[b].item(),
            episode_indices[b].item(),
            frame_indices[b].item()
        )
```

## Expected Results

1. **正确的 step counting**：每个 forward pass 中，step counter 只会被更新 1 次
2. **正确的 BPTT truncation**：梯度会在每 `bptt_steps=4` 步正确地 detach
3. **改善的训练收敛**：
   - Loss 应该开始稳定下降
   - Accuracy 应该开始提升
   - 不再出现 loss 在 4.6-6.5 之间震荡的情况

## Testing

建议重新开始训练（或从一个早期 checkpoint 继续）以验证修复效果：

```bash
bash train.sh
```

监控以下指标：
- Loss 是否开始稳定下降（而不是震荡）
- Accuracy 是否开始提升（目标 >30%）
- 梯度是否在正确的位置被 detach（可以通过添加 logging 验证）

## Related Files

- `f1_vla/src/policies/f1_policy.py`: 主要修复文件
- `f1_vla/src/models/memory.py`: Memory 管理相关代码
- `docs/TRAINING_CONVERGENCE_ANALYSIS.md`: 训练收敛问题分析文档
