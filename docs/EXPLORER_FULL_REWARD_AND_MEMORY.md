# Explorer完整Reward和KV Memory实现

## 更新日期
2026-01-02

## 概述
本文档描述了Explorer训练中完整reward系统和KV memory/cache支持的实现。

---

## 1. 完整Reward系统

### 1.1 Reward公式
```
reward = α*r1 + β*r2 + γ*r3 + ε*r4 - δ*|a_t|
```

其中：
- **r1**: `uncertainty_{t+1}` - 即时不确定性奖励（从WM预测熵）
- **r2**: `MSE(pred_emb_{t+1}, emb_{t+1})` - 预测误差奖励
- **r3**: `MSE_{t+1} - MSE_{t+2}` - **延迟奖励**：MSE改进
- **r4**: `unc_{t+1} - unc_{t+2}` - **延迟奖励**：不确定性改进
- **Action Penalty**: `δ*|a_t|` - 动作幅度惩罚

### 1.2 实现位置

#### ExplorerRewardManager (reward_computation.py)
```python
class ExplorerRewardManager:
    def step(self, pred_emb, gt_emb, uncertainty, action, is_logits):
        """处理单步奖励计算，支持immediate和delayed rewards"""
        # 1. 计算immediate reward (r1 + r2 - action_penalty)
        immediate_reward, components = self.reward_computer.compute_immediate_reward(
            pred_emb, gt_emb, uncertainty, action
        )
        
        # 2. 存入buffer
        self.reward_buffer.add(pred_emb, gt_emb, uncertainty, action, mse, immediate_reward)
        
        # 3. 如果buffer有足够数据，计算delayed reward (r3 + r4)
        if self.reward_buffer.can_compute_delayed_reward():
            delayed_reward = self.reward_computer.compute_delayed_reward(...)
            full_reward = immediate_reward + delayed_reward
            return full_reward, info
        
        return None, info  # 第一步还没有delayed reward
```

#### SequentialRolloutCollector (sequential_rollout_buffer.py)
```python
def _compute_reward_full(self, obs, next_obs, action, memory_kv=None):
    """完整reward计算，使用World Model预测和VAE embeddings"""
    # 1. 提取VAE embeddings (ground truth)
    gt_emb = self.vae_extractor.encode_image(next_img)
    pred_emb = self.vae_extractor.encode_image(curr_img)
    
    # 2. 计算uncertainty (embedding distance或WM logits)
    uncertainty = torch.norm(gt_emb - pred_emb, dim=-1).mean()
    
    # 3. 通过reward manager计算完整reward
    reward_tensor, info = self.reward_manager.step(
        pred_emb=pred_emb,
        gt_emb=gt_emb,
        uncertainty=uncertainty,
        action=action_tensor,
        is_logits=False,
    )
    
    return reward, info
```

### 1.3 Reward Components详解

#### r1: Uncertainty (不确定性)
- **目标**: 鼓励探索WM不确定的区域
- **计算**: 
  - 理想: WM生成logits的entropy
  - 当前: VAE embedding距离 `||emb_{t+1} - emb_t||`
- **意义**: 高不确定性 → 未知区域 → 高reward

#### r2: MSE (预测误差)
- **目标**: 鼓励找到WM预测不准的状态
- **计算**: `MSE(pred_emb, gt_emb)` in VAE space
- **意义**: 高MSE → WM预测差 → 高reward

#### r3: MSE Improvement (延迟)
- **目标**: 鼓励exploration带来的学习效果
- **计算**: `MSE_{t+1} - MSE_{t+2}`
- **意义**: MSE降低 → WM从新数据学到东西 → 高reward
- **时序**: 需要t+2的数据，延迟1步返回

#### r4: Uncertainty Improvement (延迟)
- **目标**: 确认WM确实减少了不确定性
- **计算**: `unc_{t+1} - unc_{t+2}`
- **意义**: 不确定性降低 → WM更confident → 高reward
- **时序**: 需要t+2的数据，延迟1步返回

#### Action Penalty
- **目标**: 防止过大的动作
- **计算**: `L1 norm: |a_t|`
- **意义**: 平滑动作，避免剧烈变化

### 1.4 配置参数
```yaml
# explorer_train_config.yaml
reward:
  alpha: 1.0      # r1权重
  beta: 1.0       # r2权重
  gamma: 0.5      # r3权重 (delayed)
  epsilon: 0.1    # r4权重 (delayed)
  delta: 0.01     # action penalty权重
  
  clip_reward: true
  reward_clip_range: [-10.0, 10.0]
  normalize_reward: true
```

---

## 2. KV Memory/Cache支持

### 2.1 Memory层次结构

Explorer使用多层memory机制：

```
┌─────────────────────────────────────────────┐
│ 1. Observation History (Image/State)       │
│    - 保存最近4帧图像                         │
│    - 保存最近4个状态向量                     │
│    - 保存最近4个动作                         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 2. Memory KV (Episode-level)               │
│    - 每个episode维护独立的memory_kv          │
│    - GRU更新: memory_{t} = GRU(info_t, mem_{t-1})│
│    - 在episode reset时初始化                 │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 3. Past Key Values (PaliGemma + Gen Expert) │
│    - Transformer的KV cache                   │
│    - 加速sequential生成                      │
│    - 跨step累积                              │
└─────────────────────────────────────────────┘
```

### 2.2 实现位置

#### SequentialRolloutCollector
```python
class SequentialRolloutCollector:
    def __init__(self, ...):
        # KV cache for memory (per episode)
        self.memory_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.past_key_values: Optional[List[torch.FloatTensor]] = None
    
    def reset(self):
        """Episode开始时重置memory"""
        self.memory_kv = None
        self.past_key_values = None
    
    def _compute_reward_full(self, obs, next_obs, action, memory_kv=None):
        """在reward计算中传递memory_kv"""
        wm_output = self.policy.model.select_action_with_world_model(
            wm_batch,
            memory_kv=memory_kv,  # 使用episode-level memory
            return_pred_emb=True,
        )
```

#### ExplorerRLTrainer
```python
class ExplorerRLTrainer:
    def __init__(self, ...):
        # KV memory/cache for Explorer
        self.episode_memory_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.episode_past_kv: Optional[List[torch.FloatTensor]] = None
    
    def forward_explorer(self, batch, actions=None, deterministic=False):
        """Forward时使用memory_kv"""
        forward_kwargs = {
            'actor_name': 'explorer',
            'return_action_stats': True,
        }
        
        # 传递memory_kv
        if self.episode_memory_kv is not None:
            forward_kwargs['memory_kv'] = self.episode_memory_kv
        
        output = self.policy.forward_with_actor(batch, **forward_kwargs)
```

### 2.3 Memory使用流程

#### Episode开始
```python
# 1. Reset collector
collector.reset()  # memory_kv = None, past_key_values = None

# 2. Reset environment
obs, info = env.reset()
```

#### Step forward
```python
# 3. Get action (使用当前memory_kv)
policy_input = collector._get_policy_input(obs)
output = policy.forward_with_actor(
    policy_input, 
    memory_kv=collector.memory_kv,  # 传入episode memory
    use_cache=True  # 启用KV cache
)

# 4. 更新memory (如果模型返回了updated_memory)
if 'memory_kv' in output:
    collector.memory_kv = output['memory_kv']
if 'past_key_values' in output:
    collector.past_key_values = output['past_key_values']
```

#### Reward computation
```python
# 5. 计算reward时使用memory
reward, info = collector._compute_reward_full(
    obs, next_obs, action,
    memory_kv=collector.memory_kv  # 使用accumulated memory
)
```

#### Episode结束
```python
# 6. Episode结束后memory自动清除
if done:
    collector.reset()  # 下一个episode重新开始
```

### 2.4 Memory的作用

#### 1. 长期依赖建模
- 保存episode历史信息
- 避免重复计算
- 提供context给policy decision

#### 2. World Model预测
- WM使用memory预测future states
- Memory包含过去的observations和actions
- 提高预测准确性

#### 3. 不确定性估计
- Memory帮助WM判断当前状态的熟悉程度
- 熟悉状态 → 低不确定性
- 新状态 → 高不确定性

---

## 3. 与现有系统集成

### 3.1 F1_VLA Policy支持
Explorer通过`F1_VLA.forward_with_actor()`使用memory：

```python
# f1_vla/src/policies/f1_policy.py
class F1_VLA:
    def forward_with_actor(self, batch, actor_name=None, memory_kv=None, **kwargs):
        """支持memory_kv参数"""
        # 获取memory state
        if memory_kv is None:
            memory_kv, memory_token, should_detach = self._get_memory_state(batch)
        
        # Forward with memory
        output = self.model.forward(..., memory_kv=memory_kv)
        
        # 返回updated memory
        if 'memory_kv' in output:
            return_dict['memory_kv'] = output['memory_kv']
        
        return return_dict
```

### 3.2 PaliGemmaWithExpert支持
Actor experts使用memory作为KV prefix：

```python
# f1_vla/src/models/paligemma_with_expert.py
def forward(self, ..., memory_kv=None, actor_name=None):
    """
    memory_kv: List[Tuple[k, v]] for each layer
    - If experts_only_memory=True: only affects experts
    - If False: affects both paligemma and experts
    """
    for layer_idx, (paligemma_layer, expert):
        # Prepend memory to expert KV
        if memory_kv is not None:
            mem_k, mem_v = memory_kv[layer_idx]
            expert_k = torch.cat([mem_k, expert_k], dim=2)
            expert_v = torch.cat([mem_v, expert_v], dim=2)
```

---

## 4. 训练流程

### 4.1 完整训练循环
```python
# 1. 初始化
collector = SequentialRolloutCollector(policy, vae, reward_manager, buffer)
trainer = ExplorerRLTrainer(policy, vae, reward_manager)

# 2. Episode循环
for episode in range(num_episodes):
    collector.reset()  # 重置memory
    obs, info = env.reset()
    
    # 3. Step循环
    for step in range(max_steps):
        # Get action with memory
        policy_input = collector._get_policy_input(obs)
        output = trainer.forward_explorer(policy_input)
        action = output['action']
        
        # Execute
        next_obs, env_reward, done, truncated, info = env.step(action)
        
        # Compute full reward with memory
        explorer_reward, reward_info = collector._compute_reward_full(
            obs, next_obs, action, 
            memory_kv=trainer.episode_memory_kv
        )
        
        # Store transition
        buffer.add_step(obs, action, explorer_reward, ...)
        
        # Update memory
        if 'memory_kv' in output:
            trainer.episode_memory_kv = output['memory_kv']
        
        obs = next_obs
        if done:
            break
    
    # 4. PPO update
    batch = buffer.sample_batch()
    metrics = trainer.train_step(batch)
```

### 4.2 Reward时序
```
Step 0: action a_0
  ↓ execute
Step 1: obs_1
  → r1(1), r2(1)  [immediate rewards]
  → No delayed rewards yet
  ↓ action a_1
  
Step 2: obs_2
  → r1(2), r2(2)  [immediate for a_1]
  → r3(1), r4(1)  [delayed for a_0]
  → Full reward for a_0 = r1(1) + r2(1) + r3(1) + r4(1)
  ↓ action a_2
  
Step 3: obs_3
  → r1(3), r2(3)  [immediate for a_2]
  → r3(2), r4(2)  [delayed for a_1]
  → Full reward for a_1 = r1(2) + r2(2) + r3(2) + r4(2)
```

---

## 5. 配置和使用

### 5.1 启用完整reward
在`explorer_train_config.yaml`:
```yaml
reward:
  # 所有权重 > 0 表示启用
  alpha: 1.0      # r1: uncertainty
  beta: 1.0       # r2: MSE
  gamma: 0.5      # r3: MSE improvement (delayed)
  epsilon: 0.1    # r4: unc improvement (delayed)
  delta: 0.01     # action penalty
```

### 5.2 启用Memory
在`explorer_train_config.yaml`:
```yaml
model:
  # F1-VLA模型配置
  use_memory: true
  memory_len: 32  # Memory length per episode
  
environment:
  observation:
    history_length: 4  # Observation history
```

### 5.3 代码示例
```python
# 完整reward + memory训练
python f1_vla/src/scripts/train_explorer.py \
    --config f1_vla/config/explorer_train_config.yaml \
    --phase 1
```

---

## 6. 改进效果

### 6.1 Reward完整性
- ✅ r1 (uncertainty): 鼓励探索未知区域
- ✅ r2 (MSE): 鼓励找到WM预测不准的状态
- ✅ r3 (MSE improvement): 鼓励带来学习效果的exploration
- ✅ r4 (unc improvement): 确认WM从exploration中学习
- ✅ Action penalty: 防止剧烈动作

### 6.2 Memory支持
- ✅ Episode-level memory_kv: 维护episode context
- ✅ Past key values: 加速sequential生成
- ✅ Observation history: 提供短期history
- ✅ 与F1_VLA memory system集成

### 6.3 预期提升
1. **更好的exploration**: 完整reward信号
2. **更长的记忆**: Episode-level memory
3. **更准确的WM预测**: Memory-conditioned
4. **更快的训练**: KV cache加速

---

## 7. 未来改进方向

### 7.1 短期
- [ ] 实现真正的World Model forward (不只是embedding distance)
- [ ] 添加entropy-based uncertainty (不只是MSE)
- [ ] 优化delayed reward的buffer size

### 7.2 中期
- [ ] 跨episode的memory连续性
- [ ] Adaptive memory length
- [ ] Memory compression (减少KV cache大小)

### 7.3 长期
- [ ] Hierarchical memory (short-term + long-term)
- [ ] Memory-based meta-learning
- [ ] 多模态memory (vision + language + action)

---

## 8. 调试和监控

### 8.1 日志信息
```python
# Reward components
logger.debug(f"Step {t}: reward={total:.3f}, "
            f"r1={info['r1_uncertainty']:.3f}, "
            f"r2={info['r2_mse']:.3f}, "
            f"r3={info.get('r3_mse_improvement', 0):.3f}, "
            f"r4={info.get('r4_uncertainty_improvement', 0):.3f}")

# Memory state
logger.debug(f"Memory KV: {len(memory_kv) if memory_kv else 0} layers, "
            f"Past KV: {len(past_kv) if past_kv else 0} layers")
```

### 8.2 TensorBoard可视化
```python
writer.add_scalar('reward/r1_uncertainty', r1, step)
writer.add_scalar('reward/r2_mse', r2, step)
writer.add_scalar('reward/r3_improvement', r3, step)
writer.add_scalar('reward/r4_improvement', r4, step)
writer.add_scalar('reward/total', total, step)
```

---

## 参考文档
- [MEMORY_IMPLEMENTATION.md](MEMORY_IMPLEMENTATION.md) - F1-VLA Memory系统
- [reward_computation.py](../f1_vla/src/models/reward_computation.py) - Reward计算实现
- [sequential_rollout_buffer.py](../f1_vla/src/models/sequential_rollout_buffer.py) - Sequential buffer实现
- [explorer_trainer.py](../f1_vla/src/models/explorer_trainer.py) - Explorer训练器
