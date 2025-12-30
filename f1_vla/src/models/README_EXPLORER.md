# Explorer Actor RL Training

Explorer Actor 是一个用于主动探索的强化学习模块，目标是选择能获取更多环境信息的动作，帮助 World Model 学习更准确的环境预测。

## 概述

Explorer 训练分为两个阶段：
1. **Phase 1**: 冻结 World Model，用 PPO 训练 Explorer
2. **Phase 2**: 解冻 World Model，进行对抗训练（WM vs Explorer）

## 架构设计

### 模型执行顺序

```
时刻 t 的执行流程：
┌─────────────────────────────────────────────────────────────┐
│ 1. PaliGemma: 图像[t-L+1:t] + 语言 → KV cache              │
│ 2. World Model: KV cache + a_t → pred_{t+1}, uncertainty   │
│ 3. 环境执行 a_t → gt_{t+1}, s_{t+1}                        │
│ 4. Explorer: KV cache + embeddings → a_{t+1}               │
└─────────────────────────────────────────────────────────────┘
```

### Explorer 输入（L=4 为例）

| 输入 | 维度 | 描述 |
|------|------|------|
| `state_history` | [s_{t-3}, ..., s_{t+1}] | L+1 帧状态历史 |
| `action_history` | [a_{t-3}, ..., a_t] | L 帧动作历史 |
| `gt_img_emb` | [emb_{t-3}, ..., emb_{t+1}] | L+1 帧 GT 图像 embedding |
| `pred_img_emb` | [pred_{t-2}, ..., pred_{t+1}] | L 帧 WM 预测 embedding |
| `pred_uncertainty` | [unc_{t-2}, ..., unc_{t+1}] | WM 预测不确定度 |

### Reward 设计

```python
reward = α * r1 + β * r2 + γ * r3 + ε * r4 - δ * |a_t|

# r1: WM 不确定度（即时）- 越高说明探索到新区域
r1 = uncertainty_{t+1}

# r2: 预测误差（即时）- 越大说明 WM 预测不准
r2 = MSE(pred_emb_{t+1}, emb_{t+1})

# r3: MSE 改善（延迟1步）- 正值说明信息有价值
r3 = MSE_{t+1} - MSE_{t+2}

# r4: 不确定度改善（延迟1步）- 正值说明 WM 变自信
r4 = uncertainty_{t+1} - uncertainty_{t+2}
```

默认权重：`α=1.0, β=1.0, γ=0.5, ε=0.1, δ=0.01`

## 模块说明

### 1. VAE Embedding 提取器
**文件**: [vae_embedding.py](vae_embedding.py)

```python
from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor

extractor = VAEEmbeddingExtractor(vae, vocab_size=4096, device='cuda')

# 从图像提取 embedding
embedding = extractor.encode_image(image)  # (B, embed_dim)

# 从 token indices 提取 embedding
embedding = extractor.get_embedding_from_indices(indices)

# 计算不确定度（entropy）
uncertainty = extractor.compute_entropy(logits)

# 计算预测误差
mse = extractor.compute_embedding_mse(pred_emb, gt_emb)
```

### 2. Reward 计算模块
**文件**: [reward_computation.py](reward_computation.py)

```python
from f1_vla.src.models.reward_computation import (
    RewardComputer, RewardConfig, ExplorerRewardManager
)

# 配置 reward 权重
config = RewardConfig(alpha=1.0, beta=1.0, gamma=0.5, epsilon=0.1, delta=0.01)

# 创建 reward manager
manager = ExplorerRewardManager(config, buffer_size=1000)

# 计算即时 reward（r1 + r2）
immediate = manager.compute_immediate_reward(pred_emb, gt_emb, uncertainty, action)

# 计算延迟 reward（r3 + r4）
delayed = manager.compute_delayed_reward(
    mse_current, mse_next, unc_current, unc_next
)

# 计算完整 reward
full_reward = manager.compute_full_reward(...)
```

### 3. Rollout 收集器
**文件**: [explorer_rollout.py](explorer_rollout.py)

```python
from f1_vla.src.models.explorer_rollout import (
    ExplorerRolloutCollector, RolloutConfig
)

config = RolloutConfig(
    history_length=4,
    max_episode_steps=200,
    gamma=0.99,
    gae_lambda=0.95
)

collector = ExplorerRolloutCollector(
    policy, env, vae, reward_manager, config, device
)

# 收集 rollout
transitions = collector.collect_rollout(num_steps=256)

# 转换为训练 batch
batch = collector.transitions_to_batch(transitions)
```

### 4. Phase 1 RL 训练器
**文件**: [explorer_trainer.py](explorer_trainer.py)

```python
from f1_vla.src.models.explorer_trainer import (
    ExplorerRLTrainer, ExplorerTrainingConfig
)

config = ExplorerTrainingConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    num_epochs=4,
    batch_size=64
)

trainer = ExplorerRLTrainer(policy, config, device)

# PPO 更新
metrics = trainer.update(batch)
# metrics: {'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction'}

# 保存/加载 checkpoint
trainer.save_checkpoint('checkpoint.pth')
trainer.load_checkpoint('checkpoint.pth')
```

### 5. Phase 2 对抗训练器
**文件**: [adversarial_trainer.py](adversarial_trainer.py)

```python
from f1_vla.src.models.adversarial_trainer import (
    AdversarialTrainingManager, AdversarialTrainingConfig
)

config = AdversarialTrainingConfig(
    wm_learning_rate=1e-4,
    explorer_learning_rate=1e-4,
    wm_updates_per_iteration=10,
    explorer_updates_per_iteration=1,
    warmup_iterations=100,
    collapse_threshold=0.1
)

manager = AdversarialTrainingManager(policy, vae, config, device)

# 对抗训练步骤
for iteration in range(1000):
    batch = collect_rollout()
    metrics = manager.train_step(batch, iteration)
    
    # metrics 包含:
    # - wm_loss: World Model 预测损失
    # - explorer_loss: Explorer PPO 损失
    # - adversarial_reward: 对抗奖励
    # - is_warmup: 是否在 warmup 阶段
```

## 使用方法

### 命令行训练

```bash
# 完整训练（Phase 1 + Phase 2）
python f1_vla/src/scripts/train_explorer.py \
    --config f1_vla/config/explorer_train_config.yaml

# 只运行 Phase 1
python f1_vla/src/scripts/train_explorer.py \
    --config f1_vla/config/explorer_train_config.yaml \
    --phase 1

# 只运行 Phase 2，从 checkpoint 恢复
python f1_vla/src/scripts/train_explorer.py \
    --config f1_vla/config/explorer_train_config.yaml \
    --phase 2 \
    --resume outputs/checkpoints/phase1/final.pth

# 指定输出目录和随机种子
python f1_vla/src/scripts/train_explorer.py \
    --config f1_vla/config/explorer_train_config.yaml \
    --output-dir ./my_experiment \
    --seed 123
```

### 配置文件

配置文件位于 [f1_vla/config/explorer_train_config.yaml](../../config/explorer_train_config.yaml)

主要配置项：

```yaml
# 模型配置
model:
  pretrained_path: "/path/to/F1_pretrain"
  vae:
    checkpoint_path: "/path/to/vae.pth"
    vocab_size: 4096

# Reward 权重
reward:
  alpha: 1.0      # uncertainty
  beta: 1.0       # MSE
  gamma: 0.5      # MSE improvement
  epsilon: 0.1    # uncertainty improvement
  delta: 0.01     # action penalty

# Phase 1 配置
phase1:
  ppo:
    learning_rate: 3.0e-4
    clip_epsilon: 0.2
  training:
    total_timesteps: 100000
    steps_per_rollout: 256

# Phase 2 配置
phase2:
  adversarial:
    wm_updates_per_iter: 10
    explorer_updates_per_iter: 1
    warmup_iterations: 100
    collapse_threshold: 0.1
```

## 文件结构

```
f1_vla/src/
├── models/
│   ├── vae_embedding.py        # VAE embedding 提取
│   ├── reward_computation.py   # Reward 计算
│   ├── explorer_rollout.py     # Rollout 收集
│   ├── explorer_trainer.py     # Phase 1 PPO 训练
│   ├── adversarial_trainer.py  # Phase 2 对抗训练
│   └── README_EXPLORER.md      # 本文档
├── scripts/
│   └── train_explorer.py       # 集成训练脚本
├── tests/
│   ├── test_vae_embedding.py
│   ├── test_reward_computation.py
│   ├── test_explorer_rollout.py
│   ├── test_explorer_trainer.py
│   └── test_adversarial_trainer.py
└── config/
    └── explorer_train_config.yaml
```

## 测试

运行所有测试：

```bash
cd /mnt/data2/ty/F1-VLA

# 单独运行各模块测试
python f1_vla/src/tests/test_vae_embedding.py
python f1_vla/src/tests/test_reward_computation.py
python f1_vla/src/tests/test_explorer_rollout.py
python f1_vla/src/tests/test_explorer_trainer.py
python f1_vla/src/tests/test_adversarial_trainer.py
```

## 实现细节

### Multi-Actor 架构

Explorer 作为一个独立的 actor 存储在 `policy.actors` 字典中：

```python
# 在 F1_VLA 模型中
self.actors = nn.ModuleDict({
    'actor': self.act_expert,      # 原始 actor
    'explorer': explorer_expert,    # Explorer actor
})

# 选择使用的 actor
action = policy.actors['explorer'](state)
```

### GAE 计算

使用 Generalized Advantage Estimation 计算优势函数：

```python
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages
```

### Mode Collapse 防止

当 WM loss 过低时，停止给 Explorer 奖励，防止训练崩溃：

```python
if wm_loss < collapse_threshold:
    explorer_reward = 0  # 不再奖励 Explorer
```

## 训练监控

训练过程中监控以下指标：

| 指标 | 描述 | 期望趋势 |
|------|------|----------|
| `episode_reward` | Episode 总奖励 | Phase 1: ↑ |
| `policy_loss` | PPO 策略损失 | 稳定 |
| `value_loss` | Value 网络损失 | ↓ |
| `entropy` | 动作分布熵 | 适中 |
| `wm_loss` | WM 预测损失 | Phase 2: ↓ |
| `adversarial_reward` | 对抗奖励 | Phase 2: 波动 |

## 注意事项

1. **延迟 Reward**: r3 和 r4 需要下一帧数据，在 episode 边界需要特殊处理
2. **Embedding 维度**: 确保 VAE embedding 维度与模型配置一致（默认 1280）
3. **Warmup 阶段**: Phase 2 开始时先训练 WM，让其稳定后再开始对抗
4. **梯度裁剪**: 建议使用 `max_grad_norm=0.5` 防止梯度爆炸

## Git 提交历史

| Commit | 描述 |
|--------|------|
| f5faa05 | Step 1: Multi-actor 架构 |
| ff7d3f7 | Step 2: Explorer 随机初始化 |
| aa1ad4c | Step 3: VAE embedding 提取器 |
| 5e758f0 | Step 4: Reward 计算模块 |
| 3312c46 | Step 5: 环境 rollout 循环 |
| fc25776 | Step 6: Phase 1 RL 训练 |
| 0075ec3 | Step 7: Phase 2 对抗训练 |
| 56bbe55 | 集成训练脚本和配置 |

## 参考

- [prompt_explorer.md](../../../../prompt_explorer.md) - 设计文档
- [RoboTwin/rl/training/train_student_rl.py](../../../../RoboTwin/rl/training/train_student_rl.py) - Phase 1 参考
- [RoboTwin/rl/training/train_adversarial_rl.py](../../../../RoboTwin/rl/training/train_adversarial_rl.py) - Phase 2 参考
