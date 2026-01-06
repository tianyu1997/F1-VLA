# F1-VLA 代码框架与注意事项

本文档总结 F1-VLA 的代码架构、数据流、以及开发中容易出错的关键点。

---

## 目录

1. [代码架构总览](#1-代码架构总览)
2. [核心模块详解](#2-核心模块详解)
3. [数据流与训练流程](#3-数据流与训练流程)
4. [⚠️ 关键注意事项](#4-️-关键注意事项)
5. [常见Bug与解决方案](#5-常见bug与解决方案)
6. [调试技巧](#6-调试技巧)

---

## 1. 代码架构总览

```
f1_vla/
├── src/
│   ├── models/                     # 模型实现
│   │   ├── modeling_f1.py          # F1FlowMatching 主模型
│   │   ├── configuration_f1.py     # 模型配置类
│   │   ├── memory.py               # KVMemoryBank + MemoryManager
│   │   ├── paligemma_with_expert.py # 带Expert的PaliGemma
│   │   ├── explorer.py             # Explorer Actor配置
│   │   ├── explorer_trainer.py     # Explorer PPO训练器
│   │   ├── adversarial_trainer.py  # 对抗训练器
│   │   ├── vae_embedding.py        # VAE嵌入
│   │   └── reward_computation.py   # 探索奖励计算
│   │
│   ├── policies/                   # 策略封装
│   │   └── f1_policy.py            # F1_VLA 高级接口
│   │
│   ├── processors/                 # 数据与训练处理
│   │   ├── data_processors/
│   │   │   ├── sequential_dataset.py    # 顺序数据集 (BPTT)
│   │   │   ├── data_loader.py           # 数据加载器
│   │   │   └── data_collator.py         # 批数据整理
│   │   └── train_processors/
│   │       ├── policy_trainer.py        # PolicyTrainer (继承HF Trainer)
│   │       └── optimizer_scheduler.py   # 优化器调度
│   │
│   └── utils/                      # 工具函数
│       └── utils.py                # 通用工具
│
└── config/                         # 配置文件
    ├── memory_from_f1pretrain.yaml # World Model训练
    ├── teacher_student_config.yaml # 蒸馏训练
    └── explorer_train_config.yaml  # Explorer RL训练
```

---

## 2. 核心模块详解

### 2.1 模型层次结构

```
F1_VLA (f1_policy.py)
    │
    ├── model: F1FlowMatching (modeling_f1.py)
    │   │
    │   ├── und_expert: PaliGemmaWithExpert (理解专家)
    │   │   └── language_model: GemmaForCausalLM
    │   │
    │   ├── gen_expert: 生成专家 (VAR架构)
    │   │   └── vae: VAE编码器/解码器
    │   │
    │   ├── act_expert: 动作专家 (Flow Matching)
    │   │
    │   └── memory_bank: KVMemoryBank (可选)
    │       └── memory_manager: MemoryManager
    │
    └── config: F1Config
```

### 2.2 Memory模块关键组件

```python
# KVMemoryBank 关键属性
class KVMemoryBank:
    init_memory      # nn.Parameter: 可学习初始memory (frame_idx=0时使用)
    memory_token     # nn.Parameter: 追加到输入的memory token
    memory_gru       # GRUCell: 更新memory的GRU
    memory_info_proj # Linear: 将hidden_size投影到GRU输入
    _memory_bank     # Dict: 运行时memory存储 {(dataset_idx, episode_idx): memory_state}

# MemoryManager 关键功能
class MemoryManager:
    - 追踪每个episode的step计数 (用于BPTT截断)
    - 管理memory bank的清理和重置
    - 处理episode边界
```

### 2.3 数据处理流程

```
SequentialMEKVMDataset
    │
    ├── episode_files: 所有episode文件路径
    ├── sample_index: [(episode_idx, frame_idx), ...] 
    └── _cache: 最近使用的episode数据缓存 (LRU)
    
    │
    ▼
SequentialBatchSampler
    │
    ├── 保证同一batch内的样本来自相同frame_idx
    ├── 维护episode顺序以支持BPTT
    └── 跨epoch打乱episode顺序
    
    │
    ▼
CollateFn (data_collator.py)
    │
    ├── 整理图像、动作、状态
    ├── Tokenize语言指令
    └── 添加 dataset_idx, episode_idx, frame_idx
```

---

## 3. 数据流与训练流程

### 3.1 Forward Pass (带Memory)

```
1. 数据准备
   batch = {images, actions, states, lang_tokens, 
            dataset_idx, episode_idx, frame_idx}
            
2. 获取Memory状态
   memory_kv, memory_token, should_detach = _get_memory_state(batch)
   │
   ├── frame_idx == 0: 使用 init_memory (可学习参数)
   └── frame_idx > 0: 从 _memory_bank 获取上一帧的memory
   
3. 主模型Forward
   action_losses, gen_logits, memory_info, past_key_values = model.forward_with_world_model(
       images, lang_tokens, state,
       world_model_input_embs, world_model_output_embs,
       actions, memory_kv, memory_token
   )
   
4. 计算Loss
   loss = wm_loss + action_loss  # (如果train_gen_expert_only则只有wm_loss)
   
5. 更新Memory (GRU)
   updated_memory = memory_bank.update_memory(memory_kv, memory_info)
   
6. 存储Memory到Bank
   memory_bank.store_memory(dataset_idx, episode_idx, updated_memory, detach=should_detach)
```

### 3.2 BPTT 机制

```
Episode Timeline:
Frame:  0    1    2    3    4    5    6    7    ...
        │    │    │    │    │    │    │    │
Memory: M0→ M1→ M2→ M3→ M4→ M5→ M6→ M7→ ...
               └──────┘    └──────┘
               BPTT=4      BPTT=4
               
当 step_count >= bptt_steps 时:
  - memory.detach() 截断梯度
  - 重置 step_count = 0
```

### 3.3 Episode边界处理

```python
# 在 MemoryManager.should_detach()
if frame_idx == 0:
    # 新episode开始，重置step计数
    step_counts[(ds_idx, ep_idx)] = 0
    return True  # 第一帧使用init_memory，需要detach之前的

# 在 Epoch结束时 (EpisodeProgressCallback.on_epoch_begin)
if epoch_complete:
    memory_bank._memory_bank.clear()  # 清空所有缓存的memory
    torch.cuda.empty_cache()          # 释放GPU显存
```

---

## 4. ⚠️ 关键注意事项

### 4.1 Memory相关

#### ❌ 易错点 1: NaN/Inf 在 Memory 中传播

```python
# 问题: GRU更新可能产生NaN，会污染整个memory bank
# 解决: 在每个关键点检查并替换

# memory.py 中的防护
if torch.isnan(memory_info).any() or torch.isinf(memory_info).any():
    logger.error("[KVMemoryBank] memory_info has NaN/Inf! Using zeros.")
    memory_info = torch.zeros_like(memory_info)

# 同时需要clamp防止极端值
memory_info_f32 = torch.clamp(memory_info_f32, -10.0, 10.0)
```

#### ❌ 易错点 2: Memory Bank 无限增长导致 OOM

```python
# 问题: memory_bank存储每个(dataset_idx, episode_idx)的memory，不清理会OOM
# 解决: 在epoch边界清理

# policy_trainer.py - EpisodeProgressCallback.on_epoch_begin()
if epoch_complete:
    model.memory_bank._memory_bank.clear()
    torch.cuda.empty_cache()
```

#### ❌ 易错点 3: init_memory 的 inplace 操作

```python
# 问题: expand()返回的是view，修改会影响原始参数
# 错误写法:
k = init_mem[layer_idx, 0].unsqueeze(0).expand(batch_size, -1, -1, -1)
k[0] = some_value  # 会修改init_mem!

# 正确写法: 加 .contiguous() 或 .clone()
k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
```

#### ❌ 易错点 4: BPTT detach时机

```python
# 问题: 不正确的detach会导致梯度消失或内存泄漏
# 正确逻辑:
should_detach = False
if frame_idx == 0:
    should_detach = True  # 新episode，使用init_memory
elif step_count >= bptt_steps:
    should_detach = True  # BPTT截断点
    
if should_detach and memory_kv is not None:
    memory_kv = [(k.detach(), v.detach()) for k, v in memory_kv]
```

### 4.2 数据加载相关

#### ❌ 易错点 5: 分布式训练数据不一致

```python
# 问题: 多GPU训练时，每个rank需要加载不同的episodes
# 解决: SequentialMEKVMDataset 按 rank 分配

for global_idx in range(total_episodes):
    if global_idx % world_size == rank:
        self.episode_files.append(all_episode_files[global_idx])
```

#### ❌ 易错点 6: Batch内 frame_idx 不一致

```python
# 问题: 标准DataLoader打乱后，同一batch的frame_idx不同，memory更新混乱
# 解决: 使用SequentialBatchSampler保证同一batch内frame_idx相同

class SequentialBatchSampler:
    def __iter__(self):
        # 按frame_idx分组，保证batch内一致
        for frame_idx in range(max_frames):
            batch = [samples with this frame_idx]
            yield batch
```

### 4.3 Teacher-Student 蒸馏相关

#### ❌ 易错点 7: Teacher/Student 设备不匹配

```python
# 问题: split_gpu模式下，teacher和student在不同GPU
# 注意: CUDA_VISIBLE_DEVICES重映射！

# 配置中写的是相对索引
teacher_device: "cuda:0"  # 实际是CUDA_VISIBLE_DEVICES的第一个
student_device: "cuda:1"  # 实际是CUDA_VISIBLE_DEVICES的第二个

# 如果运行 -g 5,6，则:
# cuda:0 → GPU 5
# cuda:1 → GPU 6
```

#### ❌ 易错点 8: past_key_values 梯度

```python
# 问题: Memory蒸馏需要student的KV有梯度，teacher的KV无梯度
# 正确做法:
teacher_kv = teacher_kv.detach()  # 必须detach
student_kv = student_kv  # 保持梯度

memory_loss = F.mse_loss(student_kv, teacher_kv)  # student有梯度
```

### 4.4 Explorer RL 相关

#### ❌ 易错点 9: Phase 1/2 梯度控制

```python
# Phase 1: 冻结WM，只训练Explorer
for param in world_model.parameters():
    param.requires_grad = False
for param in explorer.parameters():
    param.requires_grad = True

# Phase 2: 交替训练
# WM更新时:
explorer.eval()
explorer.requires_grad_(False)
world_model.train()
world_model.requires_grad_(True)

# Explorer更新时: 反过来
```

#### ❌ 易错点 10: Reward 范围不当

```python
# 问题: reward过大或过小会导致训练不稳定
# 解决: 归一化和裁剪

if config.normalize_reward:
    reward = (reward - reward.mean()) / (reward.std() + 1e-8)
    
if config.clip_reward:
    reward = torch.clamp(reward, config.reward_clip_range[0], config.reward_clip_range[1])
```

### 4.5 配置相关

#### ❌ 易错点 11: OmegaConf 嵌套访问

```python
# 问题: OmegaConf的DictConfig不支持所有dict操作
# 易错:
config.exp.memory_config['memory_len']  # 可能失败

# 正确:
config.exp.memory_config.memory_len  # 点号访问
# 或转换为dict
OmegaConf.to_container(config, resolve=True)
```

#### ❌ 易错点 12: 配置优先级

```python
# 加载顺序: YAML < 命令行override
config = OmegaConf.load(yaml_path)
override = OmegaConf.from_dotlist(["exp.use_memory=True"])
config = OmegaConf.merge(config, override)  # override覆盖yaml
```

---

## 5. 常见Bug与解决方案

### Bug 1: OOM at Epoch 2

```
症状: 第一个epoch正常，第二个epoch开始OOM
原因: Memory bank累积 + checkpoint保存时显存峰值

解决:
1. 减少 save_episodes (更频繁保存，释放显存)
2. 在epoch边界清理memory_bank
3. 调用 torch.cuda.empty_cache()
```

### Bug 2: Loss 变成 NaN

```
症状: 训练一段时间后loss突然变成NaN
可能原因:
1. Memory中累积了NaN
2. 学习率过大
3. 梯度爆炸

解决:
1. 启用 init_std: 0.02 (小初始化)
2. 启用 max_grad_norm: 1.0 (梯度裁剪)
3. 在memory.py中增加NaN检查和替换
```

### Bug 3: Resume后指标不连续

```
症状: 从checkpoint恢复后，episode计数/epoch计数不对
原因: callback状态未正确保存/恢复

解决:
1. EpisodeProgressCallback.state() 保存状态
2. EpisodeProgressCallback.load_state() 恢复状态
3. 确保trainer_state.json包含callback状态
```

### Bug 4: 分布式训练卡住

```
症状: 多GPU训练时进程hang住
可能原因:
1. 某个rank遇到空batch
2. DDP sync barrier不平衡

解决:
1. 确保每个rank分到相同数量的samples
2. 使用 drop_last=True
3. 检查 dataloader_num_workers 设置
```

### Bug 5: VAE解码图像错误

```
症状: 解码的图像全黑或噪声
可能原因:
1. VAE权重未正确加载
2. 输入tokens超出vocab_size
3. dtype不匹配

解决:
1. 验证VAE权重路径
2. 检查 gen_logits.argmax() < vocab_size
3. 确保 bf16/fp16 一致
```

---

## 6. 调试技巧

### 6.1 快速验证Memory工作

```python
# 添加到forward中
if self.config.use_memory:
    print(f"[DEBUG] Memory bank size: {len(self.model.memory_bank._memory_bank)}")
    print(f"[DEBUG] frame_idx: {batch['frame_idx'].tolist()}")
    if memory_kv is not None:
        print(f"[DEBUG] memory_kv[0].k shape: {memory_kv[0][0].shape}")
```

### 6.2 检查梯度流

```python
# 检查某个参数是否有梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRAD")
```

### 6.3 监控显存

```python
import torch
def print_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"GPU {i}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
```

### 6.4 Debug配置

```yaml
# 使用小配置快速测试
exp:
  training_args:
    max_steps: 100
    logging_episodes: 1
    save_episodes: 10
    per_device_train_batch_size: 1
    
  memory_config:
    memory_len: 4    # 小memory
    bptt_steps: 2    # 短BPTT
```

### 6.5 检查数据正确性

```python
# 在CollateFn中添加
def __call__(self, samples):
    batch = self._collate(samples)
    
    # Debug prints
    print(f"[CollateFn] Batch keys: {batch.keys()}")
    print(f"[CollateFn] Images shape: {batch['images'].shape}")
    print(f"[CollateFn] Actions shape: {batch['actions'].shape}")
    print(f"[CollateFn] frame_idx: {batch['frame_idx'].tolist()}")
    
    return batch
```

---

## 总结检查清单

在开发/调试时，检查以下关键点：

### Memory模块
- [ ] init_std 设置合理 (建议0.02)
- [ ] NaN检查和替换机制存在
- [ ] Memory bank定期清理
- [ ] BPTT detach时机正确
- [ ] expand() 后有 contiguous()

### 数据加载
- [ ] 分布式训练数据正确分配
- [ ] Batch内frame_idx一致
- [ ] Episode边界正确处理
- [ ] drop_last=True (分布式)

### 训练流程
- [ ] 梯度裁剪已启用
- [ ] Checkpoint保存/恢复正确
- [ ] Episode计数正确
- [ ] GPU显存监控

### Teacher-Student
- [ ] 设备分配正确 (CUDA_VISIBLE_DEVICES映射)
- [ ] Teacher detach, Student有梯度
- [ ] Memory loss权重合理

### Explorer RL
- [ ] Phase 1/2 梯度控制正确
- [ ] Reward归一化/裁剪
- [ ] Mode collapse检测

---

*更新日期: 2026年1月*
