在之前student policy的基础上实现一个explorer actor的强化学习训练。explorer的目的是选择能获取更多环境信息的动作。我会将流程拆分为几步，请你分步实现，每实现一步之后要进行训练验证,验证无误后，提交commit，再进行下一个点。实现代码时，不能影响已有的代码机构，尽量进行增量的设计，创建另外的代码。explorer训练分为两个阶段，第一个阶段冻结world model，进行强化学习，可参考/mnt/data2/ty/F1-VLA/RoboTwin/rl/training/train_student_rl.py；第二阶段解冻world model，与它做对抗训练，可参考/mnt/data2/ty/F1-VLA/RoboTwin/rl/training/train_adversarial_rl.py。注意原代码可能有bug，不能照搬。

## 模型执行顺序（关键约束）

原模型的执行顺序是**串行**的（非并行），Actor使用前面模块的KV cache：

```
执行顺序（时刻t）：
1. PaliGemma (Understanding Expert): 处理图像[t-L+1:t] + 语言 → 生成KV cache
2. World Model (Gen Expert): 使用KV cache + a_t → 生成 pred_{t+1}（执行a_t后的预测）
3. Actor/Explorer: 使用KV cache → 输出 a_{t+1}（下一步要执行的动作）
```

**关键约束**：WM根据**已知的a_t**预测pred_{t+1}。a_t来自上一步Explorer的输出（或数据集）。

因此**Explorer在时刻t可以访问**：
- 当前的 `gt_t`、上一步的 `pred_t`（可对比预测误差）
- 当前WM根据a_t生成的 `pred_{t+1}` 和 `uncertainty_{t+1}`
- 已知的 `a_t`（作为action_history的一部分）

## 时间线与数据流（标准RL下标）

```
时刻t的模型执行（PaliGemma + WM）：
  - 输入: 
    * gt_images [img_{t-L+1}, ..., img_t]
    * state_history [s_{t-L+1}, ..., s_t]  
    * action_history [a_{t-L+1}, ..., a_t]  ← a_t是已知的（上一步Explorer输出）
  - WM输出: pred_{t+1}, pred_emb_{t+1}, uncertainty_{t+1}
  - ✓ 此时环境执行a_t，返回gt_{t+1}, s_{t+1}

时刻t的Explorer执行（在环境返回gt_{t+1}后）：
  - Explorer输入 = 原模型输入 + emb_{t+1} + pred_emb_{t+1}:
    * gt_emb: [emb_{t-L+1}, ..., emb_t, **emb_{t+1}**]  ← L+1帧，含刚获得的gt_{t+1}
    * pred_emb: [pred_emb_{t-L+2}, ..., pred_emb_t, **pred_emb_{t+1}**]  ← 含刚预测的pred_{t+1}
    * state_history: [s_{t-L+1}, ..., s_t, s_{t+1}]  ← L+1帧
    * action_history: [a_{t-L+1}, ..., a_t]  ← L帧（a_t已执行）
    * uncertainty: [unc_{t-L+2}, ..., unc_t, unc_{t+1}]
  - Explorer输出: a_{t+1}（下一步要执行的动作）
  - ✓ 可计算: mse_{t+1} = MSE(pred_emb_{t+1}, emb_{t+1})  ← 当前帧预测误差
  - ✓ 可计算完整reward for a_t
```

**关键**: Explorer可以直接看到 `pred_emb_{t+1}` vs `emb_{t+1}` 的差异，从而学习WM在执行a_t后预测是否准确。

## 设计要点

1. **Explorer架构**: 在model中以一个actor expert的形式存在，架构与原模型的act expert一致。需要保留多actors的接口，用dict储存actors，根据config选择一个或多个actors进行训练或推理。

2. **权重初始化**: Explorer如果在config里没给ckpt则权重随机初始化。

3. **Explorer输入**（时刻t，环境返回gt_{t+1}后，L=4为例）:
   
   Explorer在WM生成pred_{t+1}、环境执行a_t返回gt_{t+1}之后执行，输入 = 原模型输入 + emb_{t+1} + pred_emb_{t+1}：
   
   具体输入：
   - `state_history`: [s_{t-3}, s_{t-2}, s_{t-1}, s_t, **s_{t+1}**] - L+1帧状态历史（含刚返回的s_{t+1}）
   - `action_history`: [a_{t-3}, a_{t-2}, a_{t-1}, a_t] - L帧动作历史（a_t已执行）
   - `gt_img_emb`: [emb_{t-3}, ..., emb_t, **emb_{t+1}**] - L+1帧GT图像embedding（含刚返回的gt_{t+1}）
   - `pred_img_emb`: [pred_emb_{t-2}, ..., pred_emb_t, **pred_emb_{t+1}**] - L帧WM预测embedding
     - `pred_emb_{t+1}`: 当前步WM根据a_t刚生成的，可与`emb_{t+1}`直接对比
   - `pred_uncertainty`: [unc_{t-2}, ..., unc_t, unc_{t+1}] - WM预测的不确定度
   
   **关键**: Explorer可以直接对比 `pred_emb_{t+1}` vs `emb_{t+1}`（WM对执行a_t后的预测是否准确）

4. **Explorer输出**: a_{t+1}（下一步要执行的动作）

5. **环境交互**: 通过config指定训练环境（如random_exploration.py），环境执行a_t并返回state_{t+1}和gt_image_{t+1}。

6. **Reward设计**（部分即时，部分延迟1步）:
   ```python
   # Reward for action a_t
   
   # Part 1: WM预测img_{t+1}时的不确定度（即时可计算）
   # 越高说明探索到了WM不确定的区域
   r1 = uncertainty_{t+1}
   
   # Part 2: 当前帧预测误差（即时可计算）
   # 越大说明WM预测不准，可能是novel state
   r2 = MSE(pred_emb_{t+1}, emb_{t+1})
   
   # Part 3: 预测准确度提升（延迟1步，需要gt_{t+2}）
   # mse_{t+1} - mse_{t+2} > 0 说明看到gt_{t+1}后WM预测变准了
   # 这意味着a_t带来的信息对WM有价值（客观指标）
   r3 = MSE(pred_emb_{t+1}, emb_{t+1}) - MSE(pred_emb_{t+2}, emb_{t+2})
   
   # Part 4: WM自信度提升（延迟1步，需要unc_{t+2}）
   # unc_{t+1} - unc_{t+2} > 0 说明看到gt_{t+1}后WM变自信了
   # 作为r3的辅助信号（主观指标，依赖WM校准质量）
   r4 = uncertainty_{t+1} - uncertainty_{t+2}
   
   # 总reward
   reward = alpha * r1 + beta * r2 + gamma * r3 + epsilon * r4 - delta * |a_t|
   ```
   
   **注意**: 
   - r3、r4需要延迟1步才能计算，可以用TD learning处理，或者在episode结束后batch计算
   - r3（MSE差）是客观指标，r4（uncertainty差）是主观指标，建议 `epsilon < gamma`
   - 如果WM校准良好，r3和r4应高度相关，可根据实验调整epsilon

7. **图像表征：使用VAE Embedding**

   **为什么用VAE embedding而非logits或原始像素：**
   - **维度对比**:
     - 原始像素: 256×256×3 = 196,608 维/帧
     - VAE logits: 4096 vocab × 680 tokens ≈ 2.8M 参数/帧
     - **VAE embedding: ~1280 维/帧**（降维150-2000倍）
   
   - **语义一致性**: GT图像和WM预测都通过同一个VAE encoder提取embedding，保证表征空间一致
   
   - **提取方式**:
     ```python
     # GT图像 → VAE embedding
     gt_emb = vae.encode(gt_image)  # (B, embed_dim)
     
     # WM预测 → VAE embedding（WM内部已经生成）
     pred_emb = wm.get_prediction_embedding()  # (B, embed_dim)
     ```
   
   - **Uncertainty计算**: 使用WM生成过程中VAE decoder的logits计算entropy
     ```python
     # WM生成时保存logits
     logits = wm.get_generation_logits()  # (B, num_tokens, vocab_size)
     probs = F.softmax(logits, dim=-1)
     entropy = -(probs * probs.log()).sum(dim=-1).mean()  # scalar
     ```

## 分步实现计划

| Step | 内容 | 验证方式 |
|------|------|----------|
| 1 | 实现multi-actor架构（dict存储） | 单元测试：加载/保存/选择actor |
| 2 | 实现explorer actor（随机初始化） | 单元测试：forward pass |
| 3 | 实现VAE embedding提取器 | 测试：从图像得到embedding |
| 4 | 实现reward计算（uncertainty + improvement） | 测试：给定WM输出计算reward |
| 5 | 实现环境rollout循环 | 测试：explorer与环境交互 |
| 6 | Phase 1 RL训练（冻结WM） | 训练验证：reward曲线上升 |
| 7 | Phase 2 对抗训练 | 训练验证：WM loss下降 |