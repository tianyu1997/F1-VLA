# VAE解冻训练 - 改动总结

## 📋 实现的功能

### 1. **像素级重建损失**
- ✅ 在 `forward_with_world_model` 中添加了 MSE/L1 像素损失
- ✅ 将预测的token indices解码为图像，与真实图像计算loss
- ✅ 支持配置损失类型（mse/l1）和权重

### 2. **VAE部分解冻**
- ✅ Encoder保持冻结（不浪费显存）
- ✅ Decoder和post_quant_conv解冻（可训练）
- ✅ Quantizer保持冻结（codebook不变）

### 3. **配置文件更新**
- ✅ 添加 `vae_config` 配置块
- ✅ 支持 `pixel_loss_weight`, `pixel_loss_type`, `freeze_encoder`
- ✅ 配置从checkpoint恢复训练

### 4. **训练脚本优化**
- ✅ 更新 `train.sh` 默认从checkpoint-episode-10000恢复
- ✅ 添加checkpoint存在性检查
- ✅ 显示VAE训练相关信息

## 🔧 修改的文件

### 核心代码
1. **f1_vla/src/policies/f1_policy.py**
   - 添加像素级损失计算
   - 分离 CE loss 和 pixel loss
   - 记录 `wm_loss_ce` 和 `wm_loss_pixel`

2. **f1_vla/src/models/wm/vqvae.py**
   - 添加 `freeze_encoder` 参数
   - 实现部分解冻逻辑（encoder冻结，decoder训练）

3. **f1_vla/src/models/configuration_f1.py**
   - 添加 `pixel_loss_weight` 和 `pixel_loss_type` 参数
   - 添加 `vae_freeze_encoder` 支持

4. **f1_vla/src/models/f1_integration.py**
   - 根据 `pixel_loss_weight` 自动设置 `test_mode`
   - 传递 `freeze_encoder` 参数到VAE

5. **f1_vla/src/utils/utils.py**
   - 在 `set_policy_config` 中处理 `vae_config`
   - 传递像素损失和VAE训练配置

### 配置文件
6. **f1_vla/config/memory_wm_clean_only.yaml**
   - 添加 `vae_config` 块
   - 设置 `resume_from_checkpoint`
   - 更新 `run_name`

### 训练脚本
7. **train.sh**
   - 默认从checkpoint-episode-10000恢复
   - 添加checkpoint验证
   - 显示VAE和像素损失信息

## 📊 显存分析

```
VAE Decoder参数量: 41,426,851
额外显存占用: ~0.31 GB per GPU

当前显存使用: ~47 GB (GPU 1-4)
预估新显存使用: ~47.3 GB
GPU总容量: 49 GB (A6000)
✅ 显存充足，无需调整batch size
```

## 🚀 使用方法

### 方法1: 使用GPU 1-4（与当前训练相同GPU）
```bash
./train.sh -g 1,2,3,4
```

### 方法2: 自动检测空闲GPU
```bash
./train.sh -a -m 4
```

### 方法3: 使用空闲的GPU 5-7（推荐，不干扰现有训练）
```bash
./train.sh -g 5,6,7
```

### 查看训练日志
```bash
tail -f logs/latest_log.log
```

### 停止训练
```bash
kill $(cat logs/train_pid.txt)
```

## 📈 预期效果

### 训练损失变化
训练开始后会看到新的loss指标：
```
wm_loss_ce: 2.450      # Token预测的交叉熵损失
wm_loss_pixel: 0.032   # 像素重建损失（MSE）
wm_loss: 2.453         # 总损失 = ce + 0.1 * pixel
```

### 改善效果
1. **纹理质量提升**: VAE decoder学习优化像素级重建
2. **细节更清晰**: 直接监督像素输出，减少重建伪影
3. **整体轮廓保持**: Token预测仍占主导（权重0.1vs1.0）

## 🔍 监控指标

### 关键指标
- `wm_acc_mean`: Token预测准确率（应保持或提升）
- `wm_loss_ce`: Token loss（应持续下降）
- `wm_loss_pixel`: 像素loss（应逐渐下降）
- `wm_acc_tail`: 高分辨率token准确率（关注细节）

### 预期趋势
- 前1000 episodes: pixel loss快速下降（0.1 → 0.03）
- 1000-5000 episodes: 稳定下降（0.03 → 0.01）
- 5000+ episodes: 收敛（< 0.01）

## ⚠️ 注意事项

### 1. Teacher Forcing已启用
代码已经在使用Teacher Forcing（训练时用GT tokens），无需额外配置。

### 2. VAE梯度流
- ✅ Decoder: requires_grad=True (可训练)
- ❌ Encoder: requires_grad=False (冻结)
- ❌ Quantizer: requires_grad=False (冻结)

### 3. 如果显存OOM
如果出现显存不足（不太可能），可以：
```yaml
per_device_train_batch_size: 1  # 从2减到1
gradient_accumulation_steps: 8  # 从4增到8
# 保持有效batch size = 1 * 8 = 8
```

## 🧪 测试验证

运行测试脚本验证实现：
```bash
python3 test_vae_pixel_loss.py
```

预期输出：
```
✓ VAE unfreezing tests passed
✓ Pixel loss config loaded
✓ Gradient flow correct
```

## 📝 配置参数说明

### vae_config块
```yaml
vae_config:
  freeze_encoder: True      # 仅训练decoder
  pixel_loss_weight: 0.1    # 像素损失权重（0=禁用）
  pixel_loss_type: mse      # 'mse' 或 'l1'
```

### 调整建议
- 如果纹理改善不明显: 增大 `pixel_loss_weight` 到 0.2
- 如果token准确率下降: 减小 `pixel_loss_weight` 到 0.05
- 如果需要更锐利的边缘: 改为 `pixel_loss_type: l1`

## 📞 问题排查

### 问题1: Checkpoint加载失败
检查路径是否正确：
```bash
ls outputs/memory_wm_clean_only/checkpoint-episode-10000
```

### 问题2: 显存OOM
查看实际显存使用：
```bash
nvidia-smi
```

### 问题3: Loss不下降
检查配置是否生效：
```bash
grep "pixel_loss" logs/latest_log.log
```
