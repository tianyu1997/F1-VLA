# Training Stability Improvements - 2025-01-03 (v2 - Aggressive)

## 问题的严重性分析

### 原始训练表现（48小时，89 epochs）
- **Loss波动**: 1.04-2.10 (接近翻倍，极度不稳定)
- **Accuracy震荡**: 12.8%-43.6% (3倍多差异)
- **无收敛迹象**: 48小时训练无任何改善趋势

### 根本原因诊断
1. **学习率过高** (1e-4): 导致参数更新过大，训练震荡
2. **无梯度裁剪**: 可能出现梯度爆炸
3. **正则化不足** (weight_decay=1e-10): 几乎无正则化效果
4. **无dropout**: VAE decoder完全无正则化
5. **数据单一**: 仅使用clean环境，过拟合风险高
6. **Warmup不足**: 200步warmup对于大模型太短

## 实施的激进改进（v2）

### 1. **降低学习率** ⚠️ CRITICAL
**文件**: `f1_vla/config/memory_wm_clean_only.yaml`
- **修改前**: `learning_rate: 1e-4`  
- **修改后**: `learning_rate: 5e-5`  
- **变化**: 降低50%
- **理由**: Loss波动1.04-2.10说明学习率过大，参数更新overshooting

### 2. **添加梯度裁剪** ⚠️ CRITICAL
**新增**: `max_grad_norm: 1.0`
- **效果**: 防止梯度爆炸，稳定训练
- **参考**: 其他配置文件都有此设置，唯独memory_wm_clean_only缺失

### 3. **大幅增加 Weight Decay**
- **修改前**: `weight_decay: 1e-10` (几乎无效)
- **修改后**: `weight_decay: 1e-4`  
- **变化**: 增加1,000,000倍（100万倍）
- **效果**: 强L2正则化，防止权重爆炸和过拟合

### 4. **更强的 VAE Dropout**
```yaml
vae_config:
  dropout: 0.2  # 从0增加到20%
```
- **位置**: VAE ResNet blocks
- **效果**: 训练时随机丢弃20%激活值，增强泛化

### 5. **增加数据多样性** 🆕
```yaml
mekvm_data_dirs:
  - /mnt/data2/ty/F1-VLA/ME_KVM_VLA/data/clean
  - /mnt/data2/ty/F1-VLA/ME_KVM_VLA/data/complex_texture
mekvm_weights:
  - 0.6  # 60% clean
  - 0.4  # 40% complex_texture
```
- **理由**: 单一clean数据导致过拟合，需要环境多样性

### 6. **延长Warmup阶段**
- **修改前**: `warmup_steps: 200`
- **修改后**: `warmup_steps: 1000`  
- **变化**: 增加5倍
- **效果**: 更平滑的学习率上升，避免初期震荡

## 改进对比表

| 参数 | 原始值 | v1修改 | v2修改 | 变化倍数 |
|------|--------|--------|--------|----------|
| learning_rate | 1e-4 | 1e-4 | **5e-5** | 0.5x |
| weight_decay | 1e-10 | 1e-6 | **1e-4** | **1,000,000x** |
| dropout | 0.0 | 0.1 | **0.2** | - |
| max_grad_norm | ❌ 无 | ❌ 无 | **1.0** | 🆕 |
| warmup_steps | 200 | 200 | **1000** | 5x |
| 数据源 | clean only | clean only | **clean+texture** | 🆕 |

## 为什么v1修改不够？

**v1修改（weight_decay 1e-10→1e-6, dropout 0→0.1）**:
- weight_decay 1e-6仍然太小（深度学习通常用1e-4到1e-2）
- dropout 0.1对于如此不稳定的训练不足
- **根本问题未解决**: 学习率过高 + 无梯度裁剪 = 持续震荡

**v2的关键改进**:
- ✅ 降低学习率：直接减少更新幅度
- ✅ 梯度裁剪：防止爆炸式更新
- ✅ 强正则化：weight_decay 1e-4 + dropout 0.2
- ✅ 数据多样性：减少过拟合风险

## 预期效果

### 短期（50-100 epochs）
- Loss应该稳定在一个较小范围（如1.2-1.5，而非1.0-2.1）
- Accuracy震荡幅度减小（目标：波动<10%）
- 梯度裁剪应触发但不频繁

### 中期（200-300 epochs）
- Loss逐步下降，目标 <1.0
- Accuracy稳步提升，目标 >50%
- 训练曲线应该平滑向下

### 长期（500+ epochs）
- Loss收敛到0.6-0.8范围
- Accuracy稳定在60-70%
- 可以观察到明确的性能plateau

## 监控指标

重点关注：
1. **梯度范数**: 观察max_grad_norm是否频繁触发
2. **Loss moving average (200步)**: 应该平滑下降
3. **Accuracy variance**: 标准差应该<5%
4. **学习率曲线**: 确认cosine scheduler正常工作

## 如果仍不收敛

如果采用v2方案后仍无改善，考虑：
1. **进一步降低学习率** → 3e-5 或 2e-5
2. **检查Memory机制**: bptt_steps=4可能不够，增加到8
3. **调试数据**: 检查complex_texture数据质量
4. **简化模型**: 临时禁用Memory，先让基础模型收敛
5. **检查loss函数**: 确认wm_out_loss计算正确
