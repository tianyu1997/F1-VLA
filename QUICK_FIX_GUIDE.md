# 🚀 Bug修复快速指南

## 📝 问题总结
发现**7个导致模型不收敛的Bug**，其中3个为严重问题（必须修复）。

## ✅ 快速修复步骤

### 步骤1: 使用修复后的配置
```bash
# 直接使用修复后的配置文件
./train.sh -c f1_vla/config/memory_from_f1pretrain_fixed.yaml -a
```

### 步骤2: (可选但推荐) 计算数据归一化统计
```bash
# 计算normalization统计 (大约需要1-2分钟)
python tools/compute_quick_norm_stats.py \
    --data_dirs data/clean_teacher_offline/part_gpu0 \
                data/clean_teacher_offline/part_gpu1 \
                data/clean_teacher_offline/part_gpu2 \
                data/clean_teacher_offline/part_gpu3 \
    --output f1_vla/config/norm_stats.yaml \
    --max_episodes 50

# 在配置文件中启用 (memory_from_f1pretrain_fixed.yaml 顶部添加):
# norm_stats: !include norm_stats.yaml
```

### 步骤3: 监控训练
```bash
# 实时查看日志
tail -f logs/latest_log.log

# 查看TensorBoard
tensorboard --logdir outputs/memory_from_f1pretrain_v3_fixed
```

---

## 🔍 主要修复内容

### ✅ 已自动修复的问题:
1. ✅ Label smoothing: 0.1 → 0.02
2. ✅ NaN检测: 静默替换 → 抛出异常

### 📋 配置文件修复 (memory_from_f1pretrain_fixed.yaml):
1. ✅ VAE: test_mode=True → False, pixel_loss_weight=0.0 → 0.1
2. ✅ Learning rate: gen_expert_lr=5e-5 → 2e-5, lr=3e-5 → 2e-5
3. ✅ Batch size: gradient_accumulation=8 → 16 (全局batch 32→64)
4. ✅ Loss warmup: 启用8帧warmup, min_weight=0.3
5. ⚠️ 数据归一化: 需手动计算并添加 (见步骤2)

---

## 📊 预期效果

修复后应该看到:
- **wm_acc**: 逐步从~0.02提升到>0.3
- **wm_loss**: 从8-9降至3-4
- **训练稳定**: 无NaN/Inf错误
- **收敛速度**: 2000-3000 episodes内明显改善

如果1000 episodes后仍无改善，检查:
1. 数据是否正确加载
2. GPU内存是否充足
3. checkpoint路径是否正确

---

## 📁 创建的文件

- ✅ `f1_vla/config/memory_from_f1pretrain_fixed.yaml` - 修复后配置
- ✅ `BUG_REPORT_CONVERGENCE.md` - 详细Bug报告
- ✅ `CONVERGENCE_BUGS_FIXED.md` - 修复说明
- ✅ `tools/compute_quick_norm_stats.py` - 归一化统计工具
- ✅ `QUICK_FIX_GUIDE.md` - 本指南

代码修改:
- ✅ `f1_vla/src/policies/f1_policy.py` (line 82)
- ✅ `f1_vla/src/models/memory.py` (line 106-113)

---

## ❓ 常见问题

**Q: 为什么要降低label smoothing?**  
A: 4096个VAE token类别，0.1的smoothing过强，削弱学习信号。

**Q: VAE decoder为什么要解冻?**  
A: test_mode=True完全冻结decoder，只有token loss没有pixel loss，监督信号不足。

**Q: 数据归一化必须做吗?**  
A: 不是必须，但强烈推荐。未归一化会导致数值范围不一致，影响稳定性。

**Q: 如果遇到OOM怎么办?**  
A: 降低gradient_accumulation_steps或减少cache_max_size。

---

## 🔗 相关文档

- 完整Bug分析: [BUG_REPORT_CONVERGENCE.md](BUG_REPORT_CONVERGENCE.md)
- 修复详情: [CONVERGENCE_BUGS_FIXED.md](CONVERGENCE_BUGS_FIXED.md)
- 配置文件: [f1_vla/config/memory_from_f1pretrain_fixed.yaml](f1_vla/config/memory_from_f1pretrain_fixed.yaml)

---

**最后更新**: 2026-01-12  
**检查者**: GitHub Copilot + Claude Sonnet 4.5
