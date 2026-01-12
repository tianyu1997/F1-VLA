"""
建议的代码修复补丁

Fix 1: 降低label smoothing (f1_vla/src/policies/f1_policy.py line 82)
建议从0.1降低到0.05或0.02

原因: 对于4096个类别的VAE token预测，0.1的label smoothing相当强，
     会将10%的概率分配给错误类别，削弱学习信号。

修改前:
    self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.1)

修改后:
    self.gen_loss_fct = nn.CrossEntropyLoss(reduction="none", label_smoothing=0.02)


Fix 2: 添加数据归一化统计 (dataset_stats.yaml)
建议在config目录创建dataset_stats.yaml文件，包含state和action的mean/std

示例结构:
norm_stats:
  state:
    mean: [0.0, 0.0, 0.0, ...]  # 根据实际数据计算
    std: [1.0, 1.0, 1.0, ...]
  action:
    mean: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    std: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


Fix 3: 改进NaN检测 (f1_vla/src/models/memory.py)
建议在发现NaN时抛出异常而不是静默替换

修改前 (line 106-113):
    def _check_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.error(f"[KVMemoryBank] {name} contains NaN/Inf! replacing with zeros.")
            return torch.where(torch.isnan(tensor) | torch.isinf(tensor), 
                              torch.zeros_like(tensor), tensor)
        return tensor

修改后:
    def _check_nan_inf(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            raise RuntimeError(
                f"[KVMemoryBank] {name} contains NaN/Inf! "
                f"nan={nan_count}, inf={inf_count}. "
                f"Training should be stopped to investigate."
            )
        return tensor


Fix 4: 修复VAE decoder冻结问题
见上面创建的 memory_from_f1pretrain_fixed.yaml 配置文件


总结关键修改:
1. VAE decoder: test_mode=True -> False, pixel_loss_weight=0.0 -> 0.1
2. Learning rates: gen_expert_lr从5e-5降至2e-5
3. Batch size: gradient_accumulation_steps从8增至16
4. Loss warmup: 从禁用改为8帧warmup，min_weight=0.3
5. Label smoothing: 建议从0.1降至0.02
6. 添加数据归一化统计
7. NaN检测改为抛出异常

使用修复后的配置:
./train.sh -c f1_vla/config/memory_from_f1pretrain_fixed.yaml -a
"""
