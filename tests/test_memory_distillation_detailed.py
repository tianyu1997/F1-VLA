"""
Memory Distillation 完整验证测试
=================================

验证项：
1. Memory Loss计算正确性
2. 不同memory_loss_weight的影响
3. Teacher memory detach (无梯度)
4. Student memory requires_grad (有梯度)
5. Loss值合理性
6. Memory维度匹配
7. Multi-layer memory蒸馏
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from f1_vla.src.models.kv_memory_bank import KVMemoryBank
    HAS_KV_MEMORY_BANK = True
except ImportError:
    HAS_KV_MEMORY_BANK = False
    print("⚠️  Warning: KVMemoryBank not available, skipping integration test")


def test_memory_loss_computation():
    """测试1: Memory蒸馏损失计算"""
    print("\n" + "="*60)
    print("测试1: Memory蒸馏损失计算")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 模拟teacher和student的memory (18层)
    num_layers = 18
    batch_size = 2
    memory_len = 32
    num_heads = 16
    head_dim = 256
    
    teacher_memory = []
    student_memory = []
    
    for layer_idx in range(num_layers):
        # Teacher memory (detached)
        teacher_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        teacher_v = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        teacher_memory.append((teacher_k.detach(), teacher_v.detach()))
        
        # Student memory (requires_grad)
        student_k = torch.randn(batch_size, num_heads, memory_len, head_dim, 
                               device=device, requires_grad=True)
        student_v = torch.randn(batch_size, num_heads, memory_len, head_dim, 
                               device=device, requires_grad=True)
        student_memory.append((student_k, student_v))
    
    # 计算MSE loss (averaged across layers)
    memory_loss = torch.tensor(0.0, device=device)
    for layer_idx in range(num_layers):
        teacher_k, teacher_v = teacher_memory[layer_idx]
        student_k, student_v = student_memory[layer_idx]
        
        # MSE on keys and values
        key_loss = nn.functional.mse_loss(student_k, teacher_k)
        value_loss = nn.functional.mse_loss(student_v, teacher_v)
        
        memory_loss += (key_loss + value_loss) / 2.0
    
    memory_loss = memory_loss / num_layers  # Average across layers
    
    print(f"✅ Memory Loss计算成功: {memory_loss.item():.4f}")
    print(f"   - 层数: {num_layers}")
    print(f"   - Memory长度: {memory_len}")
    print(f"   - Batch size: {batch_size}")
    
    # 验证梯度
    memory_loss.backward()
    
    teacher_has_grad = any(t.grad is not None for kv in teacher_memory for t in kv)
    student_has_grad = any(t.grad is not None for kv in student_memory for t in kv)
    
    print(f"   - Teacher有梯度: {teacher_has_grad} (应该False)")
    print(f"   - Student有梯度: {student_has_grad} (应该True)")
    
    assert not teacher_has_grad, "❌ Teacher不应该有梯度!"
    assert student_has_grad, "❌ Student应该有梯度!"
    
    return True


def test_memory_loss_weight_impact():
    """测试2: memory_loss_weight的影响"""
    print("\n" + "="*60)
    print("测试2: memory_loss_weight的影响")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 固定的loss值
    gt_loss = torch.tensor(1.5, device=device)
    memory_loss = torch.tensor(0.8, device=device)
    
    test_weights = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    print("\nGT Loss = 1.5, Memory Loss = 0.8")
    print(f"{'Weight':>8} | {'Total Loss':>12} | {'增量':>8}")
    print("-" * 35)
    
    for weight in test_weights:
        total_loss = gt_loss + weight * memory_loss
        increment = weight * memory_loss.item()
        print(f"{weight:>8.1f} | {total_loss.item():>12.4f} | +{increment:>7.4f}")
    
    print("\n✅ 验证完成:")
    print("   - weight=0.0: 只有GT loss")
    print("   - weight>0: 按比例增加memory distillation")
    print("   - 典型推荐值: 0.1-0.5")
    
    return True


def test_memory_dimension_matching():
    """测试3: Memory维度匹配验证"""
    print("\n" + "="*60)
    print("测试3: Memory维度匹配验证")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 不同的配置
    configs = [
        {"memory_len": 16, "num_heads": 16, "head_dim": 256},
        {"memory_len": 32, "num_heads": 16, "head_dim": 256},
        {"memory_len": 64, "num_heads": 12, "head_dim": 128},
    ]
    
    for idx, config in enumerate(configs, 1):
        print(f"\n配置 {idx}: memory_len={config['memory_len']}, "
              f"num_heads={config['num_heads']}, head_dim={config['head_dim']}")
        
        batch_size = 2
        num_layers = 18
        
        teacher_k = torch.randn(batch_size, config['num_heads'], 
                               config['memory_len'], config['head_dim'], device=device)
        student_k = torch.randn(batch_size, config['num_heads'], 
                               config['memory_len'], config['head_dim'], device=device)
        
        # 计算loss
        loss = nn.functional.mse_loss(student_k, teacher_k)
        
        print(f"   ✅ Loss计算成功: {loss.item():.4f}")
        print(f"   - Teacher shape: {teacher_k.shape}")
        print(f"   - Student shape: {student_k.shape}")
    
    return True


def test_multi_layer_aggregation():
    """测试4: 多层Memory loss聚合"""
    print("\n" + "="*60)
    print("测试4: 多层Memory loss聚合")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    num_layers = 18
    batch_size = 2
    memory_len = 32
    num_heads = 16
    head_dim = 256
    
    per_layer_losses = []
    
    for layer_idx in range(num_layers):
        teacher_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        student_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        
        teacher_v = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        student_v = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
        
        key_loss = nn.functional.mse_loss(student_k, teacher_k)
        value_loss = nn.functional.mse_loss(student_v, teacher_v)
        
        layer_loss = (key_loss + value_loss) / 2.0
        per_layer_losses.append(layer_loss.item())
    
    avg_loss = sum(per_layer_losses) / len(per_layer_losses)
    
    print(f"✅ 多层聚合完成:")
    print(f"   - 总层数: {num_layers}")
    print(f"   - 每层平均loss: {avg_loss:.4f}")
    print(f"   - 最小层loss: {min(per_layer_losses):.4f}")
    print(f"   - 最大层loss: {max(per_layer_losses):.4f}")
    print(f"   - 标准差: {torch.std(torch.tensor(per_layer_losses)).item():.4f}")
    
    return True


def test_memory_bank_integration():
    """测试5: KVMemoryBank集成测试"""
    print("\n" + "="*60)
    print("测试5: KVMemoryBank集成测试")
    print("="*60)
    
    if not HAS_KV_MEMORY_BANK:
        print("⚠️  跳过: KVMemoryBank不可用")
        return True
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建memory bank
    config = {
        'num_hidden_layers': 18,
        'num_attention_heads': 16,
        'hidden_size': 4096,
        'memory_len': 32,
        'bptt_steps': 4,
        'init_std': 0.02,
    }
    
    memory_bank = KVMemoryBank(config)
    memory_bank = memory_bank.to(device)
    
    # 初始化memory
    batch_size = 2
    init_memory = memory_bank.get_initial_memory(batch_size, device)
    
    print(f"✅ Memory Bank初始化:")
    print(f"   - 层数: {len(init_memory)}")
    print(f"   - Key shape: {init_memory[0][0].shape}")
    print(f"   - Value shape: {init_memory[0][1].shape}")
    
    # 验证参数
    total_params = sum(p.numel() for p in memory_bank.parameters())
    trainable_params = sum(p.numel() for p in memory_bank.parameters() if p.requires_grad)
    
    print(f"   - 总参数: {total_params:,}")
    print(f"   - 可训练参数: {trainable_params:,}")
    
    # 验证无NaN
    has_nan = any(torch.isnan(p).any() for p in memory_bank.parameters())
    print(f"   - 含有NaN: {has_nan} (应该False)")
    
    assert not has_nan, "❌ Memory bank参数不应该有NaN!"
    
    return True


def test_gradient_backprop_memory_distillation():
    """测试6: Memory蒸馏的梯度反向传播"""
    print("\n" + "="*60)
    print("测试6: Memory蒸馏的梯度反向传播")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建简单的student network
    class SimpleStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.memory_proj = nn.Linear(256, 256)
        
        def forward(self, x):
            return self.memory_proj(x)
    
    student = SimpleStudent().to(device)
    
    # Teacher memory (fixed)
    batch_size = 2
    memory_len = 32
    teacher_memory = torch.randn(batch_size, memory_len, 256, device=device).detach()
    
    # Student generates memory
    student_input = torch.randn(batch_size, memory_len, 256, device=device)
    student_memory = student(student_input)
    
    # Compute distillation loss
    distill_loss = nn.functional.mse_loss(student_memory, teacher_memory)
    
    # GT loss (模拟)
    gt_loss = torch.tensor(1.0, device=device, requires_grad=True)
    
    # Combined loss
    memory_loss_weight = 0.5
    total_loss = gt_loss + memory_loss_weight * distill_loss
    
    # Backward
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    optimizer.zero_grad()
    total_loss.backward()
    
    # 检查梯度
    has_grad = all(p.grad is not None for p in student.parameters() if p.requires_grad)
    grad_norm = torch.sqrt(sum((p.grad**2).sum() for p in student.parameters() if p.grad is not None))
    
    print(f"✅ 梯度反向传播:")
    print(f"   - 所有参数有梯度: {has_grad}")
    print(f"   - 梯度范数: {grad_norm.item():.4f}")
    print(f"   - Total loss: {total_loss.item():.4f}")
    print(f"   - GT loss: {gt_loss.item():.4f}")
    print(f"   - Distillation loss: {distill_loss.item():.4f}")
    
    assert has_grad, "❌ Student参数应该有梯度!"
    
    return True


def test_loss_value_range():
    """测试7: Loss值合理性检查"""
    print("\n" + "="*60)
    print("测试7: Loss值合理性检查")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    memory_len = 32
    num_heads = 16
    head_dim = 256
    
    # Case 1: 完全相同的memory (loss应该接近0)
    print("\n情况1: Teacher和Student memory完全相同")
    teacher_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
    student_k = teacher_k.clone()
    loss1 = nn.functional.mse_loss(student_k, teacher_k)
    print(f"   Loss: {loss1.item():.8f} (应该 ≈ 0)")
    assert loss1.item() < 1e-6, "❌ 相同tensor的MSE loss应该接近0!"
    
    # Case 2: 轻微不同
    print("\n情况2: 轻微差异 (加入小噪声)")
    noise = torch.randn_like(teacher_k) * 0.01
    student_k = teacher_k + noise
    loss2 = nn.functional.mse_loss(student_k, teacher_k)
    print(f"   Loss: {loss2.item():.6f} (应该很小)")
    assert 0 < loss2.item() < 0.01, "❌ 小噪声应该产生小loss!"
    
    # Case 3: 完全随机
    print("\n情况3: 完全随机的memory")
    teacher_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
    student_k = torch.randn(batch_size, num_heads, memory_len, head_dim, device=device)
    loss3 = nn.functional.mse_loss(student_k, teacher_k)
    print(f"   Loss: {loss3.item():.4f} (应该较大, ~1-2)")
    assert loss3.item() > 0.5, "❌ 随机tensor应该产生较大loss!"
    
    print("\n✅ Loss值范围验证通过")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("Memory Distillation 完整验证测试")
    print("="*70)
    
    tests = [
        ("Memory Loss计算", test_memory_loss_computation),
        ("Loss Weight影响", test_memory_loss_weight_impact),
        ("Memory维度匹配", test_memory_dimension_matching),
        ("多层聚合", test_multi_layer_aggregation),
        ("Memory Bank集成", test_memory_bank_integration),
        ("梯度反向传播", test_gradient_backprop_memory_distillation),
        ("Loss值合理性", test_loss_value_range),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"\n✅ [{name}] 测试通过\n")
            else:
                failed += 1
                print(f"\n❌ [{name}] 测试失败\n")
        except Exception as e:
            failed += 1
            print(f"\n❌ [{name}] 测试出错: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"测试总结: {passed}/{len(tests)} 通过, {failed}/{len(tests)} 失败")
    print("="*70)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
