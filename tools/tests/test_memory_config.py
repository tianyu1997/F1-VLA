#!/usr/bin/env python3
"""
测试目的：全面验证Memory配置和初始化是否正确
预期：
  1. memory_config正确传递到policy_config
  2. KVMemoryBank正确初始化
  3. Memory参数数值范围合理（无NaN/Inf）
  4. GRU更新机制正常工作
  5. BPTT配置合理

成功标准：所有断言通过，无NaN/Inf
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_memory_config_propagation():
    """测试memory_config是否正确传递到policy_config"""
    from omegaconf import OmegaConf
    from f1_vla.src.models.configuration_f1 import F1Config, DictWithAttrAccess
    from f1_vla.src.utils.utils import set_policy_config
    
    # 加载配置
    config = OmegaConf.load('f1_vla/config/memory_from_f1pretrain.yaml')
    policy_config = F1Config.from_pretrained('F1_pretrain')
    
    # 设置policy配置
    set_policy_config(policy_config, config.policy)
    
    # 设置memory配置（模拟train_hf.py逻辑）
    use_memory = config.exp.get('use_memory', False)
    policy_config.use_memory = use_memory
    
    if use_memory and hasattr(config.exp, 'memory_config'):
        mem_cfg = config.exp.memory_config
        policy_config.memory_config = DictWithAttrAccess({
            'memory_len': int(mem_cfg.get('memory_len', 4)),
            'bptt_steps': int(mem_cfg.get('bptt_steps', 8)),
            'init_std': float(mem_cfg.get('init_std', 0.02)),
            'tokenizer_max_length': int(mem_cfg.get('tokenizer_max_length', 512)),
        })
    
    # 验证
    assert policy_config.use_memory == True, "❌ use_memory未设置为True"
    assert hasattr(policy_config, 'memory_config'), "❌ memory_config不存在"
    assert hasattr(policy_config.memory_config, 'memory_len'), "❌ memory_len缺失"
    assert hasattr(policy_config.memory_config, 'bptt_steps'), "❌ bptt_steps缺失"
    assert hasattr(policy_config.memory_config, 'init_std'), "❌ init_std缺失"
    
    print(f"✅ Memory配置传递正确:")
    print(f"   use_memory: {policy_config.use_memory}")
    print(f"   memory_len: {policy_config.memory_config.memory_len}")
    print(f"   bptt_steps: {policy_config.memory_config.bptt_steps}")
    print(f"   init_std: {policy_config.memory_config.init_std}")
    
    return policy_config


def test_memory_bank_initialization():
    """测试KVMemoryBank初始化是否正确"""
    from f1_vla.src.models.memory import KVMemoryBank
    
    # 使用当前配置的参数 (与Gemma2配置一致)
    memory_len = 32  # 配置文件中的值
    num_layers = 18  # Gemma2 18层
    num_kv_heads = 1  # Gemma2 KV heads
    head_dim = 256   # Gemma2 head_dim
    hidden_size = 2048  # Gemma2 hidden_size
    init_std = 0.02
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        memory_len=memory_len,
        hidden_size=hidden_size,
        init_std=init_std,
    )
    
    print(f"\n✅ KVMemoryBank创建成功:")
    print(f"   num_layers: {num_layers}")
    print(f"   memory_len: {memory_len}")
    print(f"   hidden_size: {hidden_size}")
    
    # 检查init_memory参数
    init_mem = memory_bank.init_memory
    print(f"\n   init_memory shape: {init_mem.shape}")
    print(f"   init_memory stats: min={init_mem.min():.4f}, max={init_mem.max():.4f}, mean={init_mem.mean():.4f}, std={init_mem.std():.4f}")
    
    assert not torch.isnan(init_mem).any(), "❌ init_memory包含NaN"
    assert not torch.isinf(init_mem).any(), "❌ init_memory包含Inf"
    assert init_mem.std() > 0.001, f"❌ init_memory标准差过小: {init_mem.std():.6f}"
    assert init_mem.std() < 0.1, f"❌ init_memory标准差过大: {init_mem.std():.6f}"
    
    # 检查memory_token参数
    mem_token = memory_bank.memory_token
    print(f"\n   memory_token shape: {mem_token.shape}")
    print(f"   memory_token stats: min={mem_token.min():.4f}, max={mem_token.max():.4f}, mean={mem_token.mean():.4f}, std={mem_token.std():.4f}")
    
    assert not torch.isnan(mem_token).any(), "❌ memory_token包含NaN"
    assert not torch.isinf(mem_token).any(), "❌ memory_token包含Inf"
    
    # 检查GRU参数
    gru = memory_bank.memory_gru
    print(f"\n   memory_gru: input_size={gru.input_size}, hidden_size={gru.hidden_size}")
    
    for name, param in gru.named_parameters():
        assert not torch.isnan(param).any(), f"❌ GRU参数{name}包含NaN"
        assert not torch.isinf(param).any(), f"❌ GRU参数{name}包含Inf"
    
    print(f"   ✅ GRU参数无NaN/Inf")
    
    return memory_bank


def test_memory_forward_pass():
    """测试Memory前向传播是否正常"""
    from f1_vla.src.models.memory import KVMemoryBank
    
    memory_len = 32
    num_layers = 18
    num_kv_heads = 1
    head_dim = 256
    hidden_size = 2048
    batch_size = 2
    device = torch.device('cpu')
    dtype = torch.float32
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        memory_len=memory_len,
        hidden_size=hidden_size,
        init_std=0.02,
    )
    
    # 测试get_initial_memory
    memory_state = memory_bank.get_initial_memory(batch_size, device, dtype)
    
    print(f"\n✅ get_initial_memory测试:")
    print(f"   返回类型: {type(memory_state)}")
    print(f"   层数: {len(memory_state)}")
    
    for i, (k, v) in enumerate(memory_state[:3]):  # 检查前3层
        print(f"   Layer {i}: K={k.shape}, V={v.shape}")
        assert not torch.isnan(k).any(), f"❌ Layer {i} K包含NaN"
        assert not torch.isnan(v).any(), f"❌ Layer {i} V包含NaN"
    
    # 测试update_memory (GRU更新)
    # 模拟memory_info输出
    memory_info = torch.randn(batch_size, hidden_size)
    
    updated_memory = memory_bank.update_memory(memory_state, memory_info)
    
    print(f"\n✅ update_memory测试:")
    print(f"   更新后层数: {len(updated_memory)}")
    
    for i, (k, v) in enumerate(updated_memory[:3]):
        print(f"   Layer {i}: K={k.shape}, V={v.shape}")
        assert not torch.isnan(k).any(), f"❌ 更新后Layer {i} K包含NaN"
        assert not torch.isnan(v).any(), f"❌ 更新后Layer {i} V包含NaN"
    
    # 验证更新确实改变了memory
    original_k = memory_state[0][0]
    updated_k = updated_memory[0][0]
    diff = (original_k - updated_k).abs().mean()
    print(f"\n   Memory更新差异(mean): {diff:.6f}")
    assert diff > 1e-6, "❌ Memory更新没有改变值（GRU可能有问题）"
    
    print(f"   ✅ Memory正确更新")


def test_memory_gradient_flow():
    """测试Memory梯度是否能正确传播"""
    from f1_vla.src.models.memory import KVMemoryBank
    
    memory_len = 32
    num_layers = 18
    num_kv_heads = 1
    head_dim = 256
    hidden_size = 2048
    batch_size = 2
    device = torch.device('cpu')
    dtype = torch.float32
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        memory_len=memory_len,
        hidden_size=hidden_size,
        init_std=0.02,
    )
    
    # 确保参数需要梯度
    for param in memory_bank.parameters():
        param.requires_grad = True
    
    # 前向传播
    memory_state = memory_bank.get_initial_memory(batch_size, device, dtype)
    memory_info = torch.randn(batch_size, hidden_size, requires_grad=True)
    updated_memory = memory_bank.update_memory(memory_state, memory_info)
    
    # 模拟loss计算
    loss = sum(k.mean() + v.mean() for k, v in updated_memory)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    grad_exists = False
    for name, param in memory_bank.named_parameters():
        if param.grad is not None:
            grad_exists = True
            assert not torch.isnan(param.grad).any(), f"❌ {name}梯度包含NaN"
            assert not torch.isinf(param.grad).any(), f"❌ {name}梯度包含Inf"
    
    assert grad_exists, "❌ 没有参数接收到梯度"
    
    print(f"\n✅ 梯度流测试通过:")
    print(f"   init_memory梯度: {memory_bank.init_memory.grad is not None}")
    print(f"   memory_token梯度: {memory_bank.memory_token.grad is not None}")
    
    # 检查memory_info的梯度
    assert memory_info.grad is not None, "❌ memory_info没有梯度"
    print(f"   memory_info梯度: True")


def test_bptt_config_reasonable():
    """测试BPTT配置是否合理"""
    from omegaconf import OmegaConf
    
    config = OmegaConf.load('f1_vla/config/memory_from_f1pretrain.yaml')
    
    bptt_steps = config.exp.memory_config.bptt_steps
    memory_len = config.exp.memory_config.memory_len
    
    print(f"\n✅ BPTT配置检查:")
    print(f"   bptt_steps: {bptt_steps}")
    print(f"   memory_len: {memory_len}")
    
    # BPTT步数建议：4-8步
    assert 2 <= bptt_steps <= 16, f"❌ bptt_steps={bptt_steps}超出合理范围[2,16]"
    
    # Memory长度建议：16-64
    assert 8 <= memory_len <= 128, f"❌ memory_len={memory_len}超出合理范围[8,128]"
    
    # 检查gradient_accumulation_steps
    grad_accum = config.exp.training_args.gradient_accumulation_steps
    print(f"   gradient_accumulation_steps: {grad_accum}")
    
    # 建议：grad_accum >= bptt_steps以确保稳定训练
    if grad_accum < bptt_steps:
        print(f"   ⚠️ 警告: gradient_accumulation_steps({grad_accum}) < bptt_steps({bptt_steps})")
        print(f"      建议: grad_accum >= bptt_steps以获得更稳定的梯度")
    else:
        print(f"   ✅ gradient_accumulation_steps >= bptt_steps")


def test_checkpoint_memory_compatibility():
    """测试checkpoint与当前memory配置的兼容性"""
    from omegaconf import OmegaConf
    import json
    
    config = OmegaConf.load('f1_vla/config/memory_from_f1pretrain.yaml')
    
    # 检查F1_pretrain的config
    f1_pretrain_config_path = "F1_pretrain/config.json"
    
    if os.path.exists(f1_pretrain_config_path):
        with open(f1_pretrain_config_path, 'r') as f:
            ckpt_config = json.load(f)
        
        ckpt_use_memory = ckpt_config.get('use_memory', False)
        
        print(f"\n✅ Checkpoint兼容性检查:")
        print(f"   F1_pretrain use_memory: {ckpt_use_memory}")
        print(f"   当前配置 use_memory: {config.exp.use_memory}")
        
        if not ckpt_use_memory and config.exp.use_memory:
            print(f"   ⚠️ 警告: Checkpoint没有memory参数，将随机初始化memory模块")
            print(f"      这是预期行为（从F1_pretrain开始fresh memory training）")
        
        # 检查是否有memory_config
        if 'memory_config' in ckpt_config:
            print(f"   Checkpoint memory_config: {ckpt_config['memory_config']}")
    else:
        print(f"\n⚠️ F1_pretrain/config.json不存在，跳过兼容性检查")


def main():
    print("=" * 60)
    print("Memory配置完整性测试")
    print("=" * 60)
    
    try:
        # 测试1: 配置传递
        print("\n[1/6] 测试Memory配置传递...")
        test_memory_config_propagation()
        
        # 测试2: Memory Bank初始化
        print("\n[2/6] 测试KVMemoryBank初始化...")
        test_memory_bank_initialization()
        
        # 测试3: 前向传播
        print("\n[3/6] 测试Memory前向传播...")
        test_memory_forward_pass()
        
        # 测试4: 梯度流
        print("\n[4/6] 测试Memory梯度流...")
        test_memory_gradient_flow()
        
        # 测试5: BPTT配置
        print("\n[5/6] 测试BPTT配置合理性...")
        test_bptt_config_reasonable()
        
        # 测试6: Checkpoint兼容性
        print("\n[6/6] 测试Checkpoint兼容性...")
        test_checkpoint_memory_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ 所有Memory测试通过!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
