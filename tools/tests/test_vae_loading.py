#!/usr/bin/env python3
"""
测试目的：验证VAE是否正确加载并能正常解码
预期：
  1. VAE checkpoint文件存在
  2. VAE权重能成功加载
  3. VAE能正常解码token indices到图像
  4. 解码输出不是纯白/纯黑（数值范围合理）

成功标准：所有断言通过，输出图像数值在合理范围内
"""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_vae_checkpoint_exists():
    """测试VAE checkpoint文件是否存在"""
    vae_path = "/mnt/data2/ty/F1-VLA/var/vae_ch160v4096z32.pth"
    
    assert os.path.exists(vae_path), f"❌ VAE文件不存在: {vae_path}"
    
    file_size_mb = os.path.getsize(vae_path) / 1024 / 1024
    print(f"✅ VAE文件存在: {vae_path}")
    print(f"   文件大小: {file_size_mb:.2f} MB")
    
    assert file_size_mb > 100, f"❌ VAE文件过小 ({file_size_mb:.2f} MB)，可能损坏"
    return vae_path


def test_vae_weights_loadable(vae_path):
    """测试VAE权重能否成功加载"""
    vae_ckpt = torch.load(vae_path, map_location='cpu', weights_only=False)
    
    assert isinstance(vae_ckpt, dict), f"❌ VAE checkpoint不是dict类型: {type(vae_ckpt)}"
    assert len(vae_ckpt) > 0, "❌ VAE checkpoint为空"
    
    print(f"✅ VAE权重加载成功")
    print(f"   参数数量: {len(vae_ckpt)}")
    
    # 检查关键组件
    encoder_keys = [k for k in vae_ckpt.keys() if 'encoder' in k.lower()]
    decoder_keys = [k for k in vae_ckpt.keys() if 'decoder' in k.lower()]
    quantize_keys = [k for k in vae_ckpt.keys() if 'quantize' in k.lower() or 'quant' in k.lower()]
    
    print(f"   Encoder参数: {len(encoder_keys)}")
    print(f"   Decoder参数: {len(decoder_keys)}")
    print(f"   Quantize参数: {len(quantize_keys)}")
    
    assert len(decoder_keys) > 0, "❌ 未找到decoder参数，VAE可能不完整"
    
    return vae_ckpt


def test_vae_decode_functionality():
    """测试VAE解码功能是否正常"""
    from f1_vla.src.models.wm.vqvae import VQVAE
    
    # VAE配置
    v_patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae_path = "/mnt/data2/ty/F1-VLA/var/vae_ch160v4096z32.pth"
    
    # 创建VAE
    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True,
        share_quant_resi=4,
        v_patch_nums=v_patch_nums
    )
    
    # 加载权重
    vae_ckpt = torch.load(vae_path, map_location='cpu', weights_only=False)
    vae.load_state_dict(vae_ckpt, strict=True)
    vae.eval()
    
    print(f"✅ VAE模型创建并加载权重成功")
    
    # 测试解码
    tokens_per_scale = [pn**2 for pn in v_patch_nums]
    tokens_per_frame = sum(tokens_per_scale)  # 680
    
    # 创建测试输入 - 使用不同的token值测试
    batch_size = 1
    
    # 测试1: 全零token（可能对应某种"平均"图像）
    test_indices_zeros = torch.zeros(batch_size, tokens_per_frame, dtype=torch.long)
    
    # 测试2: 随机token
    test_indices_random = torch.randint(0, 4096, (batch_size, tokens_per_frame))
    
    # 测试3: 全最大token
    test_indices_max = torch.full((batch_size, tokens_per_frame), 4095, dtype=torch.long)
    
    results = {}
    for name, indices in [
        ("zeros", test_indices_zeros),
        ("random", test_indices_random), 
        ("max", test_indices_max)
    ]:
        # 准备multi-scale输入
        ms_idx_Bl = []
        start_idx = 0
        for num_tokens in tokens_per_scale:
            ms_idx_Bl.append(indices[:, start_idx:start_idx + num_tokens])
            start_idx += num_tokens
        
        # 解码
        with torch.no_grad():
            decoded = vae.idxBl_to_img(ms_idx_Bl, same_shape=True, last_one=True)
        
        results[name] = {
            "shape": decoded.shape,
            "min": decoded.min().item(),
            "max": decoded.max().item(),
            "mean": decoded.mean().item(),
            "std": decoded.std().item(),
        }
        
        print(f"\n   {name} tokens解码结果:")
        print(f"     Shape: {decoded.shape}")
        print(f"     Range: [{results[name]['min']:.3f}, {results[name]['max']:.3f}]")
        print(f"     Mean: {results[name]['mean']:.3f}, Std: {results[name]['std']:.3f}")
    
    # 验证解码结果
    # 1. 形状正确
    assert results["random"]["shape"] == torch.Size([1, 3, 256, 256]), \
        f"❌ 解码输出形状错误: {results['random']['shape']}"
    
    # 2. 数值范围合理 (VAE输出通常在[-1, 1])
    assert results["random"]["min"] >= -2.0, f"❌ 解码输出最小值异常: {results['random']['min']}"
    assert results["random"]["max"] <= 2.0, f"❌ 解码输出最大值异常: {results['random']['max']}"
    
    # 3. 不同输入应产生不同输出
    assert results["zeros"]["mean"] != results["random"]["mean"], \
        "❌ 不同输入产生相同输出，VAE可能有问题"
    
    # 4. 随机token应该有一定的标准差（不是纯色）
    assert results["random"]["std"] > 0.01, \
        f"❌ 随机token解码标准差过小({results['random']['std']:.4f})，可能是纯色"
    
    print(f"\n✅ VAE解码功能正常")
    return results


def test_vae_in_policy_context():
    """测试在Policy上下文中VAE是否正确初始化"""
    from omegaconf import OmegaConf
    from f1_vla.src.models.configuration_f1 import F1Config
    from f1_vla.src.utils.utils import set_policy_config
    
    # 加载配置
    config = OmegaConf.load('f1_vla/config/memory_from_f1pretrain.yaml')
    policy_config = F1Config.from_pretrained('F1_pretrain')
    
    # 设置policy配置（模拟train_hf.py）
    set_policy_config(policy_config, config.policy)
    
    # 检查VAE配置是否正确设置
    assert hasattr(policy_config, 'gen_expert_config'), "❌ gen_expert_config不存在"
    assert hasattr(policy_config.gen_expert_config, 'vae'), "❌ vae配置不存在"
    
    vae_config = policy_config.gen_expert_config.vae
    
    print(f"✅ Policy中VAE配置:")
    print(f"   vae_ckpt: {vae_config.vae_ckpt}")
    print(f"   vocab_size: {vae_config.vocab_size}")
    print(f"   z_channels: {vae_config.z_channels}")
    print(f"   test_mode: {vae_config.test_mode}")
    
    # 验证VAE checkpoint路径
    assert vae_config.vae_ckpt is not None, "❌ vae_ckpt未设置"
    assert os.path.exists(vae_config.vae_ckpt), f"❌ vae_ckpt路径不存在: {vae_config.vae_ckpt}"
    
    print(f"\n✅ Policy VAE配置正确")


def main():
    print("=" * 60)
    print("VAE加载测试")
    print("=" * 60)
    
    try:
        # 测试1: 文件存在
        print("\n[1/4] 检查VAE checkpoint文件...")
        vae_path = test_vae_checkpoint_exists()
        
        # 测试2: 权重可加载
        print("\n[2/4] 测试VAE权重加载...")
        test_vae_weights_loadable(vae_path)
        
        # 测试3: 解码功能
        print("\n[3/4] 测试VAE解码功能...")
        test_vae_decode_functionality()
        
        # 测试4: Policy上下文
        print("\n[4/4] 测试Policy中的VAE配置...")
        test_vae_in_policy_context()
        
        print("\n" + "=" * 60)
        print("✅ 所有VAE测试通过!")
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
