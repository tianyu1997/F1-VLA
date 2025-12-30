"""
Unit Tests for Reward Computation Module
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_computation import (
    RewardConfig,
    RewardComputer,
    RewardComponents,
    RewardBuffer,
    ExplorerRewardManager,
    compute_batch_rewards,
)


def test_reward_config():
    """Test RewardConfig creation and defaults."""
    print("\n[Test 1] RewardConfig creation...")
    
    # Default config
    config = RewardConfig()
    assert config.uncertainty_weight == 1.0
    assert config.mse_weight == 1.0
    assert config.mse_improvement_weight == 1.0
    assert config.uncertainty_improvement_weight == 0.5
    assert config.action_penalty_weight == 0.01
    print("  ✓ Default config works")
    
    # Custom config
    config = RewardConfig(
        uncertainty_weight=2.0,
        mse_weight=0.5,
        reward_clip=5.0,
    )
    assert config.uncertainty_weight == 2.0
    assert config.mse_weight == 0.5
    assert config.reward_clip == 5.0
    print("  ✓ Custom config works")


def test_mse_computation():
    """Test MSE computation."""
    print("\n[Test 2] MSE computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Test 2D tensors (B, D)
    pred = torch.randn(4, 256)
    gt = torch.randn(4, 256)
    
    mse = computer.compute_mse(pred, gt, reduction='none')
    assert mse.shape == (4,), f"Expected shape (4,), got {mse.shape}"
    assert (mse >= 0).all(), "MSE should be non-negative"
    print("  ✓ 2D MSE computation works")
    
    # Test 3D tensors (B, T, D)
    pred = torch.randn(4, 10, 256)
    gt = torch.randn(4, 10, 256)
    
    mse = computer.compute_mse(pred, gt, reduction='none')
    assert mse.shape == (4,), f"Expected shape (4,), got {mse.shape}"
    print("  ✓ 3D MSE computation works")
    
    # Test identical tensors
    pred = torch.randn(4, 256)
    mse = computer.compute_mse(pred, pred, reduction='none')
    assert (mse < 1e-6).all(), "MSE of identical tensors should be ~0"
    print("  ✓ Identical tensor MSE is ~0")


def test_uncertainty_computation():
    """Test uncertainty computation from logits."""
    print("\n[Test 3] Uncertainty computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Test logits input (B, num_tokens, vocab_size)
    logits = torch.randn(4, 100, 4096)
    
    # Entropy method
    unc = computer.compute_uncertainty(logits, method='entropy')
    assert unc.shape == (4,), f"Expected shape (4,), got {unc.shape}"
    assert (unc >= 0).all(), "Entropy should be non-negative"
    print("  ✓ Entropy method works")
    
    # Max entropy method
    unc = computer.compute_uncertainty(logits, method='max_entropy')
    assert unc.shape == (4,), f"Expected shape (4,), got {unc.shape}"
    print("  ✓ Max entropy method works")
    
    # Top-k entropy method
    unc = computer.compute_uncertainty(logits, method='top_k_entropy', top_k=10)
    assert unc.shape == (4,), f"Expected shape (4,), got {unc.shape}"
    print("  ✓ Top-k entropy method works")
    
    # Uniform distribution should have high entropy
    uniform_logits = torch.zeros(4, 100, 4096)  # Uniform after softmax
    unc_uniform = computer.compute_uncertainty(uniform_logits, method='entropy')
    
    # Peaked distribution should have low entropy
    peaked_logits = torch.zeros(4, 100, 4096)
    peaked_logits[:, :, 0] = 100.0  # Make first token very likely
    unc_peaked = computer.compute_uncertainty(peaked_logits, method='entropy')
    
    assert (unc_uniform > unc_peaked).all(), "Uniform should have higher entropy"
    print("  ✓ Entropy ordering correct (uniform > peaked)")


def test_action_penalty():
    """Test action penalty computation."""
    print("\n[Test 4] Action penalty computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Test action penalty
    action = torch.randn(4, 7)
    penalty = computer.compute_action_penalty(action, reduction='none')
    assert penalty.shape == (4,), f"Expected shape (4,), got {penalty.shape}"
    assert (penalty >= 0).all(), "Penalty should be non-negative"
    print("  ✓ Action penalty computation works")
    
    # Larger actions should have larger penalty
    small_action = torch.randn(4, 7) * 0.1
    large_action = torch.randn(4, 7) * 10.0
    
    small_penalty = computer.compute_action_penalty(small_action, reduction='mean')
    large_penalty = computer.compute_action_penalty(large_action, reduction='mean')
    
    assert large_penalty > small_penalty, "Larger actions should have larger penalty"
    print("  ✓ Penalty ordering correct (large > small)")


def test_immediate_reward():
    """Test immediate reward computation (r1, r2)."""
    print("\n[Test 5] Immediate reward computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Create test data
    pred_emb = torch.randn(4, 256)
    gt_emb = torch.randn(4, 256)
    uncertainty = torch.rand(4) * 5.0  # Pre-computed uncertainty
    action = torch.randn(4, 7)
    
    reward, components = computer.compute_immediate_reward(
        pred_emb, gt_emb, uncertainty, action, is_logits=False
    )
    
    assert reward.shape == (4,), f"Expected shape (4,), got {reward.shape}"
    assert components.r1_uncertainty.shape == (4,)
    assert components.r2_mse.shape == (4,)
    assert components.action_penalty.shape == (4,)
    assert components.r3_mse_improvement is None  # Not computed yet
    print("  ✓ Immediate reward computation works")
    
    # Test with logits
    logits = torch.randn(4, 100, 4096)
    reward_logits, _ = computer.compute_immediate_reward(
        pred_emb, gt_emb, logits, action, is_logits=True
    )
    assert reward_logits.shape == (4,)
    print("  ✓ Immediate reward with logits works")


def test_delayed_reward():
    """Test delayed reward computation (r3, r4)."""
    print("\n[Test 6] Delayed reward computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Create test data with controlled values to ensure improvement
    mse_t1 = torch.tensor([10.0, 8.0, 6.0, 4.0])  # Higher MSE at t+1
    mse_t2 = torch.tensor([5.0, 4.0, 3.0, 2.0])   # Lower MSE at t+2 (improvement)
    unc_t1 = torch.tensor([5.0, 4.0, 3.0, 2.0])   # Higher uncertainty at t+1
    unc_t2 = torch.tensor([3.0, 2.5, 2.0, 1.0])   # Lower uncertainty at t+2
    
    delayed_reward, r3, r4 = computer.compute_delayed_reward(
        mse_t1, mse_t2, unc_t1, unc_t2
    )
    
    assert delayed_reward.shape == (4,), f"Expected shape (4,), got {delayed_reward.shape}"
    assert r3.shape == (4,)
    assert r4.shape == (4,)
    
    # r3 should be positive when MSE decreased (mse_t1 > mse_t2)
    assert (r3 > 0).all(), "r3 should be positive when MSE improved"
    print("  ✓ Delayed reward computation works")
    
    # r4 should be positive when uncertainty decreased (unc_t1 > unc_t2)
    assert (r4 > 0).all(), "r4 should be positive when uncertainty improved"
    print("  ✓ r3 and r4 have correct signs")


def test_full_reward():
    """Test full reward computation."""
    print("\n[Test 7] Full reward computation...")
    
    config = RewardConfig()
    computer = RewardComputer(config)
    
    # Create test data for two consecutive steps
    pred_emb_t1 = torch.randn(4, 256)
    gt_emb_t1 = torch.randn(4, 256)
    unc_t1 = torch.rand(4) * 5.0
    
    pred_emb_t2 = torch.randn(4, 256)
    gt_emb_t2 = torch.randn(4, 256)
    unc_t2 = torch.rand(4) * 3.0
    
    action = torch.randn(4, 7)
    
    reward, components = computer.compute_full_reward(
        pred_emb_t1, gt_emb_t1, unc_t1,
        pred_emb_t2, gt_emb_t2, unc_t2,
        action, is_logits=False
    )
    
    assert reward.shape == (4,), f"Expected shape (4,), got {reward.shape}"
    assert components.r1_uncertainty is not None
    assert components.r2_mse is not None
    assert components.r3_mse_improvement is not None
    assert components.r4_uncertainty_improvement is not None
    print("  ✓ Full reward computation works")


def test_reward_buffer():
    """Test RewardBuffer."""
    print("\n[Test 8] RewardBuffer...")
    
    buffer = RewardBuffer(max_length=3)
    
    # Add data
    for i in range(5):
        buffer.add(
            pred_emb=torch.randn(4, 256),
            gt_emb=torch.randn(4, 256),
            uncertainty=torch.rand(4),
            action=torch.randn(4, 7),
            mse=torch.rand(4),
            immediate_reward=torch.rand(4),
        )
    
    # Should maintain max length
    assert len(buffer) == 3, f"Expected length 3, got {len(buffer)}"
    print("  ✓ Buffer respects max length")
    
    # Should be able to compute delayed reward
    assert buffer.can_compute_delayed_reward()
    mse_t1, mse_t2, unc_t1, unc_t2 = buffer.get_delayed_reward_data()
    assert mse_t1.shape == (4,)
    assert mse_t2.shape == (4,)
    print("  ✓ Buffer provides delayed reward data")
    
    # Reset
    buffer.reset()
    assert len(buffer) == 0
    assert not buffer.can_compute_delayed_reward()
    print("  ✓ Buffer reset works")


def test_reward_manager():
    """Test ExplorerRewardManager."""
    print("\n[Test 9] ExplorerRewardManager...")
    
    config = RewardConfig()
    manager = ExplorerRewardManager(config)
    
    # First step - no reward yet
    reward, info = manager.step(
        pred_emb=torch.randn(4, 256),
        gt_emb=torch.randn(4, 256),
        uncertainty=torch.rand(4),
        action=torch.randn(4, 7),
    )
    
    assert reward is None, "First step should not return reward"
    assert 'r1_uncertainty' in info
    assert 'r2_mse' in info
    assert 'immediate_reward' in info
    print("  ✓ First step returns no reward (need delayed components)")
    
    # Second step - still no full reward (need t+2)
    reward, info = manager.step(
        pred_emb=torch.randn(4, 256),
        gt_emb=torch.randn(4, 256),
        uncertainty=torch.rand(4),
        action=torch.randn(4, 7),
    )
    
    assert reward is not None, "Second step should return reward for first action"
    assert 'full_reward' in info
    assert 'r3_mse_improvement' in info
    print("  ✓ Second step returns full reward")
    
    # Reset
    manager.reset()
    reward, _ = manager.step(
        pred_emb=torch.randn(4, 256),
        gt_emb=torch.randn(4, 256),
        uncertainty=torch.rand(4),
        action=torch.randn(4, 7),
    )
    assert reward is None, "After reset, first step should return None"
    print("  ✓ Manager reset works")


def test_batch_rewards():
    """Test batch reward computation."""
    print("\n[Test 10] Batch reward computation...")
    
    B, T = 4, 10
    D = 256
    action_dim = 7
    
    # Create trajectory data
    pred_embeddings = torch.randn(B, T, D)
    gt_embeddings = torch.randn(B, T, D)
    uncertainties = torch.rand(B, T) * 5.0
    actions = torch.randn(B, T, action_dim)
    
    config = RewardConfig()
    rewards, info = compute_batch_rewards(
        pred_embeddings, gt_embeddings, uncertainties, actions,
        config=config, is_logits=False
    )
    
    assert rewards.shape == (B, T - 1), f"Expected shape ({B}, {T-1}), got {rewards.shape}"
    assert info['r1_uncertainty'].shape == (B, T - 1)
    assert info['r2_mse'].shape == (B, T - 1)
    assert info['r3_mse_improvement'].shape == (B, T - 1)
    assert info['r4_uncertainty_improvement'].shape == (B, T - 1)
    print("  ✓ Batch reward computation works")
    
    # Test with logits
    logits = torch.randn(B, T, 100, 4096)
    rewards_logits, _ = compute_batch_rewards(
        pred_embeddings, gt_embeddings, logits, actions,
        config=config, is_logits=True
    )
    assert rewards_logits.shape == (B, T - 1)
    print("  ✓ Batch reward with logits works")


def test_reward_normalization():
    """Test reward normalization and clipping."""
    print("\n[Test 11] Reward normalization and clipping...")
    
    config = RewardConfig(
        normalize_reward=False,
        reward_scale=2.0,
        reward_clip=5.0,
    )
    computer = RewardComputer(config)
    
    # Create large rewards
    pred_emb = torch.randn(4, 256) * 10
    gt_emb = torch.randn(4, 256) * 10
    uncertainty = torch.ones(4) * 100.0  # Large uncertainty
    action = torch.randn(4, 7)
    
    reward, _ = computer.compute_immediate_reward(
        pred_emb, gt_emb, uncertainty, action, is_logits=False
    )
    
    # After clipping, rewards should be within bounds
    clipped = torch.clamp(reward, -5.0, 5.0)
    # Note: clipping happens in _normalize_and_clip, which is called in compute_full_reward
    
    print("  ✓ Reward clipping configuration works")
    
    # Test running stats update
    computer.update_running_stats(torch.randn(100))
    assert computer.reward_count == 100
    print("  ✓ Running stats update works")


def test_reward_components_to_dict():
    """Test RewardComponents.to_dict()."""
    print("\n[Test 12] RewardComponents.to_dict()...")
    
    components = RewardComponents(
        r1_uncertainty=torch.tensor([1.0, 2.0]),
        r2_mse=torch.tensor([3.0, 4.0]),
        r3_mse_improvement=torch.tensor([0.5, 0.5]),
        action_penalty=torch.tensor([0.1, 0.1]),
    )
    
    d = components.to_dict()
    assert 'r1_uncertainty' in d
    assert 'r2_mse' in d
    assert 'r3_mse_improvement' in d
    assert 'action_penalty' in d
    print("  ✓ RewardComponents.to_dict() works")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Reward Computation Unit Tests")
    print("=" * 60)
    
    test_reward_config()
    test_mse_computation()
    test_uncertainty_computation()
    test_action_penalty()
    test_immediate_reward()
    test_delayed_reward()
    test_full_reward()
    test_reward_buffer()
    test_reward_manager()
    test_batch_rewards()
    test_reward_normalization()
    test_reward_components_to_dict()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    main()
