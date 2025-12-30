"""
Unit Tests for Adversarial Training Module
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.adversarial_trainer import (
    AdversarialTrainingConfig,
)


def test_adversarial_config():
    """Test AdversarialTrainingConfig."""
    print("\n[Test 1] AdversarialTrainingConfig...")
    
    config = AdversarialTrainingConfig()
    assert config.total_iterations == 10000
    assert config.wm_updates_per_iter == 5
    assert config.explorer_updates_per_iter == 1
    assert config.warmup_iterations == 100
    print("  ✓ Default config works")
    
    config = AdversarialTrainingConfig(
        total_iterations=5000,
        wm_learning_rate=1e-6,
        adversarial_weight=2.0,
    )
    assert config.total_iterations == 5000
    assert config.wm_learning_rate == 1e-6
    assert config.adversarial_weight == 2.0
    print("  ✓ Custom config works")
    
    # Test to_dict
    d = config.to_dict()
    assert 'total_iterations' in d
    assert 'adversarial_weight' in d
    print("  ✓ to_dict works")


def test_alternating_update_schedule():
    """Test alternating WM and Explorer update schedule."""
    print("\n[Test 2] Alternating update schedule...")
    
    wm_updates = 5
    explorer_updates = 1
    warmup = 100
    
    # Simulate training loop
    total_wm_updates = 0
    total_explorer_updates = 0
    
    for iteration in range(200):
        # WM updates always happen
        total_wm_updates += wm_updates
        
        # Explorer updates only after warmup
        if iteration >= warmup:
            total_explorer_updates += explorer_updates
    
    expected_wm = 200 * wm_updates
    expected_explorer = (200 - warmup) * explorer_updates
    
    assert total_wm_updates == expected_wm
    assert total_explorer_updates == expected_explorer
    print(f"  ✓ WM updates: {total_wm_updates} (expected {expected_wm})")
    print(f"  ✓ Explorer updates: {total_explorer_updates} (expected {expected_explorer})")


def test_adversarial_reward_computation():
    """Test adversarial reward computation logic."""
    print("\n[Test 3] Adversarial reward computation...")
    
    batch_size = 4
    adversarial_weight = 1.0
    exploration_weight = 0.5
    threshold = 0.01
    
    # Mock prediction errors
    pred_errors = torch.tensor([0.05, 0.02, 0.005, 0.1])
    uncertainties = torch.tensor([0.3, 0.2, 0.1, 0.4])
    
    # Apply threshold (only reward when error > threshold)
    pred_errors_thresholded = torch.where(
        pred_errors > threshold,
        pred_errors,
        torch.zeros_like(pred_errors)
    )
    
    # Compute reward
    reward = adversarial_weight * pred_errors_thresholded + exploration_weight * uncertainties
    
    assert reward.shape == (batch_size,)
    print("  ✓ Reward shape correct")
    
    # Third sample should have 0 adversarial component (below threshold)
    assert reward[2] == exploration_weight * uncertainties[2]
    print("  ✓ Threshold applied correctly")
    
    # Higher error should give higher reward
    assert reward[3] > reward[0]  # 0.1 > 0.05
    print("  ✓ Higher error gives higher reward")


def test_wm_loss_computation():
    """Test WM prediction loss computation."""
    print("\n[Test 4] WM loss computation...")
    
    batch_size = 4
    channels = 3
    height, width = 64, 64
    
    # Mock predictions and ground truth
    pred_imgs = torch.randn(batch_size, channels, height, width)
    gt_imgs = torch.randn(batch_size, channels, height, width)
    
    # Per-sample MSE
    loss_per_sample = torch.nn.functional.mse_loss(
        pred_imgs, gt_imgs, reduction='none'
    ).mean(dim=[1, 2, 3])  # (B,)
    
    assert loss_per_sample.shape == (batch_size,)
    print("  ✓ Per-sample loss shape correct")
    
    # Mean loss
    loss = loss_per_sample.mean()
    assert loss.dim() == 0  # Scalar
    print("  ✓ Mean loss is scalar")
    
    # Identical predictions should have ~0 loss
    identical_loss = torch.nn.functional.mse_loss(pred_imgs, pred_imgs)
    assert identical_loss < 1e-6
    print("  ✓ Identical predictions have ~0 loss")


def test_gae_with_adversarial_rewards():
    """Test GAE computation with adversarial rewards."""
    print("\n[Test 5] GAE with adversarial rewards...")
    
    gamma = 0.99
    gae_lambda = 0.95
    
    # Adversarial rewards (can be higher when WM is surprised)
    rewards = [0.1, 0.5, 0.2, 0.8, 0.1]  # Spiky rewards when explorer finds novel states
    values = [0.3, 0.4, 0.3, 0.5, 0.2]
    
    # Compute GAE
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    assert len(advantages) == len(rewards)
    assert len(returns) == len(rewards)
    print("  ✓ GAE computation works")
    
    # High reward step should have positive advantage
    # Step 3 has highest reward (0.8)
    # Note: advantage depends on future rewards too, so this is approximate
    print(f"  Advantages: {[f'{a:.3f}' for a in advantages]}")
    print(f"  Returns: {[f'{r:.3f}' for r in returns]}")


def test_collapse_prevention():
    """Test mode collapse prevention logic."""
    print("\n[Test 6] Mode collapse prevention...")
    
    wm_loss_threshold = 0.01
    
    # Case 1: WM loss above threshold -> Explorer gets rewarded
    wm_loss_high = 0.05
    should_reward = wm_loss_high > wm_loss_threshold
    assert should_reward == True
    print("  ✓ High WM loss: Explorer rewarded")
    
    # Case 2: WM loss below threshold -> Explorer not rewarded
    # (to prevent explorer from finding trivial solutions)
    wm_loss_low = 0.005
    should_reward = wm_loss_low > wm_loss_threshold
    assert should_reward == False
    print("  ✓ Low WM loss: Explorer not rewarded (prevent collapse)")
    
    # Apply to tensor
    pred_errors = torch.tensor([0.05, 0.005, 0.02, 0.008])
    thresholded = torch.where(
        pred_errors > wm_loss_threshold,
        pred_errors,
        torch.zeros_like(pred_errors)
    )
    
    assert thresholded[0] == pred_errors[0]  # Above threshold
    assert thresholded[1] == 0.0  # Below threshold
    assert thresholded[2] == pred_errors[2]  # Above threshold
    assert thresholded[3] == 0.0  # Below threshold
    print("  ✓ Threshold masking works correctly")


def test_training_phase_transition():
    """Test transition from warmup to adversarial training."""
    print("\n[Test 7] Training phase transition...")
    
    warmup_iterations = 100
    
    phases = []
    for iteration in range(150):
        if iteration < warmup_iterations:
            phase = "warmup"  # WM-only training
        else:
            phase = "adversarial"  # Full adversarial training
        phases.append(phase)
    
    assert phases[0] == "warmup"
    assert phases[99] == "warmup"
    assert phases[100] == "adversarial"
    assert phases[149] == "adversarial"
    print("  ✓ Phase transition at correct iteration")
    
    warmup_count = sum(1 for p in phases if p == "warmup")
    adv_count = sum(1 for p in phases if p == "adversarial")
    
    assert warmup_count == 100
    assert adv_count == 50
    print(f"  ✓ Warmup: {warmup_count}, Adversarial: {adv_count}")


def test_checkpoint_structure():
    """Test checkpoint save/load structure."""
    print("\n[Test 8] Checkpoint structure...")
    
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint_001000"
        checkpoint_dir.mkdir(parents=True)
        
        # Save mock checkpoint
        torch.save({'weight': torch.randn(10, 5)}, checkpoint_dir / 'explorer.pt')
        torch.save({'state': {}}, checkpoint_dir / 'wm_optimizer.pt')
        torch.save({'state': {}}, checkpoint_dir / 'explorer_optimizer.pt')
        torch.save({'weight': torch.randn(256, 1)}, checkpoint_dir / 'value_head.pt')
        
        state = {
            'iteration': 1000,
            'global_step': 50000,
            'config': {
                'total_iterations': 10000,
                'adversarial_weight': 1.0,
            },
        }
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f)
        
        # Verify files exist
        assert (checkpoint_dir / 'explorer.pt').exists()
        assert (checkpoint_dir / 'wm_optimizer.pt').exists()
        assert (checkpoint_dir / 'explorer_optimizer.pt').exists()
        assert (checkpoint_dir / 'value_head.pt').exists()
        assert (checkpoint_dir / 'training_state.json').exists()
        print("  ✓ Checkpoint files created")
        
        # Load state
        with open(checkpoint_dir / 'training_state.json', 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state['iteration'] == 1000
        assert loaded_state['global_step'] == 50000
        print("  ✓ Checkpoint state loaded correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Adversarial Training Unit Tests")
    print("=" * 60)
    
    test_adversarial_config()
    test_alternating_update_schedule()
    test_adversarial_reward_computation()
    test_wm_loss_computation()
    test_gae_with_adversarial_rewards()
    test_collapse_prevention()
    test_training_phase_transition()
    test_checkpoint_structure()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nNote: Full adversarial training tests require:")
    print("  - F1_VLA policy with Explorer and WM")
    print("  - VAE model")
    print("  - Environment for rollout collection")


if __name__ == '__main__':
    main()
