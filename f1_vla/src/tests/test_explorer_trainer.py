"""
Unit Tests for Explorer RL Trainer Module
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.explorer_trainer import (
    ExplorerTrainingConfig,
    PPOValueHead,
    ExplorerRLTrainer,
)


def test_training_config():
    """Test ExplorerTrainingConfig."""
    print("\n[Test 1] ExplorerTrainingConfig...")
    
    config = ExplorerTrainingConfig()
    assert config.num_episodes == 1000
    assert config.ppo_epochs == 4
    assert config.clip_epsilon == 0.2
    assert config.gamma == 0.99
    print("  ✓ Default config works")
    
    config = ExplorerTrainingConfig(
        num_episodes=500,
        learning_rate=1e-5,
        clip_epsilon=0.1,
    )
    assert config.num_episodes == 500
    assert config.learning_rate == 1e-5
    assert config.clip_epsilon == 0.1
    print("  ✓ Custom config works")
    
    # Test to_dict
    d = config.to_dict()
    assert 'num_episodes' in d
    assert d['learning_rate'] == 1e-5
    print("  ✓ to_dict works")


def test_value_head():
    """Test PPOValueHead."""
    print("\n[Test 2] PPOValueHead...")
    
    value_head = PPOValueHead(input_dim=1024, hidden_dim=256)
    
    # Test forward
    state_emb = torch.randn(4, 1024)
    values = value_head(state_emb)
    
    assert values.shape == (4,), f"Expected shape (4,), got {values.shape}"
    print("  ✓ Value head forward works")
    
    # Test gradient
    values.sum().backward()
    assert value_head.net[0].weight.grad is not None
    print("  ✓ Gradient computation works")


def test_gae_computation():
    """Test GAE computation logic."""
    print("\n[Test 3] GAE computation...")
    
    # Simple test case
    gamma = 0.99
    gae_lambda = 0.95
    
    rewards = [1.0, 1.0, 1.0, 0.0]  # Terminal at step 4
    values = [0.9, 0.9, 0.9, 0.0]
    dones = [False, False, False, True]
    
    # Manual GAE computation
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = 0.0
        else:
            next_val = values[t + 1]
        
        delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    
    assert len(advantages) == 4
    assert len(returns) == 4
    print("  ✓ GAE computation works")
    
    # Terminal step should have delta = r - v (no future)
    assert abs(advantages[-1] - (rewards[-1] - values[-1])) < 1e-6
    print("  ✓ Terminal step advantage correct")
    
    # Returns should be advantage + value
    for i in range(len(returns)):
        assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6
    print("  ✓ Returns computation correct")


def test_ppo_loss_computation():
    """Test PPO loss computation logic."""
    print("\n[Test 4] PPO loss computation...")
    
    batch_size = 8
    action_dim = 7
    clip_epsilon = 0.2
    
    # Mock data
    old_log_probs = torch.randn(batch_size)
    new_log_probs = old_log_probs + torch.randn(batch_size) * 0.1
    advantages = torch.randn(batch_size)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Compute ratio
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    assert policy_loss.dim() == 0  # Scalar
    print("  ✓ Policy loss is scalar")
    
    # Value loss
    values = torch.randn(batch_size)
    returns = values + torch.randn(batch_size) * 0.1
    value_loss = torch.nn.functional.mse_loss(values, returns)
    
    assert value_loss.dim() == 0
    print("  ✓ Value loss is scalar")
    
    # Entropy bonus
    log_std = torch.zeros(action_dim)
    std = torch.exp(log_std)
    entropy = 0.5 * (1 + torch.log(2 * np.pi * std ** 2)).sum()
    
    assert entropy.dim() == 0
    assert entropy > 0
    print("  ✓ Entropy computation works")


def test_action_sampling():
    """Test action sampling logic."""
    print("\n[Test 5] Action sampling...")
    
    batch_size = 4
    action_dim = 7
    action_scale = 1.0
    
    # Mock action mean and std
    action_mean = torch.randn(batch_size, action_dim)
    log_std = torch.zeros(action_dim) - 1.0  # std ≈ 0.37
    std = torch.exp(log_std)
    
    # Create distribution
    dist = torch.distributions.Normal(action_mean, std)
    
    # Sample actions
    actions = dist.rsample()
    log_probs = dist.log_prob(actions).sum(dim=-1)
    
    assert actions.shape == (batch_size, action_dim)
    assert log_probs.shape == (batch_size,)
    print("  ✓ Action sampling works")
    
    # Clamp actions
    actions_clamped = torch.clamp(actions, -1.0, 1.0) * action_scale
    assert (actions_clamped >= -action_scale).all()
    assert (actions_clamped <= action_scale).all()
    print("  ✓ Action clamping works")
    
    # Evaluate existing actions
    eval_actions = torch.randn(batch_size, action_dim).clamp(-1.0, 1.0)
    eval_log_probs = dist.log_prob(eval_actions).sum(dim=-1)
    assert eval_log_probs.shape == (batch_size,)
    print("  ✓ Action evaluation works")


def test_mini_batch_iteration():
    """Test mini-batch iteration logic."""
    print("\n[Test 6] Mini-batch iteration...")
    
    batch_size = 100
    mini_batch_size = 32
    
    # Shuffle indices
    indices = torch.randperm(batch_size)
    
    num_batches = 0
    total_samples = 0
    
    for start in range(0, batch_size, mini_batch_size):
        end = min(start + mini_batch_size, batch_size)
        mb_indices = indices[start:end]
        
        assert len(mb_indices) <= mini_batch_size
        total_samples += len(mb_indices)
        num_batches += 1
    
    assert total_samples == batch_size
    expected_batches = (batch_size + mini_batch_size - 1) // mini_batch_size
    assert num_batches == expected_batches
    print(f"  ✓ Mini-batch iteration works ({num_batches} batches)")


def test_gradient_clipping():
    """Test gradient clipping."""
    print("\n[Test 7] Gradient clipping...")
    
    # Create simple model
    model = nn.Linear(10, 5)
    
    # Create large gradients
    x = torch.randn(4, 10)
    y = model(x)
    loss = (y * 1000).sum()  # Large loss to create large gradients
    loss.backward()
    
    # Check gradients are large
    grad_norm_before = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert grad_norm_before > 100
    print(f"  ✓ Gradient norm before clipping: {grad_norm_before:.2f}")
    
    # Apply clipping
    max_norm = 1.0
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Verify clipping worked
    actual_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert actual_norm <= max_norm * 1.01  # Small tolerance
    print(f"  ✓ Gradient norm after clipping: {actual_norm:.4f}")


def test_learning_rate_schedule():
    """Test learning rate scheduling logic."""
    print("\n[Test 8] Learning rate scheduling...")
    
    # Test cosine schedule formula
    num_episodes = 100
    lr_init = 1e-4
    lr_min = 1e-6
    
    lrs = []
    for step in range(num_episodes + 1):
        # Cosine annealing formula
        lr = lr_min + (lr_init - lr_min) * (1 + np.cos(np.pi * step / num_episodes)) / 2
        lrs.append(lr)
    
    # LR should decrease
    assert lrs[-1] < lrs[0]
    assert lrs[-1] >= lr_min
    print(f"  ✓ LR schedule: {lrs[0]:.2e} -> {lrs[-1]:.2e}")
    
    # Middle should be roughly halfway
    mid_lr = lrs[num_episodes // 2]
    expected_mid = (lr_init + lr_min) / 2
    assert abs(mid_lr - expected_mid) < lr_init * 0.1
    print(f"  ✓ Mid-point LR: {mid_lr:.2e} (expected ~{expected_mid:.2e})")


def test_checkpoint_structure():
    """Test checkpoint save/load structure."""
    print("\n[Test 9] Checkpoint structure...")
    
    import tempfile
    import json
    from pathlib import Path
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "checkpoint_000100"
        checkpoint_dir.mkdir(parents=True)
        
        # Save mock checkpoint
        torch.save({'weight': torch.randn(10, 5)}, checkpoint_dir / 'explorer.pt')
        torch.save({'weight': torch.randn(256, 1)}, checkpoint_dir / 'value_head.pt')
        torch.save({'state': {}}, checkpoint_dir / 'optimizer.pt')
        torch.save(torch.randn(7), checkpoint_dir / 'log_std.pt')
        
        state = {
            'episode': 100,
            'global_step': 5000,
            'config': {'lr': 1e-4},
            'metrics': {'rewards': [1.0, 2.0, 3.0]},
        }
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f)
        
        # Verify files exist
        assert (checkpoint_dir / 'explorer.pt').exists()
        assert (checkpoint_dir / 'value_head.pt').exists()
        assert (checkpoint_dir / 'training_state.json').exists()
        print("  ✓ Checkpoint files created")
        
        # Load state
        with open(checkpoint_dir / 'training_state.json', 'r') as f:
            loaded_state = json.load(f)
        
        assert loaded_state['episode'] == 100
        assert loaded_state['global_step'] == 5000
        print("  ✓ Checkpoint state loaded correctly")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Explorer RL Trainer Unit Tests")
    print("=" * 60)
    
    test_training_config()
    test_value_head()
    test_gae_computation()
    test_ppo_loss_computation()
    test_action_sampling()
    test_mini_batch_iteration()
    test_gradient_clipping()
    test_learning_rate_schedule()
    test_checkpoint_structure()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nNote: Full trainer tests require:")
    print("  - F1_VLA policy with Explorer actor")
    print("  - VAE model")
    print("  - Environment for rollout collection")


if __name__ == '__main__':
    main()
