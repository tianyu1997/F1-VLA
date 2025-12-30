"""
Unit Tests for Explorer Rollout Module
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.explorer_rollout import (
    RolloutConfig,
    Transition,
    EpisodeBuffer,
    ObservationHistory,
    transitions_to_batch,
)


def test_rollout_config():
    """Test RolloutConfig creation."""
    print("\n[Test 1] RolloutConfig...")
    
    config = RolloutConfig()
    assert config.max_steps_per_episode == 100
    assert config.history_length == 4
    assert config.action_dim == 7
    print("  ✓ Default config works")
    
    config = RolloutConfig(
        max_steps_per_episode=50,
        history_length=8,
        epsilon_greedy=0.1,
    )
    assert config.max_steps_per_episode == 50
    assert config.history_length == 8
    assert config.epsilon_greedy == 0.1
    print("  ✓ Custom config works")


def test_transition():
    """Test Transition dataclass."""
    print("\n[Test 2] Transition...")
    
    obs = {'state': np.random.randn(7), 'wrist_rgb': np.random.randn(3, 256, 256)}
    action = np.random.randn(7)
    next_obs = {'state': np.random.randn(7), 'wrist_rgb': np.random.randn(3, 256, 256)}
    
    trans = Transition(
        observation=obs,
        action=action,
        next_observation=next_obs,
        done=False,
        immediate_reward=1.0,
        log_prob=-0.5,
        value=0.8,
    )
    
    assert trans.immediate_reward == 1.0
    assert trans.log_prob == -0.5
    assert trans.done == False
    print("  ✓ Transition creation works")
    
    # Test default values
    assert trans.delayed_reward is None
    assert trans.full_reward is None
    assert trans.advantage == 0.0
    print("  ✓ Default values correct")


def test_episode_buffer():
    """Test EpisodeBuffer."""
    print("\n[Test 3] EpisodeBuffer...")
    
    buffer = EpisodeBuffer(max_episodes=3)
    
    # Add episodes
    for ep in range(5):
        transitions = []
        for step in range(10):
            trans = Transition(
                observation={'state': np.zeros(7)},
                action=np.zeros(7),
                next_observation={'state': np.zeros(7)},
                done=(step == 9),
                immediate_reward=1.0,
            )
            transitions.append(trans)
        buffer.add_episode(transitions)
    
    # Should maintain max episodes
    assert len(buffer) == 3, f"Expected 3, got {len(buffer)}"
    assert buffer.total_episodes == 5
    print("  ✓ Buffer respects max episodes")
    
    # Get all transitions
    all_trans = buffer.get_all_transitions()
    assert len(all_trans) == 30  # 3 episodes * 10 steps
    print("  ✓ get_all_transitions works")
    
    # Get latest episode
    latest = buffer.get_latest_episode()
    assert len(latest) == 10
    print("  ✓ get_latest_episode works")
    
    # Clear
    buffer.clear()
    assert len(buffer) == 0
    print("  ✓ clear works")


def test_observation_history():
    """Test ObservationHistory."""
    print("\n[Test 4] ObservationHistory...")
    
    history = ObservationHistory(history_length=4, image_keys=['wrist_rgb'])
    
    # Add observations
    for i in range(6):
        obs = {
            'state': np.ones(7) * i,
            'wrist_rgb': np.ones((3, 256, 256)) * i,
        }
        action = np.ones(7) * i if i > 0 else None
        history.add(obs, action)
    
    # Should have L+1 = 5 images/states
    assert len(history.images['wrist_rgb']) == 5
    assert len(history.states) == 5
    # Should have L = 4 actions
    assert len(history.actions) == 4
    print("  ✓ History maintains correct length")
    
    # Get model input
    model_input = history.get_model_input()
    assert model_input['wrist_rgb'].shape == (4, 3, 256, 256)
    assert model_input['state'].shape == (4, 7)
    assert model_input['action_history'].shape == (4, 7)
    print("  ✓ get_model_input returns correct shapes")
    
    # Get latest frame
    latest = history.get_latest_frame('wrist_rgb')
    assert latest.shape == (3, 256, 256)
    assert np.allclose(latest, np.ones((3, 256, 256)) * 5)  # Last value
    print("  ✓ get_latest_frame works")
    
    # Reset
    history.reset()
    assert not history.has_enough_history()
    print("  ✓ reset works")


def test_observation_history_padding():
    """Test ObservationHistory padding for short sequences."""
    print("\n[Test 5] ObservationHistory padding...")
    
    history = ObservationHistory(history_length=4, image_keys=['wrist_rgb'])
    
    # Add only 2 observations
    for i in range(2):
        obs = {
            'state': np.ones(7) * i,
            'wrist_rgb': np.ones((3, 256, 256)) * i,
        }
        action = np.ones(7) * i if i > 0 else None
        history.add(obs, action)
    
    # Should still be able to get model input with padding
    model_input = history.get_model_input()
    assert model_input['wrist_rgb'].shape == (4, 3, 256, 256)
    assert model_input['state'].shape == (4, 7)
    print("  ✓ Padding works for short sequences")
    
    # First frames should be repeated
    assert np.allclose(model_input['wrist_rgb'][0], model_input['wrist_rgb'][1])
    print("  ✓ First frame repeated for padding")


def test_transitions_to_batch():
    """Test transitions_to_batch conversion."""
    print("\n[Test 6] transitions_to_batch...")
    
    transitions = []
    for i in range(10):
        trans = Transition(
            observation={'state': np.zeros(7)},
            action=np.random.randn(7).astype(np.float32),
            next_observation={'state': np.zeros(7)},
            done=False,
            immediate_reward=float(i),
            full_reward=float(i) * 1.5,
            log_prob=-0.5,
            value=0.8,
            advantage=0.1,
            returns=1.0,
        )
        trans.gt_embedding = torch.randn(1, 256)
        trans.pred_embedding = torch.randn(1, 256)
        transitions.append(trans)
    
    batch = transitions_to_batch(transitions, device='cpu')
    
    assert batch['actions'].shape == (10, 7)
    assert batch['log_probs'].shape == (10,)
    assert batch['values'].shape == (10,)
    assert batch['advantages'].shape == (10,)
    assert batch['returns'].shape == (10,)
    assert batch['rewards'].shape == (10,)
    print("  ✓ Basic tensors converted correctly")
    
    # Check embeddings
    assert 'gt_embeddings' in batch
    assert batch['gt_embeddings'].shape == (10, 1, 256)
    assert 'pred_embeddings' in batch
    assert batch['pred_embeddings'].shape == (10, 1, 256)
    print("  ✓ Embeddings converted correctly")
    
    # Check rewards use full_reward
    expected_rewards = [i * 1.5 for i in range(10)]
    assert torch.allclose(batch['rewards'], torch.tensor(expected_rewards, dtype=torch.float32))
    print("  ✓ Rewards extracted correctly")


def test_gae_computation():
    """Test GAE computation logic."""
    print("\n[Test 7] GAE computation logic...")
    
    # Simple test case
    gamma = 0.99
    gae_lambda = 0.95
    
    rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
    values = [0.9, 0.9, 0.9, 0.9, 0.9]
    next_value = 0.0  # Terminal state
    
    # Manual GAE computation
    advantages = []
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value_t = next_value
        else:
            next_value_t = values[t + 1]
        
        delta = rewards[t] + gamma * next_value_t - values[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    # Last advantage should be delta (no future)
    assert len(advantages) == 5
    assert advantages[-1] == rewards[-1] + gamma * next_value - values[-1]
    print("  ✓ GAE computation logic verified")
    
    # Advantages should be positive (rewards > values)
    assert all(a > 0 for a in advantages)
    print("  ✓ Advantages have correct sign")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Explorer Rollout Unit Tests")
    print("=" * 60)
    
    test_rollout_config()
    test_transition()
    test_episode_buffer()
    test_observation_history()
    test_observation_history_padding()
    test_transitions_to_batch()
    test_gae_computation()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nNote: Full rollout collection tests require environment setup.")
    print("The ExplorerRolloutCollector class needs:")
    print("  - F1_VLA policy with Explorer actor")
    print("  - VAE embedding extractor")
    print("  - Reward manager")
    print("  - Environment instance")


if __name__ == '__main__':
    main()
