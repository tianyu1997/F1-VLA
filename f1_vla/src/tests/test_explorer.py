#!/usr/bin/env python3
"""
Unit tests for Explorer actor initialization.

Tests:
1. ExplorerConfig creation and serialization
2. Explorer initialization with random weights
3. Explorer training setup (gradient configuration)
4. Explorer parameter extraction
"""

import os
import sys
import tempfile
import torch
import torch.nn as nn

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
f1_vla_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, f1_vla_dir)


class MockPolicy:
    """Mock F1_VLA policy for testing Explorer."""
    
    def __init__(self):
        self._actors = nn.ModuleDict()
        self._actors['actor'] = self._create_mock_actor()
        self._active_actor = 'actor'
        self.model = MockModel(self)
    
    def _create_mock_actor(self):
        """Create a mock actor module."""
        return nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )
    
    @property
    def active_actor(self):
        return self._active_actor
    
    @active_actor.setter
    def active_actor(self, name):
        if name not in self._actors:
            raise ValueError(f"Actor '{name}' not found")
        self._active_actor = name
    
    def list_actors(self):
        return list(self._actors.keys())
    
    def get_actor(self, name):
        return self._actors[name]
    
    def add_actor(self, name, random_init=True):
        if name in self._actors:
            return
        new_actor = self._create_mock_actor()
        if not random_init:
            # Copy weights from active actor
            new_actor.load_state_dict(self._actors[self._active_actor].state_dict())
        self._actors[name] = new_actor
    
    def load_actor(self, name, path):
        state_dict = torch.load(path, weights_only=True)
        self._actors[name].load_state_dict(state_dict)
    
    def set_trainable_actors(self, names):
        for actor_name, actor in self._actors.items():
            for param in actor.parameters():
                param.requires_grad = actor_name in names
    
    def parameters(self):
        for actor in self._actors.values():
            yield from actor.parameters()


class MockModel:
    """Mock model for testing."""
    def __init__(self, policy):
        self.policy = policy
        self.paligemma_with_expert = MockPaliGemmaWithExpert()


class MockPaliGemmaWithExpert:
    """Mock PaliGemmaWithExpert for testing."""
    def __init__(self):
        self.gemma_wm_expert = nn.Linear(256, 256)
        self.paligemma = MockPaliGemma()


class MockPaliGemma:
    """Mock PaliGemma for testing."""
    def __init__(self):
        self.vision_tower = nn.Linear(256, 256)
        self.language_model = nn.Linear(256, 256)
    
    def eval(self):
        pass


def test_explorer_config_creation():
    """Test ExplorerConfig creation with defaults."""
    from f1_vla.src.models.explorer import ExplorerConfig
    
    config = ExplorerConfig()
    assert config.random_init == True
    assert config.actor_checkpoint is None
    assert config.reward_uncertainty_weight == 1.0
    assert config.reward_mse_weight == 1.0
    assert config.freeze_world_model == True
    print("  ✓ ExplorerConfig created with defaults")


def test_explorer_config_custom():
    """Test ExplorerConfig with custom values."""
    from f1_vla.src.models.explorer import ExplorerConfig
    
    config = ExplorerConfig(
        random_init=False,
        actor_checkpoint='/tmp/explorer.pth',
        reward_uncertainty_weight=2.0,
        reward_mse_weight=0.5,
        freeze_world_model=False,
    )
    
    assert config.random_init == False
    assert config.actor_checkpoint == '/tmp/explorer.pth'
    assert config.reward_uncertainty_weight == 2.0
    assert config.reward_mse_weight == 0.5
    assert config.freeze_world_model == False
    print("  ✓ ExplorerConfig created with custom values")


def test_explorer_config_serialization():
    """Test ExplorerConfig to_dict and from_dict."""
    from f1_vla.src.models.explorer import ExplorerConfig
    
    config = ExplorerConfig(
        random_init=False,
        reward_uncertainty_weight=2.0,
    )
    
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict['random_init'] == False
    assert config_dict['reward_uncertainty_weight'] == 2.0
    
    restored = ExplorerConfig.from_dict(config_dict)
    assert restored.random_init == config.random_init
    assert restored.reward_uncertainty_weight == config.reward_uncertainty_weight
    print("  ✓ ExplorerConfig serialization works")


def test_initialize_explorer():
    """Test Explorer initialization."""
    from f1_vla.src.models.explorer import initialize_explorer, ExplorerConfig
    
    policy = MockPolicy()
    assert 'explorer' not in policy.list_actors()
    
    initialize_explorer(policy)
    
    assert 'explorer' in policy.list_actors()
    assert len(policy.list_actors()) == 2
    print("  ✓ Explorer initialized successfully")


def test_initialize_explorer_with_checkpoint():
    """Test Explorer initialization from checkpoint."""
    from f1_vla.src.models.explorer import initialize_explorer, ExplorerConfig
    
    # Create a checkpoint
    mock_actor = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 32),
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(mock_actor.state_dict(), f.name)
        ckpt_path = f.name
    
    try:
        policy = MockPolicy()
        config = ExplorerConfig(
            random_init=True,
            actor_checkpoint=ckpt_path,
        )
        
        initialize_explorer(policy, config)
        
        assert 'explorer' in policy.list_actors()
        print("  ✓ Explorer initialized from checkpoint")
    finally:
        os.unlink(ckpt_path)


def test_setup_explorer_training():
    """Test Explorer training setup."""
    from f1_vla.src.models.explorer import setup_explorer_training, ExplorerConfig
    
    policy = MockPolicy()
    setup_explorer_training(policy)
    
    # Check explorer is active
    assert policy.active_actor == 'explorer'
    
    # Check only explorer is trainable
    actor_trainable = any(p.requires_grad for p in policy.get_actor('actor').parameters())
    explorer_trainable = any(p.requires_grad for p in policy.get_actor('explorer').parameters())
    
    assert not actor_trainable, "Actor should be frozen"
    assert explorer_trainable, "Explorer should be trainable"
    print("  ✓ Explorer training setup works")


def test_get_explorer_parameters():
    """Test getting Explorer parameters."""
    from f1_vla.src.models.explorer import (
        initialize_explorer, 
        get_explorer_parameters,
        ExplorerConfig,
    )
    
    policy = MockPolicy()
    initialize_explorer(policy)
    
    params = get_explorer_parameters(policy)
    
    assert len(params) > 0
    assert all(isinstance(p, torch.nn.Parameter) for p in params)
    print("  ✓ Explorer parameters retrieved successfully")


def test_explorer_weight_independence():
    """Test that Explorer has independent weights from Actor."""
    from f1_vla.src.models.explorer import initialize_explorer, ExplorerConfig
    
    policy = MockPolicy()
    config = ExplorerConfig(random_init=True)  # Random init
    initialize_explorer(policy, config)
    
    # Get weights
    actor_weight = next(policy.get_actor('actor').parameters()).data.clone()
    explorer_weight = next(policy.get_actor('explorer').parameters()).data.clone()
    
    # Weights should be different
    assert not torch.allclose(actor_weight, explorer_weight)
    print("  ✓ Explorer has independent weights from Actor")


def test_explorer_copied_init():
    """Test Explorer initialization by copying Actor weights."""
    from f1_vla.src.models.explorer import initialize_explorer, ExplorerConfig
    
    policy = MockPolicy()
    config = ExplorerConfig(random_init=False)  # Copy init
    initialize_explorer(policy, config)
    
    # Get weights
    actor_weight = next(policy.get_actor('actor').parameters()).data.clone()
    explorer_weight = next(policy.get_actor('explorer').parameters()).data.clone()
    
    # Weights should be the same (copied)
    assert torch.allclose(actor_weight, explorer_weight)
    print("  ✓ Explorer copied weights from Actor")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Explorer Actor Unit Tests")
    print("=" * 60)
    
    print("\n[Test 1] ExplorerConfig creation with defaults...")
    test_explorer_config_creation()
    
    print("\n[Test 2] ExplorerConfig with custom values...")
    test_explorer_config_custom()
    
    print("\n[Test 3] ExplorerConfig serialization...")
    test_explorer_config_serialization()
    
    print("\n[Test 4] Initialize Explorer...")
    test_initialize_explorer()
    
    print("\n[Test 5] Initialize Explorer from checkpoint...")
    test_initialize_explorer_with_checkpoint()
    
    print("\n[Test 6] Setup Explorer training...")
    test_setup_explorer_training()
    
    print("\n[Test 7] Get Explorer parameters...")
    test_get_explorer_parameters()
    
    print("\n[Test 8] Explorer weight independence (random init)...")
    test_explorer_weight_independence()
    
    print("\n[Test 9] Explorer copied init...")
    test_explorer_copied_init()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_all_tests()
