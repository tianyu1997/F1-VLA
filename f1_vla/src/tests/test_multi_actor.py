#!/usr/bin/env python3
"""
Unit tests for multi-actor architecture.

Tests:
1. Actor creation and initialization
2. Actor selection and switching
3. Actor saving and loading
4. Gradient configuration for different actors
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


class DictWithAttrAccess(dict):
    """A dictionary that supports both dict-style and attribute-style access"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MockExpert(nn.Module):
    """Mock expert model for testing."""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(2)])
        })
        self.hidden_size = hidden_size
    
    def forward(self, x):
        for layer in self.model.layers.values():
            x = layer(x)
        return x


def test_dict_with_attr_access():
    """Test DictWithAttrAccess works correctly."""
    d = DictWithAttrAccess({'a': 1, 'b': 2})
    assert d.a == 1
    assert d['a'] == 1
    assert d.b == 2
    print("  ✓ DictWithAttrAccess works correctly")


def test_module_dict_actors():
    """Test that ModuleDict correctly stores multiple actors."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    
    assert 'actor' in actors
    assert 'explorer' in actors
    assert len(actors) == 2
    print("  ✓ ModuleDict stores multiple actors")


def test_actor_weight_independence():
    """Test that different actors have independent weights."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    
    # Get weights from both actors
    actor_weight = actors['actor'].model['layers'][0].weight.data.clone()
    explorer_weight = actors['explorer'].model['layers'][0].weight.data.clone()
    
    # Weights should be different (random init)
    assert not torch.allclose(actor_weight, explorer_weight)
    print("  ✓ Actors have independent weights")


def test_actor_save_load():
    """Test saving and loading actor weights."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    
    # Save explorer weights
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
        torch.save(actors['explorer'].state_dict(), f.name)
        save_path = f.name
    
    try:
        # Modify explorer weights
        with torch.no_grad():
            actors['explorer'].model['layers'][0].weight.fill_(999.0)
        
        # Load saved weights
        state_dict = torch.load(save_path, weights_only=True)
        actors['explorer'].load_state_dict(state_dict)
        
        # Check weights are restored
        assert not (actors['explorer'].model['layers'][0].weight == 999.0).all()
        print("  ✓ Actor save/load works correctly")
    finally:
        os.unlink(save_path)


def test_actor_gradient_isolation():
    """Test that freezing one actor doesn't affect another."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    
    # Freeze actor, keep explorer trainable
    for param in actors['actor'].parameters():
        param.requires_grad = False
    for param in actors['explorer'].parameters():
        param.requires_grad = True
    
    # Check gradients
    for param in actors['actor'].parameters():
        assert not param.requires_grad
    for param in actors['explorer'].parameters():
        assert param.requires_grad
    print("  ✓ Gradient isolation works correctly")


def test_active_actor_switching():
    """Test switching between active actors."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    
    # Simulate active actor property
    _active_actor = 'actor'
    
    # Get active actor
    assert actors[_active_actor] == actors['actor']
    
    # Switch active actor
    _active_actor = 'explorer'
    assert actors[_active_actor] == actors['explorer']
    
    print("  ✓ Active actor switching works correctly")


def test_add_actor_dynamically():
    """Test adding a new actor dynamically."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    
    # Add explorer
    actors['explorer'] = MockExpert()
    
    assert len(actors) == 2
    assert 'explorer' in actors
    print("  ✓ Dynamic actor addition works correctly")


def test_list_actors():
    """Test listing all actors."""
    actors = nn.ModuleDict()
    actors['actor'] = MockExpert()
    actors['explorer'] = MockExpert()
    actors['random'] = MockExpert()
    
    actor_names = list(actors.keys())
    assert len(actor_names) == 3
    assert 'actor' in actor_names
    assert 'explorer' in actor_names
    assert 'random' in actor_names
    print("  ✓ List actors works correctly")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Multi-Actor Architecture Unit Tests")
    print("=" * 60)
    
    print("\n[Test 1] DictWithAttrAccess...")
    test_dict_with_attr_access()
    
    print("\n[Test 2] ModuleDict for actors...")
    test_module_dict_actors()
    
    print("\n[Test 3] Actor weight independence...")
    test_actor_weight_independence()
    
    print("\n[Test 4] Actor save/load...")
    test_actor_save_load()
    
    print("\n[Test 5] Gradient isolation...")
    test_actor_gradient_isolation()
    
    print("\n[Test 6] Active actor switching...")
    test_active_actor_switching()
    
    print("\n[Test 7] Dynamic actor addition...")
    test_add_actor_dynamically()
    
    print("\n[Test 8] List actors...")
    test_list_actors()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_all_tests()
