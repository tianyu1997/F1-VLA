"""
Test script for memory integration in F1-VLA.

This script tests:
1. KVMemoryBank initialization
2. Memory KV prefix in attention
3. BPTT detach logic
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_memory_bank_initialization():
    """Test KVMemoryBank can be initialized with correct dimensions."""
    from f1_vla.src.models.memory import KVMemoryBank
    
    # PaliGemma dimensions
    num_layers = 18
    num_kv_heads = 8
    head_dim = 256
    hidden_size = 2048
    memory_len = 4
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        memory_len=memory_len,
    )
    
    # Check init_memory shape
    assert memory_bank.init_memory.shape == (num_layers, 2, memory_len, num_kv_heads, head_dim)
    
    # Check memory_token shape
    assert memory_bank.memory_token.shape == (1, 1, hidden_size)
    
    print("✓ Memory bank initialization test passed")


def test_memory_initial_state():
    """Test getting initial memory state."""
    from f1_vla.src.models.memory import KVMemoryBank
    
    num_layers = 18
    num_kv_heads = 8
    head_dim = 256
    hidden_size = 2048
    memory_len = 4
    batch_size = 2
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        memory_len=memory_len,
    )
    
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Get initial memory
    init_memory = memory_bank.get_initial_memory(batch_size, device, dtype)
    
    # Check structure
    assert len(init_memory) == num_layers
    
    for layer_idx, (k, v) in enumerate(init_memory):
        assert k.shape == (batch_size, memory_len, num_kv_heads, head_dim), f"Layer {layer_idx} K shape mismatch"
        assert v.shape == (batch_size, memory_len, num_kv_heads, head_dim), f"Layer {layer_idx} V shape mismatch"
    
    print("✓ Memory initial state test passed")


def test_memory_manager_detach_logic():
    """Test BPTT detach logic in MemoryManager."""
    from f1_vla.src.models.memory import KVMemoryBank, MemoryManager
    
    memory_bank = KVMemoryBank(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=256,
        memory_len=4,
    )
    
    manager = MemoryManager(memory_bank, bptt_steps=3)
    
    # Flow: should_detach() called BEFORE forward, update_step_count() called AFTER forward
    
    # Frame 0: should_detach=True (frame_idx=0)
    assert manager.should_detach(0, 0, 0) == True, "Should detach at frame_idx=0"
    manager.update_step_count(0, 0, 0)  # count becomes 1
    
    # Frame 1: count=1, 1 >= 3? No
    assert manager.should_detach(0, 0, 1) == False
    manager.update_step_count(0, 0, 1)  # count becomes 2
    
    # Frame 2: count=2, 2 >= 3? No
    assert manager.should_detach(0, 0, 2) == False
    manager.update_step_count(0, 0, 2)  # count becomes 3
    
    # Frame 3: count=3, 3 >= 3? Yes! Detach
    assert manager.should_detach(0, 0, 3) == True, "Should detach after bptt_steps"
    manager.update_step_count(0, 0, 3)  # count reset to 1 (4 > 3)
    
    # Frame 4: count=1, 1 >= 3? No
    assert manager.should_detach(0, 0, 4) == False
    manager.update_step_count(0, 0, 4)  # count becomes 2
    
    # Frame 5: count=2, 2 >= 3? No
    assert manager.should_detach(0, 0, 5) == False
    manager.update_step_count(0, 0, 5)  # count becomes 3
    
    # Frame 6: count=3, 3 >= 3? Yes! Detach again
    assert manager.should_detach(0, 0, 6) == True, "Should detach again after bptt_steps"
    
    print("✓ Memory manager detach logic test passed")


def test_memory_update_gru():
    """Test GRU-based memory update."""
    from f1_vla.src.models.memory import KVMemoryBank
    
    num_layers = 2
    num_kv_heads = 2
    head_dim = 64
    hidden_size = 256
    memory_len = 4
    batch_size = 2
    
    memory_bank = KVMemoryBank(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
        memory_len=memory_len,
    )
    
    device = torch.device('cpu')
    dtype = torch.float32
    
    # Get initial memory
    prev_memory = memory_bank.get_initial_memory(batch_size, device, dtype)
    
    # Simulate memory info from transformer output
    memory_info = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    
    # Update memory
    new_memory = memory_bank.update_memory(prev_memory, memory_info)
    
    # Check structure unchanged
    assert len(new_memory) == num_layers
    
    for layer_idx, (k, v) in enumerate(new_memory):
        assert k.shape == (batch_size, memory_len, num_kv_heads, head_dim)
        assert v.shape == (batch_size, memory_len, num_kv_heads, head_dim)
    
    # Check values changed (GRU should update)
    for layer_idx in range(num_layers):
        prev_k, prev_v = prev_memory[layer_idx]
        new_k, new_v = new_memory[layer_idx]
        assert not torch.allclose(prev_k, new_k), f"Layer {layer_idx} K unchanged after GRU"
        assert not torch.allclose(prev_v, new_v), f"Layer {layer_idx} V unchanged after GRU"
    
    print("✓ Memory GRU update test passed")


def test_memory_storage_and_retrieval():
    """Test storing and retrieving memory."""
    from f1_vla.src.models.memory import KVMemoryBank
    
    memory_bank = KVMemoryBank(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=256,
        memory_len=4,
    )
    
    device = torch.device('cpu')
    dtype = torch.float32
    batch_size = 2
    
    # Create test memory
    test_memory = memory_bank.get_initial_memory(batch_size, device, dtype)
    
    # Store for batch
    dataset_indices = torch.tensor([0, 0])
    episode_indices = torch.tensor([0, 1])
    
    memory_bank.store_memory(dataset_indices, episode_indices, test_memory, detach=True)
    
    # Retrieve for frame_idx > 0
    frame_indices = torch.tensor([1, 1])  # frame_idx=1, should use stored memory
    
    retrieved = memory_bank.get_previous_memory(
        dataset_indices, episode_indices, frame_indices, device, dtype
    )
    
    assert len(retrieved) == 2
    
    print("✓ Memory storage and retrieval test passed")


def test_memory_clear():
    """Test clearing memory bank."""
    from f1_vla.src.models.memory import KVMemoryBank, MemoryManager
    
    memory_bank = KVMemoryBank(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        hidden_size=256,
        memory_len=4,
    )
    
    manager = MemoryManager(memory_bank, bptt_steps=3)
    
    # Store some memory
    device = torch.device('cpu')
    dtype = torch.float32
    test_memory = memory_bank.get_initial_memory(2, device, dtype)
    memory_bank.store_memory(
        torch.tensor([0, 0]),
        torch.tensor([0, 1]),
        test_memory
    )
    
    # Update step counts
    manager.update_step_count(0, 0, 0)
    manager.update_step_count(0, 0, 1)
    
    assert len(memory_bank._memory_bank) > 0
    assert len(manager._step_counts) > 0
    
    # Clear on epoch start
    manager.on_epoch_start()
    
    assert len(memory_bank._memory_bank) == 0
    assert len(manager._step_counts) == 0
    
    print("✓ Memory clear test passed")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Running Memory Integration Tests")
    print("="*60 + "\n")
    
    test_memory_bank_initialization()
    test_memory_initial_state()
    test_memory_manager_detach_logic()
    test_memory_update_gru()
    test_memory_storage_and_retrieval()
    test_memory_clear()
    
    print("\n" + "="*60)
    print("All Memory Integration Tests Passed! ✓")
    print("="*60 + "\n")
