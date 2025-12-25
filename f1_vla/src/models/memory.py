"""
Memory Module for F1-VLA with KV-Cache based Memory Bank.

Design based on prompt.md requirements:
1. Memory Bank: Stores previous memory for each (dataset_idx, episode_idx)
2. Init Memory: Learnable parameters for frame_idx==0 initialization
3. KV Cache Prefix: Memory is prepended to transformer KV cache
4. Memory Token: Special token appended to input, its output becomes memory_info
5. GRU Update: Combines memory_info + previous_memory -> current_memory
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class KVMemoryBank(nn.Module):
    """
    Key-Value Memory Bank with GRU-based updates for F1-VLA.
    
    The memory is stored in KV-cache format and prepended to the transformer's
    attention computation. Memory is updated using a GRU cell based on the
    output of a special memory token.
    
    Memory State Format:
        List of (key, value) tuples, one per transformer layer.
        Each tensor has shape: (batch, memory_len, num_kv_heads, head_dim)
    
    Args:
        num_layers: Number of transformer layers
        num_kv_heads: Number of key-value attention heads
        head_dim: Dimension of each attention head
        hidden_size: Hidden size of the model (for memory token and GRU)
        memory_len: Number of memory slots per layer
        init_std: Standard deviation for parameter initialization
    """
    
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        memory_len: int = 4,
        init_std: float = 0.02,
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.memory_len = memory_len
        
        # Total KV dimension for memory state
        self.kv_dim = num_kv_heads * head_dim
        
        # Learnable initial memory (for frame_idx == 0)
        # Shape: (num_layers, 2, memory_len, num_kv_heads, head_dim)
        # [layer, 0=key/1=value, slot, head, dim]
        self.init_memory = nn.Parameter(
            torch.randn(num_layers, 2, memory_len, num_kv_heads, head_dim) * init_std
        )
        
        # Learnable memory token (appended to input sequence)
        # Its output will be used as memory_info for GRU update
        self.memory_token = nn.Parameter(
            torch.randn(1, 1, hidden_size) * init_std
        )
        
        # GRU for memory update
        # Input: memory_info from memory token output (hidden_size -> head_dim projection)
        # Hidden: flattened memory state (head_dim per slot)
        self.memory_info_proj = nn.Linear(hidden_size, head_dim)
        self.memory_gru = nn.GRUCell(
            input_size=head_dim,
            hidden_size=head_dim
        )
        
        # Runtime memory bank storage
        # Key: (dataset_idx, episode_idx) -> memory_state
        # Memory state is stored as List of (K, V) tensors per layer
        self._memory_bank: Dict[Tuple[int, int], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        logger.info(
            f"Initialized KVMemoryBank: layers={num_layers}, "
            f"heads={num_kv_heads}, head_dim={head_dim}, "
            f"hidden_size={hidden_size}, memory_len={memory_len}"
        )
    
    def get_memory_token(
        self, 
        batch_size: int, 
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Get memory token for appending to input sequence.
        
        Args:
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
            
        Returns:
            Memory token tensor: (batch_size, 1, hidden_size)
        """
        return self.memory_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()
    
    def get_initial_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get initial memory state from learnable parameters.
        Used when frame_idx == 0.
        
        Args:
            batch_size: Batch size
            device: Target device
            dtype: Target dtype
            
        Returns:
            List of (key, value) tuples, one per layer.
            Each tensor: (batch, memory_len, num_kv_heads, head_dim)
        """
        init_mem = self.init_memory.to(device=device, dtype=dtype)
        
        memory_state = []
        for layer_idx in range(self.num_layers):
            # Extract K and V for this layer
            k = init_mem[layer_idx, 0]  # (memory_len, heads, dim)
            v = init_mem[layer_idx, 1]  # (memory_len, heads, dim)
            
            # Expand batch dimension with contiguous() to make actual copies
            # This prevents inplace operation issues during backprop
            k = k.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            v = v.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
            
            memory_state.append((k, v))
        
        return memory_state
    
    def get_previous_memory(
        self,
        dataset_indices: torch.Tensor,
        episode_indices: torch.Tensor,
        frame_indices: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Retrieve previous memory for each sample in the batch.
        
        For frame_idx == 0, uses init_memory.
        For frame_idx > 0, retrieves from memory bank.
        
        Args:
            dataset_indices: (batch,) dataset index for each sample
            episode_indices: (batch,) episode index for each sample
            frame_indices: (batch,) frame index for each sample
            device: Target device
            dtype: Target dtype
            
        Returns:
            List of (key, value) tuples, one per layer.
            Each tensor: (batch, memory_len, num_kv_heads, head_dim)
        """
        batch_size = len(dataset_indices)
        
        # Initialize with init_memory for all samples
        memory_state = self.get_initial_memory(batch_size, device, dtype)
        
        # Override with stored memory for samples with frame_idx > 0
        for b in range(batch_size):
            ds_idx = dataset_indices[b].item()
            ep_idx = episode_indices[b].item()
            fr_idx = frame_indices[b].item()
            
            if fr_idx > 0:
                key = (ds_idx, ep_idx)
                if key in self._memory_bank:
                    stored_memory = self._memory_bank[key]
                    for layer_idx in range(self.num_layers):
                        # Copy stored memory into this batch position
                        k_stored, v_stored = stored_memory[layer_idx]
                        memory_state[layer_idx][0][b] = k_stored[0].to(device=device, dtype=dtype)
                        memory_state[layer_idx][1][b] = v_stored[0].to(device=device, dtype=dtype)
        
        return memory_state
    
    def update_memory(
        self,
        previous_memory: List[Tuple[torch.Tensor, torch.Tensor]],
        memory_info: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Update memory using GRU mechanism.
        
        Args:
            previous_memory: List of (K, V) tuples per layer
                Each tensor: (batch, memory_len, num_kv_heads, head_dim)
            memory_info: Output hidden state from memory token
                Shape: (batch, hidden_size)
                
        Returns:
            Updated memory state in same format
        """
        batch_size = memory_info.shape[0]
        device = memory_info.device
        dtype = memory_info.dtype
        
        # Move modules to device and convert input to float32 for computation
        memory_info_f32 = memory_info.float()
        
        # Project memory_info to head_dim
        proj = self.memory_info_proj.to(device).float()
        memory_info_proj = proj(memory_info_f32)  # (batch, head_dim)
        
        # Flatten all memory slots for GRU update
        # We treat each memory slot as a hidden state and update with the same input
        flat_slots = []
        for k, v in previous_memory:
            # k, v: (batch, memory_len, heads, dim)
            flat_slots.append(k.reshape(batch_size, -1, self.head_dim).clone())
            flat_slots.append(v.reshape(batch_size, -1, self.head_dim).clone())
        
        # Concatenate: (batch, total_slots, head_dim)
        memory_slots = torch.cat(flat_slots, dim=1).float()
        num_slots = memory_slots.shape[1]
        
        # Move GRU to device and float32
        gru = self.memory_gru.to(device).float()
        
        # Update each batch sample's memory slots
        updated_slots_list = []
        for b in range(batch_size):
            slot_b = memory_slots[b].clone()  # (num_slots, head_dim)
            inp_b = memory_info_proj[b:b+1].expand(num_slots, -1).contiguous()
            
            # Apply GRU: each slot updated with same memory_info
            new_slot_b = gru(inp_b, slot_b)
            updated_slots_list.append(new_slot_b)
        
        # Stack: (batch, num_slots, head_dim)
        updated_slots = torch.stack(updated_slots_list, dim=0)
        
        # Convert back to original dtype
        updated_slots = updated_slots.to(dtype)
        
        # Reshape back to layer-wise K, V format
        new_memory = []
        slots_per_kv = self.num_kv_heads * self.memory_len
        
        idx = 0
        for _ in range(self.num_layers):
            # Extract K
            k_flat = updated_slots[:, idx:idx + slots_per_kv, :]
            idx += slots_per_kv
            new_k = k_flat.view(
                batch_size, self.memory_len,
                self.num_kv_heads, self.head_dim
            ).contiguous()
            
            # Extract V
            v_flat = updated_slots[:, idx:idx + slots_per_kv, :]
            idx += slots_per_kv
            new_v = v_flat.view(
                batch_size, self.memory_len,
                self.num_kv_heads, self.head_dim
            ).contiguous()
            
            new_memory.append((new_k, new_v))
        
        return new_memory
    
    def store_memory(
        self,
        dataset_indices: torch.Tensor,
        episode_indices: torch.Tensor,
        memory_state: List[Tuple[torch.Tensor, torch.Tensor]],
        detach: bool = True,
    ) -> None:
        """
        Store updated memory in the memory bank.
        
        Args:
            dataset_indices: (batch,) dataset index for each sample
            episode_indices: (batch,) episode index for each sample
            memory_state: Memory state to store
            detach: Whether to detach memory from computation graph (for BPTT)
        """
        batch_size = len(dataset_indices)
        
        for b in range(batch_size):
            ds_idx = dataset_indices[b].item()
            ep_idx = episode_indices[b].item()
            key = (ds_idx, ep_idx)
            
            # Extract this sample's memory and store
            sample_memory = []
            for layer_idx in range(self.num_layers):
                k = memory_state[layer_idx][0][b:b+1]  # Keep batch dim: (1, mem_len, heads, dim)
                v = memory_state[layer_idx][1][b:b+1]
                
                if detach:
                    k = k.detach().clone()
                    v = v.detach().clone()
                else:
                    k = k.clone()
                    v = v.clone()
                
                sample_memory.append((k, v))
            
            self._memory_bank[key] = sample_memory
    
    def clear_memory_bank(self) -> None:
        """Clear all stored memory states."""
        self._memory_bank.clear()
        logger.info("Memory bank cleared")
    
    def clear_episode_memory(
        self,
        dataset_idx: int,
        episode_idx: int,
    ) -> None:
        """Clear memory for a specific episode."""
        key = (dataset_idx, episode_idx)
        if key in self._memory_bank:
            del self._memory_bank[key]
    
    def memory_to_kv_cache_prefix(
        self,
        memory_state: List[Tuple[torch.Tensor, torch.Tensor]],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get memory as KV cache prefix for a specific layer.
        
        Args:
            memory_state: Full memory state
            layer_idx: Layer index
            
        Returns:
            (key, value) tensors to prepend to attention
            Each tensor: (batch, memory_len, num_kv_heads, head_dim)
        """
        return memory_state[layer_idx]
    
    def to_dtype(self, dtype: torch.dtype) -> None:
        """Convert all parameters to specified dtype."""
        self.init_memory.data = self.init_memory.data.to(dtype)
        self.memory_token.data = self.memory_token.data.to(dtype)
        self.memory_info_proj = self.memory_info_proj.to(dtype)
        self.memory_gru = self.memory_gru.to(dtype)


class MemoryManager:
    """
    Manager for handling memory operations during training.
    
    Handles:
    - BPTT truncation: Detach gradients based on bptt_steps config
    - Episode boundary detection: Reset memory when episode changes
    - Batch-level memory operations
    """
    
    def __init__(
        self,
        memory_bank: KVMemoryBank,
        bptt_steps: int = 8,
    ):
        self.memory_bank = memory_bank
        self.bptt_steps = bptt_steps
        
        # Track step count per episode for BPTT
        self._step_counts: Dict[Tuple[int, int], int] = {}
    
    def should_detach(
        self,
        dataset_idx: int,
        episode_idx: int,
        frame_idx: int,
    ) -> bool:
        """
        Determine if gradients should be detached for BPTT.
        
        Detach at frame_idx == 0 or when step_count reaches bptt_steps.
        The check happens BEFORE update_step_count is called.
        """
        if frame_idx == 0:
            return True
        
        key = (dataset_idx, episode_idx)
        step_count = self._step_counts.get(key, 0)
        
        # Detach when we're about to exceed bptt_steps
        # step_count is incremented in update, so check if next step would exceed
        return step_count >= self.bptt_steps
    
    def update_step_count(
        self,
        dataset_idx: int,
        episode_idx: int,
        frame_idx: int,
    ) -> None:
        """Update step count for BPTT tracking. Called AFTER forward pass."""
        key = (dataset_idx, episode_idx)
        
        if frame_idx == 0:
            # Start of episode
            self._step_counts[key] = 1  # First step after frame 0
        else:
            current = self._step_counts.get(key, 0) + 1
            if current > self.bptt_steps:
                # Reset after bptt_steps (detach was applied, start new segment)
                self._step_counts[key] = 1
            else:
                self._step_counts[key] = current
    
    def process_batch(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, bool]:
        """
        Process batch to get previous memory and memory token.
        
        Args:
            batch: Batch dictionary with dataset_idx, episode_idx, frame_idx
            device: Target device
            dtype: Target dtype
            
        Returns:
            - previous_memory: Memory state for this batch
            - memory_token: Token to append to input
            - should_detach: Whether to detach gradients
        """
        dataset_indices = batch["dataset_idx"]
        episode_indices = batch["episode_idx"]
        frame_indices = batch["frame_idx"]
        batch_size = len(dataset_indices)
        
        # Get previous memory
        previous_memory = self.memory_bank.get_previous_memory(
            dataset_indices, episode_indices, frame_indices,
            device=device, dtype=dtype
        )
        
        # Get memory token
        memory_token = self.memory_bank.get_memory_token(batch_size, device, dtype)
        
        # Determine if we should detach (check first sample as proxy)
        # In sequential batching, all samples in batch have same frame_idx
        should_detach = self.should_detach(
            dataset_indices[0].item(),
            episode_indices[0].item(),
            frame_indices[0].item()
        )
        
        return previous_memory, memory_token, should_detach
    
    def store_updated_memory(
        self,
        batch: Dict[str, Any],
        updated_memory: List[Tuple[torch.Tensor, torch.Tensor]],
        detach: bool = True,
    ) -> None:
        """
        Store updated memory and update step counts.
        
        Args:
            batch: Batch dictionary
            updated_memory: Updated memory state
            detach: Whether to detach from graph
        """
        dataset_indices = batch["dataset_idx"]
        episode_indices = batch["episode_idx"]
        frame_indices = batch["frame_idx"]
        
        # Store memory
        self.memory_bank.store_memory(
            dataset_indices, episode_indices, updated_memory, detach=detach
        )
        
        # Update step counts
        for b in range(len(dataset_indices)):
            self.update_step_count(
                dataset_indices[b].item(),
                episode_indices[b].item(),
                frame_indices[b].item()
            )
    
    def on_epoch_start(self) -> None:
        """Called at the start of each epoch."""
        self.memory_bank.clear_memory_bank()
        self._step_counts.clear()
        logger.info("Memory manager reset for new epoch")
