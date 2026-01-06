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
        # Project to separate inputs for each slot (num_layers * 2 * memory_len slots)
        num_total_slots = num_layers * 2 * memory_len
        self.num_total_slots = num_total_slots
        self.memory_info_proj = nn.Linear(hidden_size, head_dim * num_total_slots)
        self.memory_gru = nn.GRUCell(
            input_size=head_dim,
            hidden_size=head_dim
        )
        
        # Runtime memory bank storage
        # Key: (dataset_idx, episode_idx) -> memory_state
        # Memory state is stored as List of (K, V) tensors per layer
        self._memory_bank: Dict[Tuple[int, int], List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        # Maximum number of episodes to keep in memory bank
        # In SequentialBatchSampler, only batch_size episodes are active at a time
        # Set to a larger value to cache more episodes and avoid frequent pruning
        self._max_memory_bank_size = 512  # Increased to handle 1440 total episodes better
        
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
        token = self.memory_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1).contiguous()
        # Check and fix NaN in memory token (can happen if params are corrupted)
        if torch.isnan(token).any() or torch.isinf(token).any():
            logger.warning(f"[KVMemoryBank] memory_token has NaN/Inf! Replacing with zeros.")
            token = torch.zeros_like(token)
        return token
    
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
        
        # Check and fix NaN in init_memory (can happen if params are corrupted)
        if torch.isnan(init_mem).any() or torch.isinf(init_mem).any():
            nan_count = torch.isnan(init_mem).sum().item()
            inf_count = torch.isinf(init_mem).sum().item()
            logger.warning(f"[KVMemoryBank] init_memory has NaN/Inf! nan={nan_count}, inf={inf_count}. Replacing with zeros.")
            init_mem = torch.where(torch.isnan(init_mem) | torch.isinf(init_mem), 
                                   torch.zeros_like(init_mem), init_mem)
        
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
        # Clone to avoid in-place operation issues with leaf variables
        init_memory = self.get_initial_memory(batch_size, device, dtype)
        memory_state = []
        for k, v in init_memory:
            memory_state.append((k.clone(), v.clone()))
        
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
                        k_val = k_stored[0].detach().to(device=device, dtype=dtype)
                        v_val = v_stored[0].detach().to(device=device, dtype=dtype)
                        
                        # Check for NaN/Inf in stored memory and replace with init_memory
                        if torch.isnan(k_val).any() or torch.isinf(k_val).any():
                            logger.warning(f"[KVMemoryBank] Stored memory key has NaN/Inf at layer {layer_idx}, "
                                         f"sample {b}, key={key}. Using init_memory instead.")
                            # Keep the init_memory value (already in memory_state)
                            continue
                        if torch.isnan(v_val).any() or torch.isinf(v_val).any():
                            logger.warning(f"[KVMemoryBank] Stored memory value has NaN/Inf at layer {layer_idx}, "
                                         f"sample {b}, key={key}. Using init_memory instead.")
                            continue
                            
                        memory_state[layer_idx][0][b] = k_val
                        memory_state[layer_idx][1][b] = v_val
        
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
        
        # Check input for NaN/Inf
        if torch.isnan(memory_info).any() or torch.isinf(memory_info).any():
            logger.error(f"[KVMemoryBank] memory_info has NaN/Inf! Using zeros.")
            memory_info = torch.zeros_like(memory_info)
        
        # Move modules to device and convert input to float32 for computation
        memory_info_f32 = memory_info.float()
        
        # Clip memory_info to prevent extreme values
        memory_info_f32 = torch.clamp(memory_info_f32, -10.0, 10.0)
        
        # Project memory_info to separate inputs for each slot
        proj = self.memory_info_proj.to(device).float()
        memory_info_proj = proj(memory_info_f32)  # (batch, head_dim * num_total_slots)
        
        # Check projection for NaN/Inf
        if torch.isnan(memory_info_proj).any() or torch.isinf(memory_info_proj).any():
            logger.error(f"[KVMemoryBank] memory_info_proj has NaN/Inf after projection! Using zeros.")
            memory_info_proj = torch.zeros_like(memory_info_proj)
        
        # Clip projected values
        memory_info_proj = torch.clamp(memory_info_proj, -10.0, 10.0)
        
        # Reshape to (batch, num_total_slots, head_dim) - each slot gets different input
        memory_info_proj = memory_info_proj.view(batch_size, self.num_total_slots, self.head_dim)
        
        # Flatten all memory slots for GRU update
        # Each slot will now be updated with its own specific input vector
        flat_slots = []
        for layer_idx, (k, v) in enumerate(previous_memory):
            # k, v: (batch, memory_len, heads, dim)
            # Check for NaN/Inf and replace with zeros if found
            if torch.isnan(k).any() or torch.isinf(k).any():
                logger.error(f"[KVMemoryBank] previous_memory layer {layer_idx} key has NaN/Inf! Replacing with zeros.")
                k = torch.zeros_like(k)
            if torch.isnan(v).any() or torch.isinf(v).any():
                logger.error(f"[KVMemoryBank] previous_memory layer {layer_idx} value has NaN/Inf! Replacing with zeros.")
                v = torch.zeros_like(v)
            
            flat_slots.append(k.reshape(batch_size, -1, self.head_dim).clone())
            flat_slots.append(v.reshape(batch_size, -1, self.head_dim).clone())
        
        # Concatenate: (batch, total_slots, head_dim)
        memory_slots = torch.cat(flat_slots, dim=1).float()
        num_slots = memory_slots.shape[1]
        
        # Clip memory slots to prevent extreme values
        memory_slots = torch.clamp(memory_slots, -10.0, 10.0)
        
        # Move GRU to device and float32
        gru = self.memory_gru.to(device).float()
        
        # Update each batch sample's memory slots with slot-specific inputs
        updated_slots_list = []
        for b in range(batch_size):
            slot_b = memory_slots[b].clone()  # (num_slots, head_dim)
            inp_b = memory_info_proj[b]  # (num_slots, head_dim) - different for each slot!
            
            # Apply GRU: each slot updated with its own specific memory_info
            new_slot_b = gru(inp_b, slot_b)
            
            # Check for NaN/Inf after GRU and replace with original slot
            if torch.isnan(new_slot_b).any() or torch.isinf(new_slot_b).any():
                logger.warning(f"[KVMemoryBank] GRU output has NaN/Inf for batch {b}! Using previous memory.")
                new_slot_b = slot_b  # Revert to previous memory
            
            # Clip to prevent extreme values
            new_slot_b = torch.clamp(new_slot_b, -10.0, 10.0)
            
            updated_slots_list.append(new_slot_b)
        
        # Stack: (batch, num_slots, head_dim)
        updated_slots = torch.stack(updated_slots_list, dim=0)
        
        # Final safety check
        if torch.isnan(updated_slots).any() or torch.isinf(updated_slots).any():
            logger.error(f"[KVMemoryBank] updated_slots has NaN/Inf after GRU! Replacing with clamped memory_slots.")
            updated_slots = torch.clamp(memory_slots, -10.0, 10.0)
        
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
            
            # Final check: if NaN/Inf detected, use zeros
            if torch.isnan(new_k).any() or torch.isinf(new_k).any():
                logger.error(f"[KVMemoryBank] new_k has NaN/Inf in final reshape! Using zeros.")
                new_k = torch.zeros_like(new_k)
            if torch.isnan(new_v).any() or torch.isinf(new_v).any():
                logger.error(f"[KVMemoryBank] new_v has NaN/Inf in final reshape! Using zeros.")
                new_v = torch.zeros_like(new_v)
            
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
            
            # Check if memory has NaN before storing
            has_nan = False
            for layer_idx in range(self.num_layers):
                k = memory_state[layer_idx][0][b:b+1]
                v = memory_state[layer_idx][1][b:b+1]
                if torch.isnan(k).any() or torch.isinf(k).any():
                    has_nan = True
                    break
                if torch.isnan(v).any() or torch.isinf(v).any():
                    has_nan = True
                    break
            
            if has_nan:
                logger.warning(f"[KVMemoryBank] Skipping store for key={key} due to NaN/Inf in memory state")
                continue
            
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
        
        # Prune old episodes if memory bank is too large
        self._prune_memory_bank_if_needed()
    
    def _prune_memory_bank_if_needed(self) -> None:
        """Remove oldest episodes if memory bank exceeds max size."""
        if len(self._memory_bank) > self._max_memory_bank_size:
            # Remove oldest entries (first added)
            # Dict maintains insertion order in Python 3.7+
            num_to_remove = len(self._memory_bank) - self._max_memory_bank_size
            keys_to_remove = list(self._memory_bank.keys())[:num_to_remove]
            for key in keys_to_remove:
                del self._memory_bank[key]
            logger.debug(f"Pruned {num_to_remove} episodes from memory bank, now {len(self._memory_bank)} episodes")
    
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
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, List[bool]]:
        """
        Process batch to get previous memory and memory token.
        
        Args:
            batch: Batch dictionary with dataset_idx, episode_idx, frame_idx
            device: Target device
            dtype: Target dtype
            
        Returns:
            - previous_memory: Memory state for this batch
            - memory_token: Token to append to input
            - should_detach_list: List of per-sample detach flags
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
        
        # Determine detach flag for EACH sample in batch
        # Different samples may have different frame_idx or bptt step counts
        should_detach_list = []
        for b in range(batch_size):
            should_detach = self.should_detach(
                dataset_indices[b].item(),
                episode_indices[b].item(),
                frame_indices[b].item()
            )
            should_detach_list.append(should_detach)
        
        return previous_memory, memory_token, should_detach_list
    
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
