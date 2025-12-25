"""
Custom dataset loader for ME_KVM_VLA data format.
Data format: List of dicts with keys ['obs', 'action', 'next_obs', 'reward', 'done', 'info']
obs keys: ['action_history', 'head_rgb', 'state', 'wrist_rgb']
"""
import os
import glob
import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MEKVMDataset(Dataset):
    """Dataset for ME_KVM_VLA data format (.pt files) with lazy loading and index caching"""
    
    def __init__(
        self,
        data_dir: str,
        n_obs_img_steps: int = 4,
        n_pred_img_steps: int = 1,
        chunk_size: int = 50,
        task_description: str = "perform the task",
    ):
        """
        Args:
            data_dir: Directory containing episode_*.pt files
            n_obs_img_steps: Number of observation image steps (history)
            n_pred_img_steps: Number of prediction image steps
            chunk_size: Action chunk size
            task_description: Default task description
        """
        self.data_dir = data_dir
        self.n_obs_img_steps = n_obs_img_steps
        self.n_pred_img_steps = n_pred_img_steps
        self.chunk_size = chunk_size
        self.task_description = task_description
        
        # Find all episode files
        self.episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
        if not self.episode_files:
            raise ValueError(f"No episode files found in {data_dir}")
        
        # Try to load cached index
        cache_file = os.path.join(data_dir, ".mekvm_index_cache.json")
        if os.path.exists(cache_file):
            print(f"Loading cached index from {cache_file}...")
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            self._episode_lengths = cache['episode_lengths']
            # Verify cache is still valid
            if len(self._episode_lengths) != len(self.episode_files):
                print("Cache invalid, re-indexing...")
                self._build_index(cache_file)
        else:
            self._build_index(cache_file)
        
        # Build sample index from episode lengths
        self.sample_index = []
        for ep_idx, num_steps in enumerate(self._episode_lengths):
            for step_idx in range(n_obs_img_steps - 1, num_steps - chunk_size):
                self.sample_index.append((ep_idx, step_idx))
        
        # Cache for loaded episodes (LRU-style)
        self._cache = {}
        self._cache_max_size = 10  # Keep only 10 episodes in memory
        
        print(f"Dataset ready: {len(self.sample_index)} samples from {len(self.episode_files)} episodes")
    
    def _build_index(self, cache_file: str):
        """Build and cache episode lengths"""
        self._episode_lengths = []
        print(f"Indexing {len(self.episode_files)} episodes from {self.data_dir}...")
        
        for ep_idx, ep_file in enumerate(self.episode_files):
            if (ep_idx + 1) % 100 == 0:
                print(f"  Indexed {ep_idx + 1}/{len(self.episode_files)} episodes...")
            episode = torch.load(ep_file, weights_only=False)
            self._episode_lengths.append(len(episode))
            del episode
        
        # Save cache
        print(f"Saving index cache to {cache_file}...")
        cache = {'episode_lengths': self._episode_lengths}
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    
    def _load_episode(self, ep_idx: int):
        """Load episode with caching"""
        if ep_idx not in self._cache:
            # Evict oldest if cache full
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[ep_idx] = torch.load(
                self.episode_files[ep_idx], 
                weights_only=False
            )
        return self._cache[ep_idx]
    
    def __len__(self):
        return len(self.sample_index)
    
    def _resize_to_256(self, img: torch.Tensor) -> torch.Tensor:
        """Resize image(s) from 224x224 to 256x256 for VAE compatibility.
        
        Args:
            img: Tensor of shape (..., 3, 224, 224)
        Returns:
            Tensor of shape (..., 3, 256, 256)
        """
        original_shape = img.shape
        if len(original_shape) == 3:
            # Single image (3, 224, 224) -> (1, 3, 224, 224) for interpolate
            img = img.unsqueeze(0)
            result = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            return result.squeeze(0)
        elif len(original_shape) == 4:
            # Batch of images (N, 3, 224, 224)
            return F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected image shape: {original_shape}")
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        ep_idx, step_idx = self.sample_index[idx]
        episode = self._load_episode(ep_idx)
        
        # Get observation images (history)
        # head_rgb: (4, 3, 224, 224) - already has history dim
        # wrist_rgb: (4, 3, 224, 224) - already has history dim
        obs = episode[step_idx]['obs']
        
        # Images - convert from uint8 to float and normalize to [0, 1]
        head_rgb = torch.from_numpy(obs['head_rgb']).float() / 255.0  # (4, 3, 224, 224)
        wrist_rgb = torch.from_numpy(obs['wrist_rgb']).float() / 255.0  # (4, 3, 224, 224)
        
        # Resize to 256x256 for VAE compatibility (VAE expects 256x256 -> 16x16 feature map)
        head_rgb = self._resize_to_256(head_rgb)  # (4, 3, 256, 256)
        wrist_rgb = self._resize_to_256(wrist_rgb)  # (4, 3, 256, 256)
        
        # State
        state = torch.from_numpy(obs['state']).float()  # (32,)
        
        # Actions - get chunk_size future actions
        actions = []
        for i in range(self.chunk_size):
            if step_idx + i < len(episode):
                action = episode[step_idx + i]['action']
                actions.append(action)
            else:
                # Pad with last action
                actions.append(actions[-1] if actions else episode[step_idx]['action'])
        
        actions = torch.tensor(actions, dtype=torch.float32)  # (chunk_size, action_dim)
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        
        # For world model: get prediction images (next frames)
        # Use head_rgb from next steps for prediction target
        pred_images = []
        for i in range(self.n_pred_img_steps):
            next_step = min(step_idx + 1 + i, len(episode) - 1)
            next_obs = episode[next_step]['obs']
            # Take the last frame from head_rgb history
            next_img = torch.from_numpy(next_obs['head_rgb'][-1]).float() / 255.0  # (3, 224, 224)
            next_img = self._resize_to_256(next_img)  # (3, 256, 256)
            pred_images.append(next_img)
        pred_images = torch.stack(pred_images)  # (n_pred_img_steps, 3, 256, 256)
        
        # Combine history and prediction images for world model
        # history: (n_obs_img_steps, 3, 256, 256), pred: (n_pred_img_steps, 3, 256, 256)
        # Combined: (n_obs_img_steps + n_pred_img_steps, 3, 256, 256)
        history_and_pred = torch.cat([head_rgb, pred_images], dim=0)  # (5, 3, 256, 256) for 4+1
        
        sample = {
            # Main observation image (last frame of history)
            "observation.images.image0": head_rgb[-1],  # (3, 256, 256) - current frame
            # History images + prediction targets for world model
            # This is used by prepare_mix_history_images which needs all frames
            "observation.images.image0_history": history_and_pred,  # (n_obs+n_pred, 3, 256, 256)
            # Wrist camera
            "observation.images.image1": wrist_rgb[-1],  # (3, 256, 256)
            "observation.images.image1_history": wrist_rgb,  # (4, 3, 256, 256)
            # State
            "observation.state": state,  # (32,)
            # Actions
            "action": actions,  # (chunk_size, action_dim)
            "action_is_pad": action_is_pad,  # (chunk_size,)
            # Task
            "task": self.task_description,
            # World model prediction target (kept for reference)
            "observation.images.image0_target": pred_images,  # (n_pred_img_steps, 3, 256, 256)
        }
        
        return sample


class MEKVMMixtureDataset(Dataset):
    """Wrapper to make MEKVMDataset compatible with F1-VLA training pipeline"""
    
    def __init__(
        self,
        data_dirs: List[str],
        n_obs_img_steps: int = 4,
        n_pred_img_steps: int = 1,
        chunk_size: int = 50,
        task_descriptions: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        stage: str = "stage1_pretrain_wm",
    ):
        """
        Args:
            data_dirs: List of directories containing episode files
            task_descriptions: Optional list of task descriptions for each data_dir
            weights: Optional sampling weights for each dataset
            stage: Training stage
        """
        self.stage = stage
        self.datasets = []
        
        if task_descriptions is None:
            task_descriptions = ["perform the task"] * len(data_dirs)
        
        for data_dir, task_desc in zip(data_dirs, task_descriptions):
            ds = MEKVMDataset(
                data_dir=data_dir,
                n_obs_img_steps=n_obs_img_steps,
                n_pred_img_steps=n_pred_img_steps,
                chunk_size=chunk_size,
                task_description=task_desc,
            )
            self.datasets.append(ds)
        
        # Compute cumulative lengths
        self.cumulative_lengths = [0]
        for ds in self.datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(ds))
        
        # Compute weights
        if weights is None:
            weights = [1.0] * len(data_dirs)
        total_weight = sum(weights)
        self.weights = [w / total_weight for w in weights]
        
        print(f"MEKVMMixtureDataset: {len(self.datasets)} datasets, {len(self)} total samples")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_len in enumerate(self.cumulative_lengths[1:], 1):
            if idx < cum_len:
                dataset_idx = i - 1
                break
        
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return (dataset_idx, self.datasets[dataset_idx][local_idx])


@dataclass
class MEKVMCollateFn:
    """Collate function for ME_KVM dataset"""
    max_state_dim: int = 50
    max_action_dim: int = 50
    suffix: str = "history"
    image_size: tuple = (224, 224)
    
    def __call__(self, items):
        dataset_idx = [x[0] for x in items]
        items = [x[1] for x in items]
        
        batch = {"dataset_idx": torch.tensor(dataset_idx, dtype=torch.long)}
        
        # Observation images
        batch["observation.images.image0"] = torch.stack([x["observation.images.image0"] for x in items])
        batch["observation.images.image0_mask"] = torch.ones(len(items), dtype=torch.bool)
        batch["observation.images.image0_history"] = torch.stack([x["observation.images.image0_history"] for x in items])
        
        if "observation.images.image1" in items[0]:
            batch["observation.images.image1"] = torch.stack([x["observation.images.image1"] for x in items])
            batch["observation.images.image1_mask"] = torch.ones(len(items), dtype=torch.bool)
            batch["observation.images.image1_history"] = torch.stack([x["observation.images.image1_history"] for x in items])
        
        # State with padding
        states = [x["observation.state"] for x in items]
        padded_states = []
        state_masks = []
        for s in states:
            length = s.shape[0]
            if length < self.max_state_dim:
                pad_size = self.max_state_dim - length
                padded = torch.cat([s, torch.zeros(pad_size, dtype=s.dtype)])
                mask = torch.cat([torch.ones(length, dtype=torch.bool), torch.zeros(pad_size, dtype=torch.bool)])
            else:
                padded = s[:self.max_state_dim]
                mask = torch.ones(self.max_state_dim, dtype=torch.bool)
            padded_states.append(padded)
            state_masks.append(mask)
        
        batch["observation.state"] = torch.stack(padded_states)
        batch["observation.state_mask"] = torch.stack(state_masks)
        
        # Actions with padding
        actions = [x["action"] for x in items]
        padded_actions = []
        action_masks = []
        for a in actions:
            action_dim = a.shape[1]
            if action_dim < self.max_action_dim:
                pad_size = self.max_action_dim - action_dim
                padded = torch.cat([a, torch.zeros(a.shape[0], pad_size, dtype=a.dtype)], dim=-1)
                mask = torch.cat([
                    torch.ones(a.shape[0], action_dim, dtype=torch.bool),
                    torch.zeros(a.shape[0], pad_size, dtype=torch.bool)
                ], dim=-1)
            else:
                padded = a[:, :self.max_action_dim]
                mask = torch.ones(a.shape[0], self.max_action_dim, dtype=torch.bool)
            padded_actions.append(padded)
            action_masks.append(mask)
        
        batch["action"] = torch.stack(padded_actions)
        batch["action_mask"] = torch.stack(action_masks)
        batch["action_is_pad"] = torch.stack([x["action_is_pad"] for x in items])
        
        # Task
        batch["task"] = [x["task"] for x in items]
        
        return batch


def create_mekvm_dataset(
    data_dirs: List[str],
    n_obs_img_steps: int = 4,
    n_pred_img_steps: int = 1,
    chunk_size: int = 50,
    task_descriptions: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    stage: str = "stage1_pretrain_wm",
):
    """Factory function to create ME_KVM dataset"""
    return MEKVMMixtureDataset(
        data_dirs=data_dirs,
        n_obs_img_steps=n_obs_img_steps,
        n_pred_img_steps=n_pred_img_steps,
        chunk_size=chunk_size,
        task_descriptions=task_descriptions,
        weights=weights,
        stage=stage,
    )
