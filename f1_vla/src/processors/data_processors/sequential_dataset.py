"""
Sequential Dataset for Memory-based Training (BPTT).

When use_memory=True, data is loaded sequentially:
- Episode order is shuffled, but frames within episode are sequential
- Each sample includes dataset_idx, episode_idx, frame_idx
- Supports Truncated Backpropagation Through Time (BPTT)
"""
import os
import glob
import json
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


class SequentialMEKVMDataset(Dataset):
    """
    Sequential dataset for ME_KVM data format supporting memory-based training.
    
    Key differences from standard dataset:
    1. Returns (dataset_idx, episode_idx, frame_idx) for memory management
    2. Episodes are processed sequentially within batches
    3. Designed for BPTT training with memory
    """
    
    def __init__(
        self,
        data_dir: str,
        dataset_idx: int = 0,
        n_obs_img_steps: int = 4,
        n_pred_img_steps: int = 1,
        chunk_size: int = 4,
        task_description: str = "perform the task",
    ):
        """
        Args:
            data_dir: Directory containing episode_*.pt files
            dataset_idx: Index of this dataset (for multi-dataset setup)
            n_obs_img_steps: Number of observation image steps (history)
            n_pred_img_steps: Number of prediction image steps
            chunk_size: Action chunk size
            task_description: Default task description
        """
        self.data_dir = data_dir
        self.dataset_idx = dataset_idx
        self.n_obs_img_steps = n_obs_img_steps
        self.n_pred_img_steps = n_pred_img_steps
        self.chunk_size = chunk_size
        self.task_description = task_description
        
        # Find all episode files
        self.episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
        if not self.episode_files:
            raise ValueError(f"No episode files found in {data_dir}")
        
        # Load or build episode length index
        cache_file = os.path.join(data_dir, ".mekvm_index_cache.json")
        if os.path.exists(cache_file):
            print(f"Loading cached index from {cache_file}...")
            with open(cache_file, 'r') as f:
                cache = json.load(f)
            self._episode_lengths = cache['episode_lengths']
            if len(self._episode_lengths) != len(self.episode_files):
                print("Cache invalid, re-indexing...")
                self._build_index(cache_file)
        else:
            self._build_index(cache_file)
        
        # Build sample index: (episode_idx, frame_idx)
        # For sequential dataset, we keep track of valid frame ranges
        self.sample_index = []
        for ep_idx, num_steps in enumerate(self._episode_lengths):
            # Valid frames: need n_obs_img_steps-1 history and chunk_size future
            for frame_idx in range(n_obs_img_steps - 1, num_steps - chunk_size):
                self.sample_index.append((ep_idx, frame_idx))
        
        # Episode cache (LRU style)
        self._cache = {}
        self._cache_max_size = 10
        
        print(f"SequentialMEKVMDataset: {len(self.sample_index)} samples from {len(self.episode_files)} episodes")
    
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
        
        print(f"Saving index cache to {cache_file}...")
        cache = {'episode_lengths': self._episode_lengths}
        with open(cache_file, 'w') as f:
            json.dump(cache, f)
    
    def _load_episode(self, ep_idx: int):
        """Load episode with caching"""
        if ep_idx not in self._cache:
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[ep_idx] = torch.load(
                self.episode_files[ep_idx],
                weights_only=False
            )
        return self._cache[ep_idx]
    
    def get_episode_lengths(self) -> List[int]:
        """Get all episode lengths"""
        return self._episode_lengths
    
    def get_num_episodes(self) -> int:
        """Get number of episodes"""
        return len(self.episode_files)
    
    def _resize_to_256(self, img: torch.Tensor) -> torch.Tensor:
        """Resize image(s) from 224x224 to 256x256 for VAE compatibility."""
        original_shape = img.shape
        if len(original_shape) == 3:
            img = img.unsqueeze(0)
            result = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
            return result.squeeze(0)
        elif len(original_shape) == 4:
            return F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        else:
            raise ValueError(f"Unexpected image shape: {original_shape}")
    
    def get_frame(self, ep_idx: int, frame_idx: int) -> Dict[str, Any]:
        """
        Get a single frame with all necessary data.
        
        Returns dict with:
            - observation.images.image0: current frame
            - observation.images.image0_history: history + prediction frames
            - observation.state: state vector
            - observation.state_history: state history (n_obs_img_steps)
            - action: action chunk
            - action_history: action history (n_obs_img_steps - 1, previous actions)
            - action_is_pad: padding mask
            - task: task description
            - dataset_idx, episode_idx, frame_idx: indices for memory
        """
        episode = self._load_episode(ep_idx)
        obs = episode[frame_idx]['obs']
        
        # Images - convert and resize
        head_rgb = torch.from_numpy(obs['head_rgb']).float() / 255.0
        wrist_rgb = torch.from_numpy(obs['wrist_rgb']).float() / 255.0
        head_rgb = self._resize_to_256(head_rgb)
        wrist_rgb = self._resize_to_256(wrist_rgb)
        
        # Current state
        state = torch.from_numpy(obs['state']).float()
        
        # State history - aligned with image history (n_obs_img_steps frames)
        # Image history has n_obs_img_steps frames, state history should match
        state_history = []
        for i in range(self.n_obs_img_steps):
            hist_frame_idx = frame_idx - (self.n_obs_img_steps - 1 - i)
            if hist_frame_idx < 0:
                # Pad with first available state
                hist_state = torch.from_numpy(episode[0]['obs']['state']).float()
            else:
                hist_state = torch.from_numpy(episode[hist_frame_idx]['obs']['state']).float()
            state_history.append(hist_state)
        state_history = torch.stack(state_history)  # (n_obs_img_steps, state_dim)
        
        # Action history - previous n_obs_img_steps-1 actions (before current frame)
        # Note: we use n_obs_img_steps-1 because at timestep t, we have t-1 previous actions
        action_dim = len(episode[0]['action'])  # action is a list
        action_history = []
        for i in range(self.n_obs_img_steps - 1):
            hist_frame_idx = frame_idx - (self.n_obs_img_steps - 1 - i)
            if hist_frame_idx <= 0:
                # Pad with zeros for first frame
                hist_action = torch.zeros(action_dim, dtype=torch.float32)
            else:
                hist_action = torch.tensor(episode[hist_frame_idx - 1]['action'], dtype=torch.float32)
            action_history.append(hist_action)
        action_history = torch.stack(action_history) if action_history else torch.zeros(0, action_dim)  # (n_obs_img_steps-1, action_dim)
        
        # Future actions - get chunk_size future actions
        actions = []
        for i in range(self.chunk_size):
            if frame_idx + i < len(episode):
                action = episode[frame_idx + i]['action']
                actions.append(action)
            else:
                actions.append(actions[-1] if actions else episode[frame_idx]['action'])
        
        actions = torch.tensor(actions, dtype=torch.float32)
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        
        # Prediction images for world model
        pred_images = []
        for i in range(self.n_pred_img_steps):
            next_step = min(frame_idx + 1 + i, len(episode) - 1)
            next_obs = episode[next_step]['obs']
            next_img = torch.from_numpy(next_obs['head_rgb'][-1]).float() / 255.0
            next_img = self._resize_to_256(next_img)
            pred_images.append(next_img)
        pred_images = torch.stack(pred_images)
        
        # Combine history and prediction for world model
        history_and_pred = torch.cat([head_rgb, pred_images], dim=0)
        
        sample = {
            # Main observation image
            "observation.images.image0": head_rgb[-1],
            # History + prediction images for world model
            "observation.images.image0_history": history_and_pred,
            # Wrist camera
            "observation.images.image1": wrist_rgb[-1],
            "observation.images.image1_history": wrist_rgb,
            # State
            "observation.state": state,
            # State history (n_obs_img_steps, state_dim)
            "observation.state_history": state_history,
            # Actions
            "action": actions,
            "action_is_pad": action_is_pad,
            # Action history (n_obs_img_steps-1, action_dim)
            "action_history": action_history,
            # Task
            "task": self.task_description,
            # Prediction target
            "observation.images.image0_target": pred_images,
            # Memory indices
            "dataset_idx": torch.tensor(self.dataset_idx, dtype=torch.int64),
            "episode_idx": torch.tensor(ep_idx, dtype=torch.int64),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int64),
        }
        
        return sample
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        ep_idx, frame_idx = self.sample_index[idx]
        return self.get_frame(ep_idx, frame_idx)


class SequentialBatchSampler(Sampler):
    """
    Sampler for sequential episode processing.
    
    - Shuffles episode order (not frame order within episodes)
    - Yields batches of parallel episodes at the same timestep
    - Handles variable episode lengths
    
    Yields: List of global indices into the dataset
    """
    
    def __init__(
        self,
        dataset: SequentialMEKVMDataset,
        batch_size: int,
        shuffle_episodes: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle_episodes = shuffle_episodes
        self.drop_last = drop_last
        
        # Build episode to sample mapping
        # episode_samples[ep_idx] = list of global sample indices for that episode
        self.episode_samples: Dict[int, List[int]] = {}
        for global_idx, (ep_idx, frame_idx) in enumerate(dataset.sample_index):
            if ep_idx not in self.episode_samples:
                self.episode_samples[ep_idx] = []
            self.episode_samples[ep_idx].append(global_idx)
        
        self.num_episodes = len(self.episode_samples)
        self.episode_ids = list(self.episode_samples.keys())
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches.
        
        Yields lists of global indices. Within each batch:
        - Episodes are processed in parallel
        - Frames within episodes are sequential
        """
        # Optionally shuffle episode order
        episode_order = self.episode_ids.copy()
        if self.shuffle_episodes:
            random.shuffle(episode_order)
        
        # Process episodes in batches
        for batch_start in range(0, self.num_episodes, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.num_episodes)
            batch_episodes = episode_order[batch_start:batch_end]
            
            if self.drop_last and len(batch_episodes) < self.batch_size:
                continue
            
            # Get sample lists for each episode in batch
            batch_sample_lists = [self.episode_samples[ep_id] for ep_id in batch_episodes]
            max_len = max(len(samples) for samples in batch_sample_lists)
            
            # Yield frame by frame across episodes
            for frame_offset in range(max_len):
                batch_indices = []
                for samples in batch_sample_lists:
                    if frame_offset < len(samples):
                        batch_indices.append(samples[frame_offset])
                
                if batch_indices:
                    yield batch_indices
    
    def __len__(self) -> int:
        # Total number of batches (approximate)
        return len(self.dataset) // self.batch_size


class SequentialCollateFn:
    """
    Collate function for sequential batches.
    Handles variable batch sizes and pads appropriately.
    """
    
    def __init__(self, max_state_dim: int = 32, max_action_dim: int = 32):
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not batch:
            return {}
        
        result = {}
        keys = batch[0].keys()
        
        for key in keys:
            values = [item[key] for item in batch]
            
            if key == "task":
                result[key] = values
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
        
        # Pad state if needed
        if "observation.state" in result:
            state = result["observation.state"]
            if state.shape[-1] < self.max_state_dim:
                pad_size = self.max_state_dim - state.shape[-1]
                result["observation.state"] = F.pad(state, (0, pad_size), value=0)
        
        # Pad state history if needed
        if "observation.state_history" in result:
            state_hist = result["observation.state_history"]
            if state_hist.shape[-1] < self.max_state_dim:
                pad_size = self.max_state_dim - state_hist.shape[-1]
                result["observation.state_history"] = F.pad(state_hist, (0, pad_size), value=0)
        
        # Pad action if needed
        if "action" in result:
            action = result["action"]
            if action.shape[-1] < self.max_action_dim:
                pad_size = self.max_action_dim - action.shape[-1]
                result["action"] = F.pad(action, (0, pad_size), value=0)
        
        # Pad action history if needed
        if "action_history" in result:
            action_hist = result["action_history"]
            if action_hist.shape[-1] < self.max_action_dim:
                pad_size = self.max_action_dim - action_hist.shape[-1]
                result["action_history"] = F.pad(action_hist, (0, pad_size), value=0)
        
        return result


def create_sequential_mekvm_data(
    policy_config,
    dataset_config,
    training_args,
    stage: str,
):
    """Create sequential dataset for memory-based training."""
    from f1_vla.src.processors.data_processors.image_transforms import (
        ImageTransforms, ImageTransformsConfig
    )
    
    # Create image transforms
    img_trans_cfg = ImageTransformsConfig(
        enable=training_args.image_transforms_enabled,
        max_num_transforms=training_args.image_transforms_max_num_transforms,
        random_order=training_args.image_transforms_random_order,
    )
    filtered_tfs = {
        name: tf for name, tf in img_trans_cfg.tfs.items() 
        if name in training_args.image_transforms_type
    }
    img_trans_cfg.tfs = filtered_tfs
    image_transforms = ImageTransforms(img_trans_cfg)
    
    # Get config
    data_dirs = dataset_config.get('mekvm_data_dirs', [])
    task_descriptions = dataset_config.get('mekvm_task_descriptions', None)
    n_obs_img_steps = dataset_config.get('n_obs_img_steps', 4)
    n_pred_img_steps = dataset_config.get('n_pred_img_steps', 1)
    chunk_size = dataset_config.get('chunk_size', policy_config.chunk_size)
    
    if task_descriptions is None:
        task_descriptions = ["perform the task"] * len(data_dirs)
    
    # Create datasets - use first one for simplicity (extend for multiple later)
    if len(data_dirs) > 1:
        raise NotImplementedError("Multiple data directories not yet supported for sequential")
    
    dataset = SequentialMEKVMDataset(
        data_dir=data_dirs[0],
        dataset_idx=0,
        n_obs_img_steps=n_obs_img_steps,
        n_pred_img_steps=n_pred_img_steps,
        chunk_size=chunk_size,
        task_description=task_descriptions[0],
    )
    
    # Sample weights (uniform)
    training_ds_weights = np.ones(len(dataset)) / len(dataset)
    
    return dataset, image_transforms, training_ds_weights, n_obs_img_steps, n_pred_img_steps
