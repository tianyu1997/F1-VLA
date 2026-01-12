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
import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class SequentialMEKVMDataset(Dataset):
    """
    Sequential dataset for ME_KVM data format supporting memory-based training.
    
    Key differences from standard dataset:
    1. Returns (dataset_idx, episode_idx, frame_idx) for memory management
    2. Episodes are processed sequentially within batches
    3. Designed for BPTT training with memory
    4. Supports multiple data directories
    5. Supports distributed training - each rank loads only its portion
    6. Supports configurable multi-camera input
    """
    
    def __init__(
        self,
        data_dirs: List[str],
        dataset_idx: int = 0,
        n_obs_img_steps: int = 4,
        n_pred_img_steps: int = 1,
        chunk_size: int = 4,
        task_descriptions: Optional[List[str]] = None,
        rank: int = 0,
        world_size: int = 1,
        camera_config: Optional[Dict[str, Any]] = None,
        cache_max_size: int = 64,
        norm_stats: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Args:
            data_dirs: List of directories containing episode_*.pt files
            dataset_idx: Index of this dataset (for multi-dataset setup)
            n_obs_img_steps: Number of observation image steps (history)
            n_pred_img_steps: Number of prediction image steps
            chunk_size: Action chunk size
            task_descriptions: Task descriptions for each data directory
            rank: Current process rank for distributed training
            world_size: Total number of processes
            camera_config: Camera configuration dict
            cache_max_size: Maximum number of episodes to cache in memory
            norm_stats: Dictionary containing 'state_mean', 'state_std', 'action_mean', 'action_std'
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        self.data_dirs = data_dirs
        self.dataset_idx = dataset_idx
        self.n_obs_img_steps = n_obs_img_steps
        self.n_pred_img_steps = n_pred_img_steps
        self.chunk_size = chunk_size
        self.rank = rank
        self.world_size = world_size
        
        # Normalization statistics
        self.norm_stats = norm_stats
        self.do_normalize = False
        if norm_stats is not None:
            logger.info(f"[Rank {rank}] Normalization stats provided. Checking keys...")
            
            # Helper to safely load tensor
            def load_tensor(data, key_path, default=None):
                val = data
                for k in key_path:
                    if isinstance(val, dict) and k in val:
                        val = val[k]
                    elif hasattr(val, k): # OmegaConf
                        val = getattr(val, k)
                    else:
                        return default
                return torch.tensor(val, dtype=torch.float32)

            # Support both flat and nested structure
            if 'state_mean' in norm_stats:
                # Flat structure
                required_keys = ['state_mean', 'state_std', 'action_mean', 'action_std']
                if all(k in norm_stats for k in required_keys):
                    self.state_mean = torch.tensor(norm_stats['state_mean'], dtype=torch.float32)
                    self.state_std = torch.tensor(norm_stats['state_std'], dtype=torch.float32)
                    self.action_mean = torch.tensor(norm_stats['action_mean'], dtype=torch.float32)
                    self.action_std = torch.tensor(norm_stats['action_std'], dtype=torch.float32)
                    self.do_normalize = True
            elif 'state' in norm_stats and 'action' in norm_stats:
                # Nested structure (from yaml)
                try:
                    self.state_mean = load_tensor(norm_stats, ['state', 'mean'])
                    self.state_std = load_tensor(norm_stats, ['state', 'std'])
                    self.action_mean = load_tensor(norm_stats, ['action', 'mean'])
                    self.action_std = load_tensor(norm_stats, ['action', 'std'])
                    self.do_normalize = True
                except Exception as e:
                    logger.warning(f"[Rank {rank}] Error parsing nested normalization stats: {e}")

            if self.do_normalize:
                # Check for zero std and replace with 1.0 to avoid division by zero
                # Replace very small values with 1.0 (assuming they are constant features 0.0 with std 0.0, or 1.0 with std 0.0)
                # Actually if std is 0, normalization (x-mean)/std implies x must be mean, so result 0/0.
                # If we set std=1.0, result is (x-mean), which is 0. This is correct for constant features.
                self.state_std[self.state_std < 1e-8] = 1.0
                self.action_std[self.action_std < 1e-8] = 1.0
                logger.info(f"[Rank {rank}] Normalization ENABLED.")
            else:
                logger.warning(f"[Rank {rank}] Missing normalization keys. Normalization DISABLED.")
        
        # Camera configuration
        if camera_config is None:
            camera_config = {}
        self.camera_keys = camera_config.get("und_camera_keys", ["head_rgb", "wrist_rgb"])
        self.world_model_camera = camera_config.get("wm_camera_key", "head_rgb")
        
        if task_descriptions is None:
            task_descriptions = ["perform the task"] * len(data_dirs)
        self.task_descriptions = task_descriptions
        
        # Find all episode files from all directories (just paths, no loading)
        all_episode_files = []
        all_episode_dir_idx = []
        
        for dir_idx, data_dir in enumerate(data_dirs):
            ep_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
            if not ep_files:
                logger.warning(f"No episode files found in {data_dir}")
                continue
            for ep_file in ep_files:
                all_episode_files.append(ep_file)
                all_episode_dir_idx.append(dir_idx)
        
        if not all_episode_files:
            raise ValueError(f"No episode files found in any of: {data_dirs}")
        
        total_episodes = len(all_episode_files)
        
        # Distribute episodes across ranks - each rank gets a subset
        self.episode_files = []
        self.episode_dir_idx = []
        self.global_episode_idx = []  # Map local idx -> global idx
        
        for global_idx in range(total_episodes):
            if global_idx % world_size == rank:
                self.episode_files.append(all_episode_files[global_idx])
                self.episode_dir_idx.append(all_episode_dir_idx[global_idx])
                self.global_episode_idx.append(global_idx)
        
        if world_size > 1:
            logger.info(f"[Rank {rank}] Assigned {len(self.episode_files)}/{total_episodes} episodes")
        
        # Load episode lengths only for this rank's episodes
        self._episode_lengths = self._load_episode_lengths()
        
        # Build sample index: (local_episode_idx, frame_idx)
        self.sample_index = []
        for local_ep_idx, num_steps in enumerate(self._episode_lengths):
            for frame_idx in range(n_obs_img_steps - 1, num_steps - chunk_size):
                self.sample_index.append((local_ep_idx, frame_idx))
        
        # Episode cache (LRU style)
        self._cache = {}
        # Increase cache size to improve performance on servers with sufficient RAM
        # Assuming average episode size ~50MB, 64 episodes ~= 3.2GB RAM
        self._cache_max_size = cache_max_size
        
        # Calculate total frames for this rank (for debugging distributed balance)
        total_frames = len(self.sample_index)
        
        logger.info(f"[Rank {rank}] SequentialMEKVMDataset: {len(self.sample_index)} samples (frames) from {len(self.episode_files)} episodes. Cache size={self._cache_max_size}")
        logger.info(f"[Rank {rank}] Total frames: {total_frames}")
    
    def _load_episode_lengths(self) -> List[int]:
        """Load episode lengths for this rank's episodes, using cache when possible"""
        episode_lengths = []
        
        # Group episodes by directory for efficient cache loading
        dir_caches = {}
        for dir_idx, data_dir in enumerate(self.data_dirs):
            cache_file = os.path.join(data_dir, ".mekvm_index_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    dir_caches[data_dir] = json.load(f)['episode_lengths']
            else:
                dir_caches[data_dir] = None
        
        for local_idx, ep_file in enumerate(self.episode_files):
            dir_idx = self.episode_dir_idx[local_idx]
            data_dir = self.data_dirs[dir_idx]
            
            # Try to get from cache
            cache = dir_caches.get(data_dir)
            if cache is not None:
                # Extract episode number from filename
                ep_num = int(os.path.basename(ep_file).split('_')[1].split('.')[0])
                if ep_num < len(cache):
                    episode_lengths.append(cache[ep_num])
                    continue
            
            # Fallback: assume constant length 50 to skip slow loading
            # episode = torch.load(ep_file, weights_only=False)
            # episode_lengths.append(len(episode))
            # del episode
            episode_lengths.append(50)

        
        return episode_lengths
    
    def _get_task_description(self, ep_idx: int) -> str:
        """Get task description for episode"""
        dir_idx = self.episode_dir_idx[ep_idx]
        return self.task_descriptions[dir_idx]
    
    def _load_episode(self, ep_idx: int):
        """Load episode with caching"""
        if ep_idx not in self._cache:
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
            
            self._cache[ep_idx] = torch.load(
                self.episode_files[ep_idx],
                map_location='cpu',
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
        if img.shape[-2:] == (256, 256):
            return img
            
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
            - observation.images.image{i}: current frame for each camera
            - observation.images.image{i}_history: history frames for each camera
            - observation.images.image0_history: history + prediction frames for WM
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
        
        # Process all configured cameras
        camera_images = {}  # camera_key -> (history_tensor, current_frame)
        for cam_key in self.camera_keys:
            if cam_key in obs:
                # Normalize to [0, 1] for compatibility with ImageTransforms and Policy
                # Policy will convert to [-1, 1] internally for Vision Encoder
                cam_data = torch.from_numpy(obs[cam_key]).float() / 255.0
                cam_data = self._resize_to_256(cam_data)
                camera_images[cam_key] = cam_data
        
        # Get world model camera images (for history + prediction)
        wm_cam_key = self.world_model_camera
        if wm_cam_key not in camera_images:
            # Fallback to first available camera
            wm_cam_key = self.camera_keys[0] if self.camera_keys else "head_rgb"
        wm_history = camera_images.get(wm_cam_key, torch.zeros(self.n_obs_img_steps, 3, 256, 256))
        
        # Current state
        state = torch.from_numpy(obs['state']).float()
        if self.do_normalize:
            state = (state - self.state_mean) / self.state_std
        
        # State history - aligned with image history (n_obs_img_steps frames)
        state_history = []
        for i in range(self.n_obs_img_steps):
            hist_frame_idx = frame_idx - (self.n_obs_img_steps - 1 - i)
            if hist_frame_idx < 0:
                hist_state = torch.from_numpy(episode[0]['obs']['state']).float()
            else:
                hist_state = torch.from_numpy(episode[hist_frame_idx]['obs']['state']).float()
            
            if self.do_normalize:
                hist_state = (hist_state - self.state_mean) / self.state_std
            state_history.append(hist_state)
        state_history = torch.stack(state_history)  # (n_obs_img_steps, state_dim)
        
        # Action history - n_obs_img_steps actions aligned with state_history (a_{t-n+1}, ..., a_{t-1}, a_t)
        # This mirrors state_history so both have same temporal alignment
        # a_t is needed because we predict image at t+1 after executing action a_t
        action_dim = len(episode[0]['action'])
        action_history = []
        for i in range(self.n_obs_img_steps):
            hist_frame_idx = frame_idx - (self.n_obs_img_steps - 1 - i)
            if hist_frame_idx < 0:
                hist_action = torch.zeros(action_dim, dtype=torch.float32)
            else:
                hist_action = torch.tensor(episode[hist_frame_idx]['action'], dtype=torch.float32)
            
            if self.do_normalize:
                # For zero padding actions, we technically shouldn't normalize zeros, but 0 is usually close to mean if centered.
                # However, normalized values might not be 0.
                if hist_frame_idx >= 0:
                    hist_action = (hist_action - self.action_mean) / self.action_std
                else:
                    # For padding frames, keep as 0 or normalize 0?
                    # Usually better to keep as 0 if 0 represents "neutral" or "no action", 
                    # but network expects normalized distribution.
                    # Given Pad is 0, let's normalize it to consistency.
                    hist_action = (hist_action - self.action_mean) / self.action_std
            
            action_history.append(hist_action)
        action_history = torch.stack(action_history)  # (n_obs_img_steps, action_dim)
        
        # Future actions - get chunk_size future actions
        actions = []
        for i in range(self.chunk_size):
            if frame_idx + i < len(episode):
                action = episode[frame_idx + i]['action']
                action_t = torch.tensor(action, dtype=torch.float32)
                if self.do_normalize:
                     action_t = (action_t - self.action_mean) / self.action_std
                actions.append(action_t)
            else:
                last_action = actions[-1] if actions else torch.tensor(episode[frame_idx]['action'], dtype=torch.float32)
                if not actions and self.do_normalize: # if actions was empty, last_action is raw
                     last_action = (last_action - self.action_mean) / self.action_std
                actions.append(last_action)
        
        actions = torch.stack(actions)  # (chunk_size, action_dim)
        action_is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
        
        # Prediction images for world model (from world_model_camera)
        pred_images = []
        for i in range(self.n_pred_img_steps):
            next_step = min(frame_idx + 1 + i, len(episode) - 1)
            next_obs = episode[next_step]['obs']
            if wm_cam_key in next_obs:
                # Normalize to [0, 1]
                next_img = torch.from_numpy(next_obs[wm_cam_key][-1]).float() / 255.0
            else:
                next_img = torch.from_numpy(next_obs['head_rgb'][-1]).float() / 255.0
            next_img = self._resize_to_256(next_img)
            pred_images.append(next_img)
        pred_images = torch.stack(pred_images)
        
        # History for world model (only history, policy will append target)
        history_only = wm_history
        
        # Build sample dict with all cameras
        sample = {
            # State
            "observation.state": state,
            "observation.state_history": state_history,
            # Actions
            "action": actions,
            "action_is_pad": action_is_pad,
            "action_history": action_history,
            # Task
            "task": self._get_task_description(ep_idx),
            # World model specific (always use image0 naming for WM)
            "observation.images.image0_history": history_only,
            "observation.images.image0_target": pred_images,
            # Memory indices
            "dataset_idx": torch.tensor(self.dataset_idx, dtype=torch.int64),
            "episode_idx": torch.tensor(self.global_episode_idx[ep_idx], dtype=torch.int64),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int64),
        }
        
        # Add each camera's images with standard naming
        for i, cam_key in enumerate(self.camera_keys):
            if cam_key in camera_images:
                cam_data = camera_images[cam_key]
                sample[f"observation.images.image{i}"] = cam_data[-1]  # Current frame
                sample[f"observation.images.image{i}_mask"] = torch.tensor(True)
                sample[f"observation.images.image{i}_history"] = cam_data  # All history frames
            else:
                # Empty placeholder
                sample[f"observation.images.image{i}"] = torch.zeros(3, 256, 256)
                sample[f"observation.images.image{i}_mask"] = torch.tensor(False)
                sample[f"observation.images.image{i}_history"] = torch.zeros(self.n_obs_img_steps, 3, 256, 256)
        
        # Add empty image2 slot for compatibility if only 2 cameras
        if len(self.camera_keys) < 3:
            for i in range(len(self.camera_keys), 3):
                sample[f"observation.images.image{i}"] = torch.zeros(3, 256, 256)
                sample[f"observation.images.image{i}_mask"] = torch.tensor(False)
        
        return sample
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        ep_idx, frame_idx = self.sample_index[idx]
        return self.get_frame(ep_idx, frame_idx)


class SequentialBatchSampler(Sampler):
    """
    Sampler for sequential episode processing with distributed training support.
    
    - Shuffles episode order (not frame order within episodes)
    - Yields batches of parallel episodes at the same timestep
    - Handles variable episode lengths
    - Supports distributed training: each rank gets a subset of episodes
    
    Window design for memory-based training:
    - window_length = n_obs_img_steps + k_bptt (context + gradient frames)
    - stride = k_bptt (only gradient frames advance each step)
    - First n_obs_img_steps frames: history context (loss detached)
    - Last k_bptt frames: compute gradients
    
    Yields: List of global indices into the dataset
    """
    
    def __init__(
        self,
        dataset: SequentialMEKVMDataset,
        batch_size: int,
        chunk_size: int = 1,
        stride: int = None,  # Window stride, defaults to chunk_size if not specified
        shuffle_episodes: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = max(1, chunk_size)
        self.stride = stride if stride is not None else self.chunk_size  # Default: non-overlapping
        self.shuffle_episodes = shuffle_episodes
        self.drop_last = drop_last
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self._epoch = 0  # Track epoch for deterministic shuffling
        
        # Build episode to sample mapping
        # episode_samples[ep_idx] = list of global sample indices for that episode
        self.episode_samples: Dict[int, List[int]] = {}
        for global_idx, (ep_idx, frame_idx) in enumerate(dataset.sample_index):
            if ep_idx not in self.episode_samples:
                self.episode_samples[ep_idx] = []
            self.episode_samples[ep_idx].append(global_idx)
        
        self.num_episodes = len(self.episode_samples)
        self.episode_ids = list(self.episode_samples.keys())
        
        # Distribute episodes across ranks
        # Each rank gets every world_size-th episode
        self.local_episode_ids = [
            ep_id for i, ep_id in enumerate(self.episode_ids)
            if i % self.world_size == self.rank
        ]
        self.local_num_episodes = len(self.local_episode_ids)
        
        if self.world_size > 1:
            logger.info(f"[Rank {self.rank}] SequentialBatchSampler: {self.local_num_episodes}/{self.num_episodes} episodes assigned to this rank")
    
    def __iter__(self) -> Iterator[List[int]]:
        """
        Iterate over batches.
        
        Yields lists of global indices. Within each batch:
        - Episodes are processed in parallel
        - Frames within episodes are sequential
        """
        logger.info(f"[SequentialBatchSampler] Starting __iter__. num_episodes={self.local_num_episodes}, batch_size={self.batch_size}, epoch={self._epoch}")
        
        # Deterministic shuffling: use seed + epoch so all ranks produce consistent order
        episode_order = self.local_episode_ids.copy()
        if self.shuffle_episodes:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(episode_order)
            logger.debug(f"[SequentialBatchSampler] Shuffled episodes with seed {self.seed + self._epoch}. First 5: {episode_order[:5]}")
        
        count_batches = 0
        
        # Process episodes in batches
        for batch_start in range(0, self.local_num_episodes, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.local_num_episodes)
            batch_episodes = episode_order[batch_start:batch_end]
            
            if self.drop_last and len(batch_episodes) < self.batch_size:
                logger.debug(f"[SequentialBatchSampler] Dropping last batch of size {len(batch_episodes)}")
                continue
            
            # Get sample lists for each episode in batch
            batch_sample_lists = [self.episode_samples[ep_id] for ep_id in batch_episodes]
            if not batch_sample_lists:
                 logger.error(f"[SequentialBatchSampler] Found empty batch sample list! batch_episodes={batch_episodes}")
                 continue
                 
            max_len = max(len(samples) for samples in batch_sample_lists)
            
            # Yield frame by frame across episodes
            # Yield in chunks of consecutive frames (chunk_size) with stride
            # stride < chunk_size enables overlapping windows for memory context
            for window_start in range(0, max_len, self.stride):
                window_end = window_start + self.chunk_size
                batch_indices = []
                for samples in batch_sample_lists:
                    if window_end <= len(samples):
                        # Append consecutive indices for this episode
                        batch_indices.extend(samples[window_start:window_end])
                # Require full chunk to keep sequence length consistent
                if batch_indices and len(batch_indices) == self.chunk_size * len(batch_sample_lists):
                    yield batch_indices
                    count_batches += 1
                    
        logger.info(f"[SequentialBatchSampler] Finished __iter__. Yielded {count_batches} batches.")
    
    def __len__(self) -> int:
        # Accurate count: total batches for one pass through all episodes
        # This allows Trainer to correctly calculate epoch boundaries
        total_batches = 0
        for batch_start in range(0, self.local_num_episodes, self.batch_size):
            batch_end = min(batch_start + self.batch_size, self.local_num_episodes)
            batch_episodes = list(range(batch_start, batch_end))
            if self.drop_last and len(batch_episodes) < self.batch_size:
                continue
            # Count frames in this batch
            batch_sample_lists = [self.episode_samples[self.local_episode_ids[i]] for i in batch_episodes]
            max_len = max(len(samples) for samples in batch_sample_lists)
            # Calculate number of windows with stride
            if max_len >= self.chunk_size:
                total_batches += (max_len - self.chunk_size) // self.stride + 1
        return total_batches

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling across distributed workers."""
        if self.rank == 0:
             logger.info(f"SequentialBatchSampler from dataset: set epoch to {epoch} for shuffling")
        self._epoch = epoch


class SequentialCollateFn:
    """
    Collate function for sequential batches.
    Handles variable batch sizes and pads appropriately.
    """
    
    def __init__(self, max_state_dim: int = 32, max_action_dim: int = 32):
        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
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
    rank: int = 0,
    world_size: int = 1,
):
    """Create sequential dataset for memory-based training with distributed support."""
    from lerobot.datasets.transforms import (
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
    cache_max_size = dataset_config.get('cache_max_size', 64)
    norm_stats = dataset_config.get('normalization', None)
    
    if task_descriptions is None:
        task_descriptions = ["perform the task"] * len(data_dirs)
    
    # Get camera configuration from policy_config
    camera_config = None
    if hasattr(policy_config, 'camera_config'):
        camera_config = dict(policy_config.camera_config)
    
    # Create dataset with distributed support - each rank loads only its portion
    dataset = SequentialMEKVMDataset(
        data_dirs=data_dirs,
        dataset_idx=0,
        n_obs_img_steps=n_obs_img_steps,
        n_pred_img_steps=n_pred_img_steps,
        chunk_size=chunk_size,
        task_descriptions=task_descriptions,
        rank=rank,
        world_size=world_size,
        camera_config=camera_config,
        cache_max_size=cache_max_size,
        norm_stats=norm_stats,
    )
    
    # Sample weights (uniform)
    training_ds_weights = np.ones(len(dataset)) / len(dataset)
    
    return dataset, image_transforms, training_ds_weights, n_obs_img_steps, n_pred_img_steps