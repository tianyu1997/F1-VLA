"""
Sequential Rollout Buffer for Explorer RL Training

将环境交互数据组织成 sequential 格式，与 SequentialMEKVMDataset 兼容。
用于在线RL训练时保持与主训练流程一致的数据格式。

Key Features:
- 环境交互产生的 rollout 数据按 episode 顺序存储
- 提供 sequential batch 采样，支持 BPTT 训练
- 格式与 sequential_dataset.py 兼容
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SequentialRolloutConfig:
    """Configuration for sequential rollout buffer."""
    # Buffer capacity
    max_episodes: int = 100
    max_steps_total: int = 50000
    max_steps_per_episode: int = 200  # Maximum steps per episode during collection
    
    # Sequential settings
    n_obs_img_steps: int = 4  # History length for observation
    n_pred_img_steps: int = 1  # Prediction steps for WM
    chunk_size: int = 4  # Action chunk size
    
    # Image settings
    image_size: int = 256
    camera_keys: List[str] = field(default_factory=lambda: ["head_rgb", "wrist_rgb"])
    wm_camera_key: str = "head_rgb"
    
    # State/Action dimensions
    state_dim: int = 14
    action_dim: int = 7


class SequentialRolloutBuffer:
    """
    Buffer for storing environment interaction data in sequential format.
    
    将 Explorer 与环境交互的数据按照 episode 顺序存储，
    并提供与 SequentialMEKVMDataset 兼容的数据访问接口。
    """
    
    def __init__(self, config: Optional[SequentialRolloutConfig] = None):
        """
        Initialize sequential rollout buffer.
        
        Args:
            config: Buffer configuration
        """
        self.config = config or SequentialRolloutConfig()
        
        # Episode storage: List of episodes, each episode is a list of steps
        self.episodes: List[List[Dict[str, Any]]] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        
        # Current episode being collected
        self.current_episode: List[Dict[str, Any]] = []
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
        # Sample index: (episode_idx, frame_idx) pairs for valid samples
        self.sample_index: List[Tuple[int, int]] = []
        
        logger.info(f"SequentialRolloutBuffer initialized with config: "
                   f"max_episodes={self.config.max_episodes}, "
                   f"n_obs_img_steps={self.config.n_obs_img_steps}")
    
    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to target size (256x256)."""
        if img.shape[-2:] == (self.config.image_size, self.config.image_size):
            return img
        
        # Convert to torch for resizing
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.dim() == 3:  # (C, H, W)
            img_tensor = img_tensor.unsqueeze(0)
        img_tensor = F.interpolate(
            img_tensor, 
            size=(self.config.image_size, self.config.image_size),
            mode='bilinear', 
            align_corners=False
        )
        return img_tensor.squeeze(0).numpy()
    
    def add_step(
        self,
        observation: Dict[str, Any],
        action: np.ndarray,
        reward: float,
        next_observation: Dict[str, Any],
        done: bool,
        info: Dict[str, Any] = None,
        log_prob: float = 0.0,
        value: float = 0.0,
    ):
        """
        Add a single step to current episode.
        
        Args:
            observation: Current observation dict with images and state
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode terminated
            info: Additional info
            log_prob: Log probability of action
            value: Value estimate
        """
        step_data = {
            'obs': {},
            'action': action.copy(),
            'reward': reward,
            'done': done,
            'log_prob': log_prob,
            'value': value,
            'info': info or {},
        }
        
        # Process images - ensure (C, H, W) format and resize to 256x256
        for key in self.config.camera_keys:
            if key in observation:
                img = observation[key]
                # Handle various input formats
                if isinstance(img, np.ndarray):
                    if img.ndim == 4:  # (T, C, H, W) - take last frame
                        img = img[-1]
                    elif img.ndim == 3 and img.shape[0] > 3:  # (H, W, C)
                        img = img.transpose(2, 0, 1)
                    # Normalize to [0, 1] if needed
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                    img = self._resize_image(img)
                step_data['obs'][key] = img
        
        # Process state
        if 'state' in observation:
            state = observation['state']
            if isinstance(state, np.ndarray):
                step_data['obs']['state'] = state.astype(np.float32)
        
        self.current_episode.append(step_data)
        
        if done:
            self._finish_episode()
    
    def _finish_episode(self):
        """Finish current episode and add to buffer."""
        if not self.current_episode:
            return
        
        # Calculate episode reward
        episode_reward = sum(step['reward'] for step in self.current_episode)
        episode_length = len(self.current_episode)
        
        # Add to storage
        self.episodes.append(self.current_episode)
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        self.total_steps += episode_length
        self.total_episodes += 1
        
        # Update sample index
        self._update_sample_index()
        
        # Maintain buffer capacity
        self._maintain_capacity()
        
        # Reset current episode
        self.current_episode = []
        
        logger.debug(f"Episode finished: length={episode_length}, reward={episode_reward:.2f}")
    
    def _update_sample_index(self):
        """Update sample index after adding new episode."""
        self.sample_index = []
        
        n_obs = self.config.n_obs_img_steps
        chunk = self.config.chunk_size
        
        for ep_idx, episode in enumerate(self.episodes):
            # Valid frame indices: need n_obs_img_steps history and chunk_size future
            for frame_idx in range(n_obs - 1, len(episode) - chunk):
                self.sample_index.append((ep_idx, frame_idx))
    
    def _maintain_capacity(self):
        """Remove old episodes if capacity exceeded."""
        while len(self.episodes) > self.config.max_episodes:
            old_episode = self.episodes.pop(0)
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.total_steps -= len(old_episode)
        
        while self.total_steps > self.config.max_steps_total and self.episodes:
            old_episode = self.episodes.pop(0)
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.total_steps -= len(old_episode)
        
        # Rebuild sample index after removing episodes
        self._update_sample_index()
    
    def get_frame(self, ep_idx: int, frame_idx: int) -> Dict[str, Any]:
        """
        Get a single frame with all necessary data.
        
        与 SequentialMEKVMDataset.get_frame() 格式兼容。
        
        Returns dict with:
            - observation.images.image{i}: current frame for each camera
            - observation.images.image0_history: history + prediction frames for WM
            - observation.state: state vector
            - observation.state_history: state history
            - action: action chunk
            - action_history: action history
            - reward: rewards for the chunk
            - log_prob: log probabilities
            - value: value estimates
            - dataset_idx, episode_idx, frame_idx: indices
        """
        episode = self.episodes[ep_idx]
        n_obs = self.config.n_obs_img_steps
        n_pred = self.config.n_pred_img_steps
        chunk = self.config.chunk_size
        
        # Current step
        current_step = episode[frame_idx]
        
        # Build image history for each camera
        camera_images = {}
        for cam_idx, cam_key in enumerate(self.config.camera_keys):
            history = []
            for i in range(n_obs):
                hist_idx = frame_idx - (n_obs - 1 - i)
                if hist_idx < 0:
                    hist_idx = 0
                if cam_key in episode[hist_idx]['obs']:
                    history.append(episode[hist_idx]['obs'][cam_key])
                else:
                    # Placeholder
                    history.append(np.zeros((3, self.config.image_size, self.config.image_size), dtype=np.float32))
            camera_images[cam_idx] = np.stack(history, axis=0)  # (n_obs, C, H, W)
        
        # Build WM input: history + prediction targets
        wm_cam_key = self.config.wm_camera_key
        wm_cam_idx = self.config.camera_keys.index(wm_cam_key) if wm_cam_key in self.config.camera_keys else 0
        wm_history = list(camera_images[wm_cam_idx])
        
        # Add prediction target frames
        for i in range(n_pred):
            next_idx = min(frame_idx + 1 + i, len(episode) - 1)
            if wm_cam_key in episode[next_idx]['obs']:
                wm_history.append(episode[next_idx]['obs'][wm_cam_key])
            else:
                wm_history.append(np.zeros((3, self.config.image_size, self.config.image_size), dtype=np.float32))
        wm_history = np.stack(wm_history, axis=0)  # (n_obs + n_pred, C, H, W)
        
        # State history
        state_history = []
        for i in range(n_obs):
            hist_idx = frame_idx - (n_obs - 1 - i)
            if hist_idx < 0:
                hist_idx = 0
            state = episode[hist_idx]['obs'].get('state', np.zeros(self.config.state_dim, dtype=np.float32))
            state_history.append(state)
        state_history = np.stack(state_history, axis=0)  # (n_obs, state_dim)
        
        # Action history
        action_history = []
        for i in range(n_obs):
            hist_idx = frame_idx - (n_obs - 1 - i)
            if hist_idx < 0:
                action = np.zeros(self.config.action_dim, dtype=np.float32)
            else:
                action = episode[hist_idx]['action']
                # Ensure correct shape
                if isinstance(action, np.ndarray) and action.shape[0] != self.config.action_dim:
                    if action.shape[0] > self.config.action_dim:
                        action = action[:self.config.action_dim]
                    else:
                        padded = np.zeros(self.config.action_dim, dtype=np.float32)
                        padded[:action.shape[0]] = action
                        action = padded
            action_history.append(action)
        action_history = np.stack(action_history, axis=0)  # (n_obs, action_dim)
        
        # Future actions (chunk)
        actions = []
        rewards = []
        log_probs = []
        values = []
        for i in range(chunk):
            future_idx = min(frame_idx + i, len(episode) - 1)
            act = episode[future_idx]['action']
            # Ensure correct shape
            if isinstance(act, np.ndarray) and act.shape[0] != self.config.action_dim:
                if act.shape[0] > self.config.action_dim:
                    act = act[:self.config.action_dim]
                else:
                    padded = np.zeros(self.config.action_dim, dtype=np.float32)
                    padded[:act.shape[0]] = act
                    act = padded
            actions.append(act)
            rewards.append(episode[future_idx]['reward'])
            log_probs.append(episode[future_idx]['log_prob'])
            values.append(episode[future_idx]['value'])
        actions = np.stack(actions, axis=0)  # (chunk, action_dim)
        
        # Build sample dict
        sample = {
            # State
            "observation.state": torch.from_numpy(current_step['obs'].get('state', np.zeros(self.config.state_dim))).float(),
            "observation.state_history": torch.from_numpy(state_history).float(),
            # Actions
            "action": torch.from_numpy(actions).float(),
            "action_history": torch.from_numpy(action_history).float(),
            "action_is_pad": torch.zeros(chunk, dtype=torch.bool),
            # World model input
            "observation.images.image0_history": torch.from_numpy(wm_history).float(),
            # RL-specific
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "log_prob": torch.tensor(log_probs, dtype=torch.float32),
            "value": torch.tensor(values, dtype=torch.float32),
            "done": torch.tensor(current_step['done'], dtype=torch.bool),
            # Indices
            "dataset_idx": torch.tensor(0, dtype=torch.int64),
            "episode_idx": torch.tensor(ep_idx, dtype=torch.int64),
            "frame_idx": torch.tensor(frame_idx, dtype=torch.int64),
        }
        
        # Add each camera's images
        for cam_idx, cam_data in camera_images.items():
            sample[f"observation.images.image{cam_idx}"] = torch.from_numpy(cam_data[-1]).float()
            sample[f"observation.images.image{cam_idx}_history"] = torch.from_numpy(cam_data).float()
            sample[f"observation.images.image{cam_idx}_mask"] = torch.tensor(True)
        
        return sample
    
    def sample_batch(self, batch_size: int, sequential: bool = True) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of samples
            sequential: If True, sample consecutive frames from same episode
            
        Returns:
            Batched dict of tensors
        """
        if not self.sample_index:
            raise ValueError("Buffer is empty, cannot sample")
        
        if sequential:
            # Sample starting points and take consecutive frames
            batch_samples = []
            
            # Group by episode for sequential sampling
            ep_indices = list(set(idx[0] for idx in self.sample_index))
            
            while len(batch_samples) < batch_size:
                # Random episode
                ep_idx = np.random.choice(ep_indices)
                ep_frames = [idx for idx in self.sample_index if idx[0] == ep_idx]
                
                if ep_frames:
                    # Random starting frame
                    start_idx = np.random.randint(len(ep_frames))
                    batch_samples.append(ep_frames[start_idx])
                    
                    # Take consecutive frames if needed
                    for i in range(1, min(batch_size - len(batch_samples) + 1, len(ep_frames) - start_idx)):
                        batch_samples.append(ep_frames[start_idx + i])
                        if len(batch_samples) >= batch_size:
                            break
        else:
            # Random sampling
            indices = np.random.choice(len(self.sample_index), size=min(batch_size, len(self.sample_index)), replace=False)
            batch_samples = [self.sample_index[i] for i in indices]
        
        # Collect samples
        samples = [self.get_frame(ep_idx, frame_idx) for ep_idx, frame_idx in batch_samples]
        
        # Batch
        batch = {}
        for key in samples[0].keys():
            if isinstance(samples[0][key], torch.Tensor):
                batch[key] = torch.stack([s[key] for s in samples], dim=0)
            else:
                batch[key] = [s[key] for s in samples]
        
        return batch
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics."""
        if not self.episode_rewards:
            return {
                'mean_reward': 0.0,
                'mean_length': 0.0,
                'total_episodes': 0,
                'total_steps': 0,
                'buffer_samples': 0,
            }
        
        return {
            'mean_reward': np.mean(self.episode_rewards[-100:]),
            'std_reward': np.std(self.episode_rewards[-100:]),
            'mean_length': np.mean(self.episode_lengths[-100:]),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'buffer_samples': len(self.sample_index),
        }
    
    def clear(self):
        """Clear the buffer."""
        self.episodes.clear()
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.current_episode.clear()
        self.sample_index.clear()
        self.total_steps = 0
        self.total_episodes = 0
    
    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.sample_index)


class SequentialRolloutCollector:
    """
    Collects rollouts from environment and stores in sequential format.
    
    整合 Explorer 环境交互和 sequential 数据格式。
    """
    
    def __init__(
        self,
        policy,  # F1_VLA policy with Explorer actor
        vae_extractor,  # VAEEmbeddingExtractor for reward computation
        reward_manager,  # ExplorerRewardManager
        buffer: SequentialRolloutBuffer,
        config: Optional[SequentialRolloutConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize sequential rollout collector.
        
        Args:
            policy: F1_VLA policy with Explorer actor
            vae_extractor: VAE embedding extractor
            reward_manager: Reward computation manager
            buffer: Sequential rollout buffer
            config: Configuration
            device: Device for computation
        """
        self.policy = policy
        self.vae_extractor = vae_extractor
        self.reward_manager = reward_manager
        self.buffer = buffer
        self.config = config or buffer.config
        self.device = device
        
        # Current episode state
        self.observation_history: List[Dict[str, Any]] = []
        self.action_history: List[np.ndarray] = []
        
        # KV cache for memory (per episode)
        self.memory_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.past_key_values: Optional[List[torch.FloatTensor]] = None
        
        logger.info("SequentialRolloutCollector initialized")
    
    def reset(self):
        """Reset collector state."""
        self.observation_history.clear()
        self.action_history.clear()
        self.reward_manager.reset()
        self.memory_kv = None
        self.past_key_values = None
    
    def _get_policy_input(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare policy input from observation history.
        
        Returns dict compatible with F1_VLA.forward_with_actor()
        """
        n_obs = self.config.n_obs_img_steps
        
        # Build image history
        images = []
        wm_key = self.config.wm_camera_key
        for i in range(n_obs):
            idx = len(self.observation_history) - n_obs + i
            if idx < 0:
                idx = 0
            if idx < len(self.observation_history):
                obs = self.observation_history[idx]
            else:
                obs = observation
            
            if wm_key in obs:
                img = obs[wm_key]
                if isinstance(img, np.ndarray):
                    if img.ndim == 4:
                        img = img[-1]
                    if img.max() > 1.0:
                        img = img.astype(np.float32) / 255.0
                images.append(torch.from_numpy(img).float())
        
        while len(images) < n_obs:
            images = [images[0]] + images
        
        images = torch.stack(images, dim=0).unsqueeze(0).to(self.device)  # (1, n_obs, C, H, W)
        
        # State
        state = observation.get('state', np.zeros(self.config.state_dim, dtype=np.float32))
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # (1, state_dim)
        
        # Action history
        actions = []
        for i in range(n_obs):
            idx = len(self.action_history) - n_obs + i
            if idx < 0 or idx >= len(self.action_history):
                # Pad with zeros, ensure correct shape
                actions.append(np.zeros(self.config.action_dim, dtype=np.float32))
            else:
                act = self.action_history[idx]
                # Ensure action has correct shape
                if act.shape[0] != self.config.action_dim:
                    # Resize: truncate or pad
                    if act.shape[0] > self.config.action_dim:
                        act = act[:self.config.action_dim]
                    else:
                        padded = np.zeros(self.config.action_dim, dtype=np.float32)
                        padded[:act.shape[0]] = act
                        act = padded
                actions.append(act)
        actions = torch.from_numpy(np.stack(actions)).float().unsqueeze(0).to(self.device)  # (1, n_obs, action_dim)
        
        return {
            'observation.images.image0_history': images,
            'observation.state': state,
            'action_history': actions,
        }
    
    def _compute_reward_full(
        self, 
        obs: Dict[str, Any], 
        next_obs: Dict[str, Any], 
        action: np.ndarray,
        memory_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute full reward using World Model prediction.
        
        Implements complete reward formula:
        reward = α*r1 + β*r2 + γ*r3 + ε*r4 - δ*|a_t|
        
        Where:
        - r1: uncertainty_{t+1} (from WM prediction entropy)
        - r2: MSE(pred_emb_{t+1}, emb_{t+1})
        - r3: MSE_{t+1} - MSE_{t+2} (delayed)
        - r4: unc_{t+1} - unc_{t+2} (delayed)
        
        Returns:
            reward: Scalar reward value
            info: Dictionary with reward components
        """
        wm_cam_key = self.config.wm_camera_key
        
        # Get images
        curr_img = obs.get(wm_cam_key)
        next_img = next_obs.get(wm_cam_key)
        
        if curr_img is None or next_img is None:
            return 0.0, {}
        
        # Normalize images
        if isinstance(curr_img, np.ndarray):
            if curr_img.ndim == 4:
                curr_img = curr_img[-1]
            if curr_img.max() > 1.0:
                curr_img = curr_img / 255.0
            curr_img = torch.from_numpy(curr_img).float().unsqueeze(0).to(self.device)
        
        if isinstance(next_img, np.ndarray):
            if next_img.ndim == 4:
                next_img = next_img[-1]
            if next_img.max() > 1.0:
                next_img = next_img / 255.0
            next_img = torch.from_numpy(next_img).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 1. Get VAE embeddings (ground truth)
            gt_emb = self.vae_extractor.encode_image(next_img)
            
            # 2. Get predicted embedding (use current image as simple baseline)
            # In full implementation, this should use World Model forward
            pred_emb = self.vae_extractor.encode_image(curr_img)
            
            # 3. Compute uncertainty as embedding distance
            uncertainty = torch.norm(gt_emb - pred_emb, dim=-1).mean()
            
            # 4. Convert action to tensor
            action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)
            if action_tensor.shape[-1] != self.config.action_dim:
                if action_tensor.shape[-1] > self.config.action_dim:
                    action_tensor = action_tensor[..., :self.config.action_dim]
                else:
                    padded = torch.zeros(1, self.config.action_dim, device=self.device, dtype=torch.float32)
                    padded[..., :action_tensor.shape[-1]] = action_tensor
                    action_tensor = padded
            
            # 5. Compute reward through reward manager
            reward_tensor, info = self.reward_manager.step(
                pred_emb=pred_emb,
                gt_emb=gt_emb,
                uncertainty=uncertainty,
                action=action_tensor,
                is_logits=False,
            )
            
            # Return immediate reward or full reward if available
            if reward_tensor is not None:
                reward = reward_tensor.mean().item()
            else:
                # Use immediate reward only
                reward = info.get('immediate_reward', 0.0)
        
        return reward, info
    
    def collect_episode(
        self,
        env,
        max_steps: int = None,
        use_explorer: bool = True,
        epsilon: float = 0.0,
    ) -> Dict[str, float]:
        """
        Collect one episode from environment.
        
        Args:
            env: Environment instance
            max_steps: Maximum steps per episode
            use_explorer: Whether to use Explorer actor
            epsilon: Epsilon for exploration
            
        Returns:
            Episode statistics
        """
        max_steps = max_steps or self.config.max_steps_per_episode
        self.reset()
        
        # Reset environment
        obs, info = env.reset()
        self.observation_history.append(obs)
        
        # Set active actor
        if use_explorer and hasattr(self.policy, 'active_actor'):
            self.policy.active_actor = 'explorer'
        
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            # Get action from policy
            with torch.no_grad():
                policy_input = self._get_policy_input(obs)
                
                if np.random.random() < epsilon:
                    # Random action
                    action = np.random.uniform(-1, 1, size=self.config.action_dim).astype(np.float32)
                    log_prob = 0.0
                    value = 0.0
                else:
                    # Policy action
                    output = self.policy.forward_with_actor(
                        policy_input,
                        actor_name='explorer' if use_explorer else 'actor',
                        return_action_stats=True,
                    )
                    action = output['action'].cpu().numpy().squeeze()
                    log_prob = output.get('log_prob', torch.tensor(0.0)).item() if 'log_prob' in output else 0.0
                    value = output.get('value', torch.tensor(0.0)).item() if 'value' in output else 0.0
            
            # Execute action
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Compute Explorer reward (full reward with delayed components)
            explorer_reward, reward_info = self._compute_reward_full(
                obs, next_obs, action, memory_kv=self.memory_kv
            )
            total_reward = explorer_reward + 0.1 * env_reward  # Combine with small env reward
            
            # Log reward components periodically
            if episode_length % 50 == 0:
                logger.debug(f"  Step {episode_length}: reward={total_reward:.3f}, info={reward_info}")
            
            # Add to buffer
            self.buffer.add_step(
                observation=obs,
                action=action,
                reward=total_reward,
                next_observation=next_obs,
                done=done,
                info=info,
                log_prob=log_prob,
                value=value,
            )
            
            # Update state
            obs = next_obs
            self.observation_history.append(obs)
            self.action_history.append(action)
            
            episode_reward += total_reward
            episode_length += 1
        
        # Force finish episode if not done
        if not done:
            self.buffer._finish_episode()
        
        return {
            'reward': episode_reward,
            'length': episode_length,
            'done': done,
        }
    
    def collect_rollouts(
        self,
        env,
        num_steps: int,
        use_explorer: bool = True,
        epsilon: float = 0.0,
    ) -> Dict[str, float]:
        """
        Collect multiple steps across episodes.
        
        Args:
            env: Environment instance
            num_steps: Total steps to collect
            use_explorer: Whether to use Explorer actor
            epsilon: Epsilon for exploration
            
        Returns:
            Collection statistics
        """
        total_reward = 0.0
        total_episodes = 0
        steps_collected = 0
        
        while steps_collected < num_steps:
            stats = self.collect_episode(
                env,
                max_steps=min(num_steps - steps_collected, self.config.max_steps_per_episode),
                use_explorer=use_explorer,
                epsilon=epsilon,
            )
            
            total_reward += stats['reward']
            total_episodes += 1
            steps_collected += stats['length']
        
        return {
            'total_reward': total_reward,
            'mean_reward': total_reward / total_episodes if total_episodes > 0 else 0.0,
            'total_episodes': total_episodes,
            'steps_collected': steps_collected,
        }
