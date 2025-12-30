"""
Explorer Environment Rollout Module

This module implements the environment interaction loop for Explorer RL training.
The Explorer collects trajectories by interacting with environments and
computes rewards based on WM uncertainty and prediction error.

Key Features:
- Rollout collection with VAE embedding extraction
- Reward computation using WM predictions
- Support for both single and parallel environments
- Episode statistics tracking
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RolloutConfig:
    """Configuration for environment rollout."""
    # Episode settings
    max_steps_per_episode: int = 100
    num_episodes_per_collect: int = 1
    
    # History length
    history_length: int = 4  # L frames of observation
    
    # Action settings
    action_dim: int = 7
    action_scale: float = 1.0
    
    # Image settings
    image_size: Tuple[int, int] = (256, 256)
    
    # Reward computation
    compute_immediate_only: bool = False  # If True, skip delayed rewards
    
    # Random exploration settings
    epsilon_greedy: float = 0.0  # Probability of random action
    initial_random_steps: int = 0  # Random steps at start of episode


@dataclass
class Transition:
    """Single transition in an episode."""
    # Observation data
    observation: Dict[str, Any]
    action: np.ndarray
    
    # Next step data
    next_observation: Dict[str, Any]
    done: bool
    
    # Reward information
    immediate_reward: float = 0.0
    delayed_reward: Optional[float] = None
    full_reward: Optional[float] = None
    
    # Policy outputs
    log_prob: float = 0.0
    value: float = 0.0
    
    # Embedding data (for reward computation)
    gt_embedding: Optional[torch.Tensor] = None
    pred_embedding: Optional[torch.Tensor] = None
    uncertainty: Optional[torch.Tensor] = None
    mse: Optional[torch.Tensor] = None
    
    # Additional info
    info: Dict[str, Any] = field(default_factory=dict)
    
    # For GAE computation
    advantage: float = 0.0
    returns: float = 0.0


class EpisodeBuffer:
    """Buffer for storing episode transitions."""
    
    def __init__(self, max_episodes: int = 100):
        """
        Initialize episode buffer.
        
        Args:
            max_episodes: Maximum episodes to store
        """
        self.max_episodes = max_episodes
        self.episodes: List[List[Transition]] = []
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
    def add_episode(self, transitions: List[Transition]):
        """Add an episode to the buffer."""
        self.episodes.append(transitions)
        self.total_steps += len(transitions)
        self.total_episodes += 1
        
        # Maintain max size
        while len(self.episodes) > self.max_episodes:
            old_episode = self.episodes.pop(0)
            self.total_steps -= len(old_episode)
    
    def get_all_transitions(self) -> List[Transition]:
        """Get all transitions from all episodes."""
        return [t for episode in self.episodes for t in episode]
    
    def get_latest_episode(self) -> List[Transition]:
        """Get the most recent episode."""
        if not self.episodes:
            return []
        return self.episodes[-1]
    
    def clear(self):
        """Clear the buffer."""
        self.episodes.clear()
        self.total_steps = 0
        self.total_episodes = 0
    
    def __len__(self) -> int:
        return len(self.episodes)


class ObservationHistory:
    """
    Manages observation history for policy input.
    
    The Explorer needs L+1 frames of history:
    - L frames for the base model (PaliGemma + WM)
    - 1 additional frame (gt_{t+1}) for embedding comparison
    """
    
    def __init__(
        self,
        history_length: int = 4,
        image_keys: List[str] = None,
    ):
        """
        Initialize observation history.
        
        Args:
            history_length: Number of history frames (L)
            image_keys: Keys for image observations
        """
        self.history_length = history_length
        self.image_keys = image_keys or ["head_rgb", "wrist_rgb"]
        
        # Store history
        self.images: Dict[str, List[np.ndarray]] = {k: [] for k in self.image_keys}
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
    
    def reset(self):
        """Reset history."""
        for key in self.image_keys:
            self.images[key] = []
        self.states = []
        self.actions = []
    
    def add(
        self,
        observation: Dict[str, np.ndarray],
        action: Optional[np.ndarray] = None,
    ):
        """
        Add observation to history.
        
        Args:
            observation: Observation dict with images and state
            action: Action taken (None for initial observation)
        """
        # Add images
        for key in self.image_keys:
            if key in observation:
                img = observation[key]
                # Handle stacked images (T, C, H, W)
                if img.ndim == 4:
                    img = img[-1]  # Take last frame
                self.images[key].append(img)
        
        # Add state
        if 'state' in observation:
            self.states.append(observation['state'])
        
        # Add action
        if action is not None:
            self.actions.append(action)
        
        # Maintain max length (L+1 for images/states, L for actions)
        max_len = self.history_length + 1
        for key in self.image_keys:
            while len(self.images[key]) > max_len:
                self.images[key].pop(0)
        while len(self.states) > max_len:
            self.states.pop(0)
        while len(self.actions) > self.history_length:
            self.actions.pop(0)
    
    def get_model_input(self) -> Dict[str, np.ndarray]:
        """
        Get input for the base model (PaliGemma + WM).
        
        Returns dict with L frames of history.
        """
        result = {}
        
        # Get last L images for each key
        for key in self.image_keys:
            if self.images[key]:
                imgs = self.images[key][-self.history_length:]
                # Pad if needed
                while len(imgs) < self.history_length:
                    imgs = [imgs[0]] + imgs
                result[key] = np.stack(imgs, axis=0)  # (L, C, H, W)
        
        # Get last L states
        if self.states:
            states = self.states[-self.history_length:]
            while len(states) < self.history_length:
                states = [states[0]] + states
            result['state'] = np.stack(states, axis=0)  # (L, state_dim)
        
        # Get last L actions
        if self.actions:
            actions = self.actions[-self.history_length:]
            while len(actions) < self.history_length:
                actions = [np.zeros_like(actions[0])] + actions
            result['action_history'] = np.stack(actions, axis=0)  # (L, action_dim)
        
        return result
    
    def get_latest_frame(self, key: str = 'wrist_rgb') -> Optional[np.ndarray]:
        """Get the most recent frame for a given key."""
        if key in self.images and self.images[key]:
            return self.images[key][-1]
        return None
    
    def has_enough_history(self) -> bool:
        """Check if we have enough history for model input."""
        # Need at least 1 frame
        for key in self.image_keys:
            if not self.images[key]:
                return False
        return True


class ExplorerRolloutCollector:
    """
    Collects rollouts for Explorer RL training.
    
    This handles:
    1. Environment interaction
    2. VAE embedding extraction
    3. Reward computation
    4. Episode statistics
    """
    
    def __init__(
        self,
        policy,  # F1_VLA policy
        vae_extractor,  # VAEEmbeddingExtractor
        reward_manager,  # ExplorerRewardManager
        config: Optional[RolloutConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize rollout collector.
        
        Args:
            policy: F1_VLA policy with Explorer actor
            vae_extractor: VAE embedding extractor
            reward_manager: Reward computation manager
            config: Rollout configuration
            device: Device for computation
        """
        self.policy = policy
        self.vae_extractor = vae_extractor
        self.reward_manager = reward_manager
        self.config = config or RolloutConfig()
        self.device = device
        
        # Episode buffer
        self.episode_buffer = EpisodeBuffer()
        
        # Statistics
        self.episode_rewards: deque = deque(maxlen=100)
        self.episode_lengths: deque = deque(maxlen=100)
        
        # Current episode state
        self.current_history: Optional[ObservationHistory] = None
        self.current_transitions: List[Transition] = []
        self.step_count = 0
        
    def reset_episode(self):
        """Reset for new episode."""
        self.current_history = ObservationHistory(
            history_length=self.config.history_length,
        )
        self.current_transitions = []
        self.step_count = 0
        self.reward_manager.reset()
    
    def collect_episode(
        self,
        env,
        use_explorer: bool = True,
        record_embeddings: bool = True,
    ) -> List[Transition]:
        """
        Collect one episode.
        
        Args:
            env: Environment instance
            use_explorer: If True, use Explorer actor; else use default actor
            record_embeddings: If True, compute and store embeddings
            
        Returns:
            List of transitions for the episode
        """
        self.reset_episode()
        
        # Reset environment
        obs, info = env.reset()
        self.current_history.add(obs)
        
        # Set active actor
        if use_explorer:
            self.policy.active_actor = 'explorer'
        
        done = False
        episode_reward = 0.0
        
        while not done and self.step_count < self.config.max_steps_per_episode:
            # Get action
            if self.step_count < self.config.initial_random_steps:
                # Random action at start
                action = np.random.uniform(
                    -1, 1,
                    size=self.config.action_dim
                ).astype(np.float32) * self.config.action_scale
                log_prob = 0.0
                value = 0.0
            elif np.random.random() < self.config.epsilon_greedy:
                # Epsilon-greedy random action
                action = np.random.uniform(
                    -1, 1,
                    size=self.config.action_dim
                ).astype(np.float32) * self.config.action_scale
                log_prob = 0.0
                value = 0.0
            else:
                # Policy action
                action, log_prob, value = self._get_action(obs)
            
            # Execute action
            next_obs, env_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Add to history
            self.current_history.add(next_obs, action)
            
            # Compute embeddings and rewards
            transition = self._create_transition(
                obs, action, next_obs, done,
                log_prob, value, info,
                record_embeddings=record_embeddings,
            )
            
            episode_reward += transition.full_reward or transition.immediate_reward
            self.current_transitions.append(transition)
            
            obs = next_obs
            self.step_count += 1
        
        # Finalize delayed rewards
        self._finalize_episode_rewards()
        
        # Compute GAE advantages
        self._compute_advantages()
        
        # Store statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(self.step_count)
        
        # Add to buffer
        self.episode_buffer.add_episode(self.current_transitions)
        
        return self.current_transitions
    
    def _get_action(
        self,
        obs: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get action from policy.
        
        Args:
            obs: Current observation
            
        Returns:
            action: Action array
            log_prob: Log probability of action
            value: Value estimate
        """
        # Build batch from history
        batch = self._build_batch()
        
        # Forward pass
        with torch.no_grad():
            output = self.policy.forward_with_actor(
                batch,
                actor_name='explorer',
                return_action_stats=True,
            )
        
        # Extract action
        action = output['action'][0].cpu().numpy()
        
        # Get log prob and value if available
        log_prob = output.get('log_prob', torch.tensor(0.0))[0].item()
        value = output.get('value', torch.tensor(0.0))[0].item()
        
        # Scale action
        action = action * self.config.action_scale
        
        return action, log_prob, value
    
    def _build_batch(self) -> Dict[str, torch.Tensor]:
        """Build batch from observation history."""
        model_input = self.current_history.get_model_input()
        
        batch = {}
        
        # Images
        for key in ['head_rgb', 'wrist_rgb']:
            if key in model_input:
                imgs = model_input[key]  # (L, C, H, W)
                # Normalize if needed
                if imgs.max() > 1.0:
                    imgs = imgs.astype(np.float32) / 255.0
                batch[f'observation.{key}'] = torch.from_numpy(imgs).unsqueeze(0).to(self.device)
        
        # State
        if 'state' in model_input:
            batch['observation.state'] = torch.from_numpy(
                model_input['state']
            ).unsqueeze(0).float().to(self.device)
        
        # Action history
        if 'action_history' in model_input:
            batch['action_history'] = torch.from_numpy(
                model_input['action_history']
            ).unsqueeze(0).float().to(self.device)
        
        return batch
    
    def _create_transition(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs: Dict[str, np.ndarray],
        done: bool,
        log_prob: float,
        value: float,
        info: Dict[str, Any],
        record_embeddings: bool = True,
    ) -> Transition:
        """Create transition with reward computation."""
        transition = Transition(
            observation=obs,
            action=action,
            next_observation=next_obs,
            done=done,
            log_prob=log_prob,
            value=value,
            info=info,
        )
        
        if not record_embeddings:
            return transition
        
        # Extract embeddings
        try:
            # Get GT embedding from next_obs
            next_img = next_obs.get('wrist_rgb')
            if next_img is not None:
                if next_img.ndim == 4:
                    next_img = next_img[-1]
                gt_emb = self.vae_extractor.encode_image(
                    torch.from_numpy(next_img).unsqueeze(0).to(self.device)
                )
                transition.gt_embedding = gt_emb
            
            # Get prediction embedding and uncertainty from WM
            # This requires running WM forward pass
            batch = self._build_batch()
            batch['action'] = torch.from_numpy(action).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                wm_output = self.policy.predict_next_frame(batch)
                
                if 'pred_embedding' in wm_output:
                    transition.pred_embedding = wm_output['pred_embedding']
                if 'uncertainty' in wm_output:
                    transition.uncertainty = wm_output['uncertainty']
                elif 'logits' in wm_output:
                    # Compute uncertainty from logits
                    from models.vae_embedding import UncertaintyEstimator
                    estimator = UncertaintyEstimator()
                    transition.uncertainty = estimator.compute_entropy(wm_output['logits'])
            
            # Compute reward
            if transition.gt_embedding is not None and transition.pred_embedding is not None:
                reward, reward_info = self.reward_manager.step(
                    pred_emb=transition.pred_embedding,
                    gt_emb=transition.gt_embedding,
                    uncertainty=transition.uncertainty if transition.uncertainty is not None else torch.zeros(1),
                    action=torch.from_numpy(action).unsqueeze(0).to(self.device),
                )
                
                transition.immediate_reward = reward_info.get('immediate_reward', 0.0)
                if reward is not None:
                    transition.full_reward = reward.mean().item()
                    transition.delayed_reward = reward_info.get('delayed_reward', 0.0)
                
                # Store MSE
                transition.mse = torch.tensor(reward_info.get('r2_mse', 0.0))
                
                # Update info
                transition.info.update(reward_info)
                
        except Exception as e:
            logger.warning(f"Error computing embeddings/rewards: {e}")
        
        return transition
    
    def _finalize_episode_rewards(self):
        """Finalize rewards at end of episode."""
        # Get final reward from manager
        final_reward = self.reward_manager.finalize_episode()
        
        if final_reward is not None and self.current_transitions:
            # Last transition only gets immediate reward
            last_trans = self.current_transitions[-1]
            last_trans.full_reward = last_trans.immediate_reward
    
    def _compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages for episode.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda
        """
        if not self.current_transitions:
            return
        
        # Get rewards and values
        rewards = [t.full_reward or t.immediate_reward for t in self.current_transitions]
        values = [t.value for t in self.current_transitions]
        
        # Add bootstrap value
        if self.current_transitions[-1].done:
            next_value = 0.0
        else:
            next_value = values[-1]
        
        # Compute returns and advantages
        advantages = []
        returns = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Update transitions
        for i, trans in enumerate(self.current_transitions):
            trans.advantage = advantages[i]
            trans.returns = returns[i]
    
    def get_statistics(self) -> Dict[str, float]:
        """Get rollout statistics."""
        stats = {}
        
        if self.episode_rewards:
            stats['mean_episode_reward'] = np.mean(self.episode_rewards)
            stats['std_episode_reward'] = np.std(self.episode_rewards)
        
        if self.episode_lengths:
            stats['mean_episode_length'] = np.mean(self.episode_lengths)
        
        stats['total_episodes'] = self.episode_buffer.total_episodes
        stats['total_steps'] = self.episode_buffer.total_steps
        
        # Reward manager stats
        stats.update(self.reward_manager.get_stats())
        
        return stats


def collect_trajectories(
    policy,
    env,
    vae_extractor,
    reward_manager,
    num_episodes: int = 10,
    config: Optional[RolloutConfig] = None,
    device: str = "cuda",
    progress_callback: Optional[Callable] = None,
) -> Tuple[List[List[Transition]], Dict[str, float]]:
    """
    Convenience function to collect multiple episodes.
    
    Args:
        policy: F1_VLA policy
        env: Environment instance
        vae_extractor: VAE embedding extractor
        reward_manager: Reward manager
        num_episodes: Number of episodes to collect
        config: Rollout configuration
        device: Device
        progress_callback: Optional callback(episode_idx, stats)
        
    Returns:
        all_episodes: List of episodes (each is list of transitions)
        stats: Collection statistics
    """
    collector = ExplorerRolloutCollector(
        policy=policy,
        vae_extractor=vae_extractor,
        reward_manager=reward_manager,
        config=config,
        device=device,
    )
    
    all_episodes = []
    
    for ep_idx in range(num_episodes):
        transitions = collector.collect_episode(env)
        all_episodes.append(transitions)
        
        if progress_callback:
            progress_callback(ep_idx, collector.get_statistics())
    
    return all_episodes, collector.get_statistics()


def transitions_to_batch(
    transitions: List[Transition],
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Convert transitions to batch tensors for training.
    
    Args:
        transitions: List of transitions
        device: Device
        
    Returns:
        Batch dictionary
    """
    batch = {
        'actions': torch.tensor(
            np.stack([t.action for t in transitions]),
            dtype=torch.float32,
            device=device,
        ),
        'log_probs': torch.tensor(
            [t.log_prob for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
        'values': torch.tensor(
            [t.value for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
        'advantages': torch.tensor(
            [t.advantage for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
        'returns': torch.tensor(
            [t.returns for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
        'rewards': torch.tensor(
            [t.full_reward or t.immediate_reward for t in transitions],
            dtype=torch.float32,
            device=device,
        ),
    }
    
    # Add embeddings if available
    gt_embs = [t.gt_embedding for t in transitions if t.gt_embedding is not None]
    if gt_embs:
        batch['gt_embeddings'] = torch.stack(gt_embs, dim=0).to(device)
    
    pred_embs = [t.pred_embedding for t in transitions if t.pred_embedding is not None]
    if pred_embs:
        batch['pred_embeddings'] = torch.stack(pred_embs, dim=0).to(device)
    
    return batch
