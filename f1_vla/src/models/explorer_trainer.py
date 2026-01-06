"""
Explorer Phase 1 RL Training Module

This module implements Phase 1 of Explorer training:
- Freeze World Model and policy actor
- Train only Explorer actor with PPO
- Reward based on WM uncertainty and prediction error

Key Features:
- PPO algorithm with GAE
- Value head for baseline
- Gradient isolation (only Explorer trainable)
- Compatible with F1_VLA multi-actor architecture
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class ExplorerTrainingConfig:
    """Configuration for Explorer training."""
    # Episode settings
    num_episodes: int = 1000
    max_steps_per_episode: int = 100
    
    # PPO hyperparameters
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # GAE parameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_schedule_type: str = "cosine"  # "cosine" or "linear"
    warmup_episodes: int = 50
    
    # Reward normalization
    normalize_rewards: bool = True
    normalize_advantages: bool = True
    clip_rewards: Optional[float] = 10.0
    
    # Action settings
    action_dim: int = 7
    action_scale: float = 1.0
    init_log_std: float = -1.0  # Initial action std = exp(-1) ≈ 0.37
    
    # Logging and saving
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "./outputs/explorer_rl"
    
    # Phase 1 specific
    freeze_world_model: bool = True
    freeze_policy_actor: bool = True
    
    # Mode collapse detection
    detect_mode_collapse: bool = True
    min_action_std: float = 0.1          # Alert if action std falls below this
    min_entropy_threshold: float = 0.5   # Alert if entropy falls below this
    collapse_check_window: int = 50      # Window size for collapse detection
    entropy_bonus_on_collapse: float = 0.1  # Extra entropy bonus when collapse detected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'num_episodes': self.num_episodes,
            'max_steps_per_episode': self.max_steps_per_episode,
            'ppo_epochs': self.ppo_epochs,
            'mini_batch_size': self.mini_batch_size,
            'clip_epsilon': self.clip_epsilon,
            'value_loss_coef': self.value_loss_coef,
            'entropy_coef': self.entropy_coef,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'normalize_rewards': self.normalize_rewards,
            'normalize_advantages': self.normalize_advantages,
            'action_dim': self.action_dim,
            'action_scale': self.action_scale,
            'freeze_world_model': self.freeze_world_model,
            'freeze_policy_actor': self.freeze_policy_actor,
        }


class PPOValueHead(nn.Module):
    """Value head for PPO baseline."""
    
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate.
        
        Args:
            state_emb: State embedding (B, D)
            
        Returns:
            Value estimates (B,)
        """
        return self.net(state_emb).squeeze(-1)


class ExplorerRLTrainer:
    """
    PPO trainer for Explorer actor.
    
    Phase 1: Freeze WM and policy actor, train only Explorer.
    """
    
    def __init__(
        self,
        policy,  # F1_VLA policy
        vae_extractor,  # VAEEmbeddingExtractor
        reward_manager,  # ExplorerRewardManager
        config: Optional[ExplorerTrainingConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize trainer.
        
        Args:
            policy: F1_VLA policy with Explorer actor
            vae_extractor: VAE embedding extractor
            reward_manager: Reward computation manager
            config: Training configuration
            device: Device
        """
        self.policy = policy
        self.vae_extractor = vae_extractor
        self.reward_manager = reward_manager
        self.config = config or ExplorerTrainingConfig()
        self.device = device
        
        # Ensure explorer exists
        if 'explorer' not in policy.list_actors():
            raise ValueError("Explorer actor not found in policy. Call initialize_explorer() first.")
        
        # Setup training
        self._setup_training()
        
        # KV memory/cache for Explorer
        # This maintains memory across steps within an episode
        self.episode_memory_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self.episode_past_kv: Optional[List[torch.FloatTensor]] = None
        
        # Metrics tracking
        self.metrics = {
            'episode_rewards': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100),
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'total_loss': deque(maxlen=100),
            'r1_uncertainty': deque(maxlen=100),
            'r2_mse': deque(maxlen=100),
            'r3_mse_improvement': deque(maxlen=100),
            'r4_uncertainty_improvement': deque(maxlen=100),
            # Mode collapse detection metrics
            'action_std': deque(maxlen=100),
            'action_mean_norm': deque(maxlen=100),
            'state_coverage': deque(maxlen=100),
        }
        
        # Mode collapse state
        self._collapse_detected = False
        self._collapse_count = 0
        self._visited_states = set()  # Hash of visited state embeddings
        
        # Training state
        self.global_step = 0
        self.current_episode = 0
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_training(self):
        """Setup training components."""
        # Set explorer as active
        self.policy.active_actor = 'explorer'
        
        # Freeze non-explorer components
        if self.config.freeze_world_model:
            self._freeze_world_model()
        if self.config.freeze_policy_actor:
            self._freeze_policy_actor()
        
        # Make only explorer trainable
        self.policy.set_trainable_actors(['explorer'])
        
        # Get explorer parameters
        explorer = self.policy.get_actor('explorer')
        explorer_params = list(explorer.parameters())
        
        # Value head
        # Get input dimension from policy config
        proj_width = getattr(self.policy.model.config, 'proj_width', 1024)
        self.value_head = PPOValueHead(input_dim=proj_width).to(self.device)
        
        # Action log std (learnable)
        self.log_std = nn.Parameter(
            torch.ones(self.config.action_dim, device=self.device) * self.config.init_log_std
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [
                {'params': explorer_params, 'lr': self.config.learning_rate},
                {'params': self.value_head.parameters(), 'lr': self.config.learning_rate},
                {'params': [self.log_std], 'lr': self.config.learning_rate * 0.1},
            ],
            weight_decay=self.config.weight_decay,
        )
        
        # Learning rate scheduler
        if self.config.use_lr_schedule:
            if self.config.lr_schedule_type == "cosine":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.num_episodes,
                    eta_min=1e-6,
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=1.0,
                    end_factor=0.01,
                    total_iters=self.config.num_episodes,
                )
        else:
            self.scheduler = None
        
        # Log setup
        trainable_params = sum(p.numel() for p in explorer_params if p.requires_grad)
        value_params = sum(p.numel() for p in self.value_head.parameters())
        logger.info(f"Explorer RL trainer setup:")
        logger.info(f"  Explorer trainable params: {trainable_params:,}")
        logger.info(f"  Value head params: {value_params:,}")
        logger.info(f"  Learning rate: {self.config.learning_rate}")
    
    def _freeze_world_model(self):
        """Freeze World Model parameters."""
        # Check if this is a F1_VLA policy (has .model attribute) or direct model
        model = getattr(self.policy, 'model', self.policy)
        
        if not hasattr(model, 'paligemma_with_expert'):
            logger.info("Model has no 'paligemma_with_expert' attribute, skipping WM freeze")
            return
            
        pwm = model.paligemma_with_expert
        
        # Freeze WM expert
        if hasattr(pwm, 'gemma_wm_expert'):
            for param in pwm.gemma_wm_expert.parameters():
                param.requires_grad = False
        
        # Freeze VAE
        if hasattr(model, 'vae') and model.vae is not None:
            for param in model.vae.parameters():
                param.requires_grad = False
        
        # Freeze vision tower
        if hasattr(pwm, 'paligemma'):
            for param in pwm.paligemma.vision_tower.parameters():
                param.requires_grad = False
            
            # Freeze language model embeddings
            for param in pwm.paligemma.language_model.parameters():
                param.requires_grad = False
        
        logger.info("Froze World Model and shared components")
    
    def _freeze_policy_actor(self):
        """Freeze the main policy actor."""
        if 'actor' in self.policy.list_actors():
            actor = self.policy.get_actor('actor')
            for param in actor.parameters():
                param.requires_grad = False
            logger.info("Froze policy actor")
    
    def detect_mode_collapse(
        self,
        actions: torch.Tensor,
        entropy: torch.Tensor,
        state_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Detect mode collapse based on action diversity and entropy.
        
        Mode collapse indicators:
        1. Low action standard deviation (actions becoming uniform)
        2. Low policy entropy (deterministic policy)
        3. Low state coverage (visiting same states repeatedly)
        
        Args:
            actions: Batch of actions (B, action_dim)
            entropy: Policy entropy (scalar or per-sample)
            state_emb: Optional state embeddings for coverage tracking
            
        Returns:
            Dict with collapse detection results and metrics
        """
        if not self.config.detect_mode_collapse:
            return {'collapsed': False}
        
        result = {
            'collapsed': False,
            'reasons': [],
            'metrics': {},
        }
        
        # 1. Check action standard deviation
        action_std = actions.std(dim=0).mean().item()
        action_mean_norm = actions.mean(dim=0).norm().item()
        result['metrics']['action_std'] = action_std
        result['metrics']['action_mean_norm'] = action_mean_norm
        
        self.metrics['action_std'].append(action_std)
        self.metrics['action_mean_norm'].append(action_mean_norm)
        
        if action_std < self.config.min_action_std:
            result['collapsed'] = True
            result['reasons'].append(f'Low action std: {action_std:.4f} < {self.config.min_action_std}')
        
        # 2. Check entropy
        entropy_val = entropy.mean().item() if entropy.dim() > 0 else entropy.item()
        result['metrics']['entropy'] = entropy_val
        
        if entropy_val < self.config.min_entropy_threshold:
            result['collapsed'] = True
            result['reasons'].append(f'Low entropy: {entropy_val:.4f} < {self.config.min_entropy_threshold}')
        
        # 3. Check state coverage (if state embeddings provided)
        if state_emb is not None:
            # Discretize state embeddings for hashing
            state_hash = self._hash_states(state_emb)
            new_states = 0
            for h in state_hash:
                if h not in self._visited_states:
                    self._visited_states.add(h)
                    new_states += 1
            
            coverage_rate = new_states / len(state_hash) if len(state_hash) > 0 else 0
            result['metrics']['new_state_rate'] = coverage_rate
            result['metrics']['total_visited_states'] = len(self._visited_states)
            self.metrics['state_coverage'].append(coverage_rate)
            
            # Low coverage over recent window suggests collapse
            if len(self.metrics['state_coverage']) >= self.config.collapse_check_window:
                recent_coverage = np.mean(list(self.metrics['state_coverage'])[-self.config.collapse_check_window:])
                if recent_coverage < 0.05:  # Less than 5% new states
                    result['collapsed'] = True
                    result['reasons'].append(f'Low state coverage: {recent_coverage:.2%}')
        
        # Update collapse state
        if result['collapsed']:
            self._collapse_count += 1
            if not self._collapse_detected:
                self._collapse_detected = True
                logger.warning(f"[Mode Collapse Detected] Episode {self.current_episode}: {result['reasons']}")
        else:
            self._collapse_detected = False
        
        return result
    
    def _hash_states(self, state_emb: torch.Tensor, num_bins: int = 100) -> List[int]:
        """Hash state embeddings for coverage tracking."""
        # Discretize to bins and create hash
        state_np = state_emb.detach().cpu().numpy()
        # Use first few dimensions for hashing
        state_reduced = state_np[:, :min(16, state_np.shape[1])]
        # Discretize
        bins = np.linspace(-3, 3, num_bins)
        digitized = np.digitize(state_reduced, bins)
        # Create hash from digitized values
        hashes = [hash(tuple(row)) for row in digitized]
        return hashes
    
    def get_entropy_coef(self) -> float:
        """Get entropy coefficient, with bonus if collapse detected."""
        base_coef = self.config.entropy_coef
        if self._collapse_detected:
            return base_coef + self.config.entropy_bonus_on_collapse
        return base_coef

    def forward_explorer(
        self,
        batch: Dict[str, torch.Tensor],
        actions: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Explorer.
        
        Args:
            batch: Input batch (can include images for full vision path)
            actions: Optional actions to evaluate
            deterministic: If True, use mean action
            
        Returns:
            actions: Sampled or provided actions (B, action_dim)
            log_probs: Log probabilities (B,)
            values: Value estimates (B,)
        """
        # Check if we have full image data for vision understanding
        has_images = 'observation.images.image0' in batch
        
        if has_images:
            B = batch['observation.state'].shape[0]
            # Use full forward path with vision encoding
            forward_batch = {
                'observation.images.image0': batch['observation.images.image0'],
                'observation.images.image0_mask': batch.get('observation.images.image0_mask', 
                    torch.ones(B, dtype=torch.bool, device=batch['observation.state'].device)),
                'observation.state': batch['observation.state'],
                'task': batch.get('task', ["perform the task"] * B),
            }
        else:
            # Fallback: Re-compute state_emb from observation.state to get gradients
            # Don't use pre-computed state_emb as it may be detached
            if 'observation.state' in batch:
                state = batch['observation.state']
                state_emb = self.policy.model.state_proj(state)
                # Create new batch for forward_with_actor with fresh state_emb
                forward_batch = {'state_emb': state_emb}
            else:
                forward_batch = batch
        
        # Forward through policy with explorer actor
        # Include memory_kv if using memory
        forward_kwargs = {
            'actor_name': 'explorer',
            'return_action_stats': True,
        }
        
        # Add memory_kv if available (for KV cache support)
        if self.episode_memory_kv is not None:
            forward_kwargs['memory_kv'] = self.episode_memory_kv
        
        output = self.policy.forward_with_actor(
            forward_batch,
            **forward_kwargs
        )
        
        # Get action mean
        action_mean = output['action']  # (B, action_dim)
        
        # Check for NaN in action_mean and handle gracefully
        if torch.isnan(action_mean).any():
            logger.warning(f"NaN detected in action_mean, replacing with zeros. Input batch keys: {list(forward_batch.keys())}")
            # Debug: log input stats
            for k, v in forward_batch.items():
                if isinstance(v, torch.Tensor):
                    logger.warning(f"  {k}: shape={v.shape}, has_nan={torch.isnan(v).any()}, min={v.min().item():.4f}, max={v.max().item():.4f}")
            action_mean = torch.zeros_like(action_mean)
        
        # Get state_emb for value head (from output or recompute)
        if 'state_emb' in output:
            state_emb = output['state_emb']
        elif 'observation.state' in batch:
            state = batch['observation.state']
            state_emb = self.policy.model.state_proj(state)
        else:
            raise ValueError("Cannot compute state_emb for value head")
        
        # Get std
        std = torch.exp(self.log_std)
        
        # Create distribution
        dist = torch.distributions.Normal(action_mean, std)
        
        if actions is not None:
            # Ensure actions have correct dimension
            if actions.shape[-1] != action_mean.shape[-1]:
                if actions.shape[-1] > action_mean.shape[-1]:
                    # Truncate if got more dimensions than expected
                    actions = actions[..., :action_mean.shape[-1]]
                else:
                    # Pad if got fewer dimensions
                    pad_size = action_mean.shape[-1] - actions.shape[-1]
                    actions = torch.cat([actions, torch.zeros(*actions.shape[:-1], pad_size, device=actions.device)], dim=-1)
            # Evaluate provided actions
            log_probs = dist.log_prob(actions).sum(dim=-1)
        elif deterministic:
            actions = action_mean
            log_probs = torch.zeros(action_mean.shape[0], device=self.device)
        else:
            # Sample actions
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Clamp actions
        actions = torch.clamp(actions, -1.0, 1.0) * self.config.action_scale
        
        # Value estimate using state_emb
        values = self.value_head(state_emb)
        
        return actions, log_probs, values
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0,
    ) -> Tuple[List[float], List[float]]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Bootstrap value
            
        Returns:
            advantages: GAE advantages
            returns: Target returns
        """
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = 1.0
            else:
                next_val = values[t + 1]
                next_done = 1.0 - float(dones[t + 1])
            
            delta = rewards[t] + self.config.gamma * next_val * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def build_ppo_batch(
        self,
        transitions: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Build PPO training batch from transitions.
        
        Args:
            transitions: List of transition dicts
            
        Returns:
            Batch dict with tensors
        """
        # Extract data
        actions = torch.stack([t['action'] for t in transitions]).to(self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in transitions], device=self.device)
        values = [t['value'] for t in transitions]
        rewards = [t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        
        # Normalize rewards if enabled
        if self.config.normalize_rewards and len(rewards) > 1:
            rewards_tensor = torch.tensor(rewards)
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std() + 1e-8
            rewards = ((rewards_tensor - rewards_mean) / rewards_std).tolist()
        
        # Clip rewards if enabled
        if self.config.clip_rewards is not None:
            rewards = [max(-self.config.clip_rewards, min(self.config.clip_rewards, r)) for r in rewards]
        
        # Compute GAE
        next_value = 0.0 if dones[-1] else values[-1]
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        old_values = torch.tensor(values, device=self.device)
        
        # Build observation batch
        # This depends on how transitions store observations
        batch = {
            'actions': actions,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': advantages,
            'returns': returns,
        }
        
        # Add observations if stored in transitions
        if 'batch' in transitions[0]:
            # Concatenate stored batches
            obs_keys = transitions[0]['batch'].keys()
            for key in obs_keys:
                tensors = [t['batch'][key] for t in transitions]
                batch[key] = torch.cat(tensors, dim=0)
        
        return batch
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Execute one PPO training step with mini-batching.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss dict
        """
        batch_size = batch['actions'].shape[0]
        mini_batch_size = min(self.config.mini_batch_size, batch_size)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle indices
        indices = torch.randperm(batch_size, device=self.device)
        
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            mb_indices = indices[start:end]
            
            # Create mini-batch
            mb_actions = batch['actions'][mb_indices]
            mb_old_log_probs = batch['old_log_probs'][mb_indices]
            mb_old_values = batch['old_values'][mb_indices]
            mb_advantages = batch['advantages'][mb_indices]
            mb_returns = batch['returns'][mb_indices]
            
            # Create observation mini-batch
            mb_batch = {}
            for key in batch:
                if key not in ['actions', 'old_log_probs', 'old_values', 'advantages', 'returns']:
                    if key == 'task':
                        # Handle task field (list of strings) with list indexing
                        mb_batch[key] = [batch[key][i.item()] for i in mb_indices]
                    else:
                        mb_batch[key] = batch[key][mb_indices]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            _, log_probs, values = self.forward_explorer(mb_batch, actions=mb_actions)
            
            # Normalize advantages
            if self.config.normalize_advantages and mb_advantages.numel() > 1:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            
            # Policy loss (clipped)
            ratio = torch.exp(log_probs - mb_old_log_probs)
            
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss (clipped)
            values_clipped = mb_old_values + torch.clamp(
                values - mb_old_values,
                -self.config.clip_epsilon,
                self.config.clip_epsilon
            )
            value_loss_unclipped = F.mse_loss(values, mb_returns, reduction='none')
            value_loss_clipped = F.mse_loss(values_clipped, mb_returns, reduction='none')
            value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Entropy bonus
            std = torch.exp(self.log_std)
            entropy = 0.5 * (1 + torch.log(2 * np.pi * std ** 2)).sum()
            
            # Total loss
            loss = (
                policy_loss
                + self.config.value_loss_coef * value_loss
                - self.config.entropy_coef * entropy
            )
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.policy.get_actor('explorer').parameters(),
                self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.value_head.parameters(),
                self.config.max_grad_norm
            )
            
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Track additional metrics (accumulate from last mini-batch)
            if num_batches == 1:
                avg_ratio = ratio.mean().item()
                clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean().item()
                avg_advantage = mb_advantages.mean().item()
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'entropy': total_entropy / num_batches,
            'total_loss': total_loss / num_batches,
            'ratio': avg_ratio,
            'clip_fraction': clip_frac,
            'advantage_mean': avg_advantage,
            'std': std.mean().item(),
        }
    
    def save_checkpoint(self, checkpoint_path: str):
        """Save training checkpoint.
        
        Args:
            checkpoint_path: Full path to save checkpoint (e.g., 'checkpoints/phase1/step_10000.pth')
        """
        import os
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create checkpoint dict
        checkpoint = {
            'optimizer': self.optimizer.state_dict(),
            'value_head': self.value_head.state_dict(),
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save explorer actor separately (for easy loading)
        actor_path = checkpoint_path.replace('.pth', '_explorer.pt')
        self.policy.save_actor('explorer', actor_path)
        
        # Save log_std
        log_std_path = os.path.join(checkpoint_dir, 'log_std.pt')
        torch.save(self.log_std.data, log_std_path)
        
        # Save training state
        state = {
            'global_step': self.global_step,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            'metrics': {k: list(v) for k, v in self.metrics.items()},
        }
        state_path = os.path.join(checkpoint_dir, 'training_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_dir: Path to checkpoint directory
            
        Returns:
            Starting episode number
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load explorer actor
        explorer_path = checkpoint_dir / 'explorer.pt'
        if explorer_path.exists():
            self.policy.load_actor('explorer', str(explorer_path))
        
        # Load value head
        value_head_path = checkpoint_dir / 'value_head.pt'
        if value_head_path.exists():
            self.value_head.load_state_dict(torch.load(value_head_path, map_location=self.device))
        
        # Load optimizer
        optimizer_path = checkpoint_dir / 'optimizer.pt'
        if optimizer_path.exists():
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        
        # Load scheduler
        scheduler_path = checkpoint_dir / 'scheduler.pt'
        if scheduler_path.exists() and self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
        
        # Load log_std
        log_std_path = checkpoint_dir / 'log_std.pt'
        if log_std_path.exists():
            self.log_std.data = torch.load(log_std_path, map_location=self.device)
        
        # Load training state
        state_path = checkpoint_dir / 'training_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state.get('global_step', 0)
            return state.get('episode', 0)
        
        return 0
    
    def log_metrics(self, episode: int):
        """Log training metrics."""
        logger.info(f"Episode {episode}:")
        logger.info(f"  Episode reward: {np.mean(self.metrics['episode_rewards']):.4f}")
        logger.info(f"  Episode length: {np.mean(self.metrics['episode_lengths']):.1f}")
        logger.info(f"  Policy loss: {np.mean(self.metrics['policy_loss']):.6f}")
        logger.info(f"  Value loss: {np.mean(self.metrics['value_loss']):.6f}")
        logger.info(f"  Entropy: {np.mean(self.metrics['entropy']):.6f}")
        logger.info(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # Reward components
        if self.metrics['r1_uncertainty']:
            logger.info(f"  r1 (uncertainty): {np.mean(self.metrics['r1_uncertainty']):.6f}")
        if self.metrics['r2_mse']:
            logger.info(f"  r2 (mse): {np.mean(self.metrics['r2_mse']):.6f}")
        if self.metrics['r3_mse_improvement']:
            logger.info(f"  r3 (mse_improvement): {np.mean(self.metrics['r3_mse_improvement']):.6f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current training statistics."""
        stats = {
            'mean_episode_reward': np.mean(self.metrics['episode_rewards']) if self.metrics['episode_rewards'] else 0.0,
            'mean_episode_length': np.mean(self.metrics['episode_lengths']) if self.metrics['episode_lengths'] else 0.0,
            'mean_policy_loss': np.mean(self.metrics['policy_loss']) if self.metrics['policy_loss'] else 0.0,
            'mean_value_loss': np.mean(self.metrics['value_loss']) if self.metrics['value_loss'] else 0.0,
            'mean_entropy': np.mean(self.metrics['entropy']) if self.metrics['entropy'] else 0.0,
            'global_step': self.global_step,
            'current_episode': self.current_episode,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        return stats


def setup_explorer_phase1_training(
    policy,
    vae,
    config: Optional[ExplorerTrainingConfig] = None,
    device: str = "cuda",
) -> Tuple[ExplorerRLTrainer, Any, Any]:
    """
    Convenience function to setup Phase 1 Explorer training.
    
    Args:
        policy: F1_VLA policy
        vae: VAE model
        config: Training configuration
        device: Device
        
    Returns:
        trainer: ExplorerRLTrainer
        vae_extractor: VAEEmbeddingExtractor
        reward_manager: ExplorerRewardManager
    """
    from .explorer import initialize_explorer, ExplorerConfig
    from .vae_embedding import VAEEmbeddingExtractor
    from .reward_computation import ExplorerRewardManager, RewardConfig
    
    # Initialize explorer if not exists
    if 'explorer' not in policy.list_actors():
        explorer_config = ExplorerConfig(random_init=True)
        initialize_explorer(policy, explorer_config, device)
    
    # Create VAE extractor
    vae_extractor = VAEEmbeddingExtractor(vae, device=device)
    
    # Create reward manager
    reward_config = RewardConfig()
    reward_manager = ExplorerRewardManager(reward_config)
    
    # Create trainer
    trainer = ExplorerRLTrainer(
        policy=policy,
        vae_extractor=vae_extractor,
        reward_manager=reward_manager,
        config=config,
        device=device,
    )
    
    return trainer, vae_extractor, reward_manager
