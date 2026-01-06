"""
Explorer Phase 2 Adversarial Training Module

This module implements Phase 2 of Explorer training:
- Unfreeze World Model
- Alternating training: WM tries to predict, Explorer tries to surprise
- Adversarial game for exploration

Key Features:
- Alternating WM and Explorer updates
- WM: minimize prediction error
- Explorer: maximize WM prediction error (find novel states)
- Balanced training to prevent mode collapse
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
class AdversarialTrainingConfig:
    """Configuration for adversarial training."""
    # Iteration settings
    total_iterations: int = 10000
    steps_per_iteration: int = 100
    
    # Alternating update schedule
    wm_updates_per_iter: int = 5  # WM updates per iteration
    explorer_updates_per_iter: int = 1  # Explorer updates per iteration
    
    # Warmup
    warmup_iterations: int = 100  # WM-only warmup before adversarial
    
    # WM learning
    wm_learning_rate: float = 1e-5
    wm_weight_decay: float = 0.01
    wm_freeze_encoder: bool = True  # Keep vision encoder frozen
    
    # Explorer learning (PPO)
    explorer_learning_rate: float = 3e-5
    explorer_weight_decay: float = 0.01
    ppo_epochs: int = 4
    mini_batch_size: int = 32
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02
    
    # GAE
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # Reward balance
    adversarial_weight: float = 1.0  # Weight for WM prediction error in Explorer reward
    exploration_weight: float = 0.5  # Weight for uncertainty-based exploration
    
    # Gradient control
    max_grad_norm: float = 1.0
    
    # Logging and saving
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "./outputs/adversarial_rl"
    
    # Advanced options
    prevent_collapse: bool = True  # Add regularization to prevent mode collapse
    wm_loss_threshold: float = 0.01  # Min WM loss below which explorer stops being rewarded
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_iterations': self.total_iterations,
            'steps_per_iteration': self.steps_per_iteration,
            'wm_updates_per_iter': self.wm_updates_per_iter,
            'explorer_updates_per_iter': self.explorer_updates_per_iter,
            'warmup_iterations': self.warmup_iterations,
            'wm_learning_rate': self.wm_learning_rate,
            'explorer_learning_rate': self.explorer_learning_rate,
            'adversarial_weight': self.adversarial_weight,
            'exploration_weight': self.exploration_weight,
        }


class WorldModelTrainer:
    """
    Trainer for World Model in adversarial setting.
    
    WM objective: minimize prediction error.
    """
    
    def __init__(
        self,
        policy,
        config: AdversarialTrainingConfig,
        device: str = "cuda",
    ):
        """
        Initialize WM trainer.
        
        Args:
            policy: F1_VLA policy with WM expert
            config: Training configuration
            device: Device
        """
        self.policy = policy
        self.config = config
        self.device = device
        
        # Setup WM for training
        self._setup_wm_training()
        
        # Optimizer
        wm_params = self._get_wm_parameters()
        self.optimizer = torch.optim.AdamW(
            wm_params,
            lr=config.wm_learning_rate,
            weight_decay=config.wm_weight_decay,
        )
        
        # Metrics
        self.losses = deque(maxlen=100)
    
    def _setup_wm_training(self):
        """Configure model for WM training."""
        model = self.policy.model
        
        if hasattr(model, 'paligemma_with_expert'):
            pwm = model.paligemma_with_expert
            
            # Unfreeze WM expert
            if hasattr(pwm, 'gemma_wm_expert'):
                for param in pwm.gemma_wm_expert.parameters():
                    param.requires_grad = True
            
            # Keep vision encoder frozen (if configured)
            if self.config.wm_freeze_encoder and hasattr(pwm, 'paligemma'):
                for param in pwm.paligemma.vision_tower.parameters():
                    param.requires_grad = False
                for param in pwm.paligemma.language_model.parameters():
                    param.requires_grad = False
            
            # Freeze all actors
            if hasattr(pwm, 'gemma_experts'):
                for name, actor in pwm.gemma_experts.items():
                    for param in actor.parameters():
                        param.requires_grad = False
        
        # Keep VAE frozen
        if hasattr(model, 'vae'):
            for param in model.vae.parameters():
                param.requires_grad = False
    
    def _get_wm_parameters(self) -> List[torch.nn.Parameter]:
        """Get WM trainable parameters."""
        params = []
        model = self.policy.model
        
        if hasattr(model, 'paligemma_with_expert'):
            if hasattr(model.paligemma_with_expert, 'gemma_wm_expert'):
                for param in model.paligemma_with_expert.gemma_wm_expert.parameters():
                    if param.requires_grad:
                        params.append(param)
        
        return params
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        gt_next_imgs: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one WM training step.
        
        Uses F1Policy.forward_with_world_model for world model training.
        
        Args:
            batch: Input batch
            gt_next_imgs: Ground truth next images (not directly used, WM gets GT from batch)
            
        Returns:
            Loss dict
        """
        self.optimizer.zero_grad()
        
        try:
            # Temporarily set train_gen_expert_only=True on the model
            # This makes forward_with_world_model skip action head and only train WM
            original_train_gen_expert_only = getattr(self.policy.model, 'train_gen_expert_only', False)
            self.policy.model.train_gen_expert_only = True
            
            # Use F1Policy's forward_with_world_model method for training
            # This trains the world model using the batch's built-in GT images
            loss_dict = self.policy.forward_with_world_model(
                batch,
                train_gen_expert_only=True,  # Also pass as parameter for f1_policy level
                cur_n_obs_img_steps=4,  # Typically 4 history frames
                cur_n_pred_img_steps=1,  # Predict 1 frame
            )
            
            # Restore original value
            self.policy.model.train_gen_expert_only = original_train_gen_expert_only
            
            wm_loss = loss_dict.get('wm_loss', loss_dict.get('loss', torch.tensor(0.0)))
            if isinstance(wm_loss, torch.Tensor):
                wm_loss_value = wm_loss.item()
            else:
                wm_loss_value = wm_loss
            
            # Backward
            if isinstance(wm_loss, torch.Tensor) and wm_loss.requires_grad:
                wm_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self._get_wm_parameters(),
                    self.config.max_grad_norm
                )
                
                self.optimizer.step()
        except Exception as e:
            logger.error(f"WM train_step failed: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise the exception instead of using dummy loss
            # This ensures bugs are caught rather than silently ignored
            # Restore model state before re-raising
            if hasattr(self, 'policy') and hasattr(self.policy, 'model'):
                self.policy.model.train_gen_expert_only = False
            raise RuntimeError(f"WM train_step failed: {e}") from e
        
        self.losses.append(wm_loss_value)
        
        return {'wm_loss': wm_loss_value}
    
    def get_prediction_error(
        self,
        batch: Dict[str, torch.Tensor],
        gt_next_imgs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get WM prediction error (for Explorer reward).
        
        Args:
            batch: Input batch
            gt_next_imgs: Ground truth next images
            
        Returns:
            MSE error per sample (B,)
        """
        with torch.no_grad():
            try:
                # Temporarily set train_gen_expert_only=True
                original_train_gen_expert_only = getattr(self.policy.model, 'train_gen_expert_only', False)
                self.policy.model.train_gen_expert_only = True
                
                # Use forward_with_world_model to get predictions and accuracy
                loss_dict = self.policy.forward_with_world_model(
                    batch,
                    train_gen_expert_only=True,
                    cur_n_obs_img_steps=4,
                    cur_n_pred_img_steps=1,
                )
                
                # Restore
                self.policy.model.train_gen_expert_only = original_train_gen_expert_only
                
                # WM accuracy gives us an indication of prediction quality
                # Lower accuracy = higher prediction error = higher reward for explorer
                wm_acc = loss_dict.get('wm_acc_mean', torch.tensor(1.0))
                
                # Convert accuracy to error: error = 1 - accuracy
                # This gives higher error (reward) when WM predictions are worse
                batch_size = batch.get('observation.state', batch.get('action', gt_next_imgs)).shape[0]
                error = (1.0 - wm_acc) * torch.ones(batch_size, device=self.device)
            except Exception as e:
                logger.warning(f"get_prediction_error failed: {e}")
                batch_size = batch.get('observation.state', batch.get('action', gt_next_imgs)).shape[0]
                error = torch.ones(batch_size, device=self.device) * 0.5  # Default error
                if hasattr(self, 'policy') and hasattr(self.policy, 'model'):
                    self.policy.model.train_gen_expert_only = False
            
            return error
    
    def get_statistics(self) -> Dict[str, float]:
        """Get WM training statistics."""
        return {
            'mean_wm_loss': np.mean(self.losses) if self.losses else 0.0,
        }


class AdversarialExplorerTrainer:
    """
    Trainer for Explorer in adversarial setting.
    
    Explorer objective: find actions that maximize WM prediction error.
    """
    
    def __init__(
        self,
        policy,
        wm_trainer: WorldModelTrainer,
        vae_extractor,
        config: AdversarialTrainingConfig,
        device: str = "cuda",
    ):
        """
        Initialize adversarial Explorer trainer.
        
        Args:
            policy: F1_VLA policy with Explorer actor
            wm_trainer: World Model trainer
            vae_extractor: VAE embedding extractor
            config: Training configuration
            device: Device
        """
        self.policy = policy
        self.wm_trainer = wm_trainer
        self.vae_extractor = vae_extractor
        self.config = config
        self.device = device
        
        # Ensure explorer exists
        if 'explorer' not in policy.list_actors():
            raise ValueError("Explorer actor not found")
        
        # Setup explorer for training
        self._setup_explorer_training()
        
        # Value head
        proj_width = getattr(policy.model.config, 'proj_width', 1024)
        self.value_head = nn.Linear(proj_width, 1).to(device)
        
        # Action log std
        
        # Step counter
        self.global_step = 0
        self.log_std = nn.Parameter(
            torch.zeros(config.mini_batch_size, device=device) - 1.0
        )
        
        # Optimizer
        explorer_params = list(policy.get_actor('explorer').parameters())
        self.optimizer = torch.optim.AdamW(
            [
                {'params': explorer_params, 'lr': config.explorer_learning_rate},
                {'params': self.value_head.parameters(), 'lr': config.explorer_learning_rate},
                {'params': [self.log_std], 'lr': config.explorer_learning_rate * 0.1},
            ],
            weight_decay=config.explorer_weight_decay,
        )
        
        # Metrics
        self.policy_losses = deque(maxlen=100)
        self.value_losses = deque(maxlen=100)
        self.adversarial_rewards = deque(maxlen=100)
    
    def _setup_explorer_training(self):
        """Configure model for Explorer training."""
        # Set explorer as active
        self.policy.active_actor = 'explorer'
        
        # Make only explorer trainable
        self.policy.set_trainable_actors(['explorer'])
        
        # Freeze WM (trained separately)
        model = self.policy.model
        if hasattr(model, 'paligemma_with_expert'):
            if hasattr(model.paligemma_with_expert, 'gemma_wm_expert'):
                for param in model.paligemma_with_expert.gemma_wm_expert.parameters():
                    param.requires_grad = False
    
    def compute_adversarial_reward(
        self,
        batch: Dict[str, torch.Tensor],
        gt_next_imgs: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute adversarial reward for Explorer.
        
        Higher reward for:
        1. Higher WM prediction error (novel states)
        2. Higher WM uncertainty (exploration)
        
        Args:
            batch: Input batch
            gt_next_imgs: Ground truth next images
            uncertainty: Optional uncertainty values
            
        Returns:
            Reward per sample (B,)
        """
        # WM prediction error
        pred_error = self.wm_trainer.get_prediction_error(batch, gt_next_imgs)
        
        # Apply threshold to prevent trivial solutions
        if self.config.prevent_collapse:
            threshold = self.config.wm_loss_threshold
            # Reward only when WM can't predict well
            pred_error = torch.where(
                pred_error > threshold,
                pred_error,
                torch.zeros_like(pred_error)
            )
        
        # Combine with uncertainty
        reward = self.config.adversarial_weight * pred_error
        
        if uncertainty is not None:
            reward = reward + self.config.exploration_weight * uncertainty
        
        self.adversarial_rewards.append(reward.mean().item())
        
        return reward
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        actions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Execute one PPO training step for Explorer.
        
        Args:
            batch: Input batch
            old_log_probs: Old action log probs
            old_values: Old value estimates
            advantages: GAE advantages
            returns: Target returns
            actions: Actions taken
            
        Returns:
            Loss dict
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        output = self.policy.forward_with_actor(
            batch,
            actor_name='explorer',
            return_action_stats=True,
        )
        
        action_mean = output['action']
        std = torch.exp(self.log_std[:action_mean.shape[-1]])
        dist = torch.distributions.Normal(action_mean, std)
        
        # New log probs
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Value estimate
        state_emb = output.get('state_emb')
        if state_emb is None:
            state_emb = self.policy.model.state_proj(batch['observation.state'])
        values = self.value_head(state_emb).squeeze(-1)  # [batch, 1] -> [batch]
        
        # Ensure old_values is also squeezed
        if old_values.dim() > 1:
            old_values = old_values.squeeze(-1)
        
        # Ensure returns has correct shape
        if returns.dim() > 1:
            returns = returns.squeeze(-1)
        
        # Normalize advantages only for large batches
        if self.config.ppo_epochs > 1 and advantages.numel() >= 8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss (clipped)
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Debug logging
        if self.global_step % 10 == 0:
            logger.debug(f"[Explorer PPO] ratio: mean={ratio.mean():.4f}, min={ratio.min():.4f}, max={ratio.max():.4f}")
            logger.debug(f"[Explorer PPO] log_probs: mean={log_probs.mean():.4f}, old: {old_log_probs.mean():.4f}")
            logger.debug(f"[Explorer PPO] advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
        
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.config.clip_epsilon,
            self.config.clip_epsilon
        )
        value_loss_unclipped = F.mse_loss(values, returns, reduction='none')
        value_loss_clipped = F.mse_loss(values_clipped, returns, reduction='none')
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
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
        
        self.optimizer.step()
        
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.global_step += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get Explorer training statistics."""
        return {
            'mean_policy_loss': np.mean(self.policy_losses) if self.policy_losses else 0.0,
            'mean_value_loss': np.mean(self.value_losses) if self.value_losses else 0.0,
            'mean_adversarial_reward': np.mean(self.adversarial_rewards) if self.adversarial_rewards else 0.0,
        }


class AdversarialTrainingManager:
    """
    High-level manager for adversarial training.
    
    Coordinates WM and Explorer training in alternating fashion.
    """
    
    def __init__(
        self,
        policy,
        vae,
        config: Optional[AdversarialTrainingConfig] = None,
        device: str = "cuda",
    ):
        """
        Initialize adversarial training manager.
        
        Args:
            policy: F1_VLA policy
            vae: VAE model
            config: Training configuration
            device: Device
        """
        from .vae_embedding import VAEEmbeddingExtractor
        
        self.policy = policy
        self.config = config or AdversarialTrainingConfig()
        self.device = device
        
        # Create VAE extractor
        self.vae_extractor = VAEEmbeddingExtractor(vae)
        self.vae_extractor = self.vae_extractor.to(device)
        
        # Create trainers
        self.wm_trainer = WorldModelTrainer(policy, self.config, device)
        self.explorer_trainer = AdversarialExplorerTrainer(
            policy, self.wm_trainer, self.vae_extractor, self.config, device
        )
        
        # Training state
        self.iteration = 0
        self.global_step = 0
        
        # Metrics
        self.metrics = {
            'wm_loss': deque(maxlen=100),
            'explorer_policy_loss': deque(maxlen=100),
            'explorer_value_loss': deque(maxlen=100),
            'adversarial_reward': deque(maxlen=100),
        }
        
        # Output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_iteration(
        self,
        episode_data: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Execute one adversarial training iteration.
        
        Args:
            episode_data: List of transition dicts from rollout
            
        Returns:
            Metrics dict
        """
        iteration_metrics = {}
        
        # Phase 1: WM updates
        wm_losses = []
        for _ in range(self.config.wm_updates_per_iter):
            for transition in episode_data:
                batch = transition.get('batch', {})
                gt_next_imgs = transition.get('gt_next_imgs')
                
                if gt_next_imgs is None:
                    continue
                
                loss_dict = self.wm_trainer.train_step(batch, gt_next_imgs)
                wm_losses.append(loss_dict['wm_loss'])
        
        if wm_losses:
            iteration_metrics['wm_loss'] = np.mean(wm_losses)
            self.metrics['wm_loss'].append(iteration_metrics['wm_loss'])
        
        # Phase 2: Explorer updates (skip during warmup)
        if self.iteration >= self.config.warmup_iterations:
            # Compute adversarial rewards
            for transition in episode_data:
                batch = transition.get('batch', {})
                gt_next_imgs = transition.get('gt_next_imgs')
                
                if gt_next_imgs is not None:
                    adv_reward = self.explorer_trainer.compute_adversarial_reward(
                        batch, gt_next_imgs
                    )
                    transition['adversarial_reward'] = adv_reward.mean().item()
            
            # Build PPO batch and train
            ppo_batch = self._build_ppo_batch(episode_data)
            
            if ppo_batch is not None:
                for _ in range(self.config.explorer_updates_per_iter):
                    for _ in range(self.config.ppo_epochs):
                        loss_dict = self.explorer_trainer.train_step(**ppo_batch)
                
                iteration_metrics.update({
                    'explorer_policy_loss': loss_dict['policy_loss'],
                    'explorer_value_loss': loss_dict['value_loss'],
                    'explorer_entropy': loss_dict['entropy'],
                })
                
                self.metrics['explorer_policy_loss'].append(loss_dict['policy_loss'])
                self.metrics['explorer_value_loss'].append(loss_dict['value_loss'])
        
        self.iteration += 1
        self.global_step += len(episode_data)
        
        return iteration_metrics
    
    def train_step_offline(
        self,
        batch: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, float]:
        """
        Offline adversarial training step using dataset batch.
        
        Unlike train_iteration which uses rollout data, this method
        works with dataset batches directly for Phase 2 adversarial training.
        
        Args:
            batch: Dataset batch containing images, actions, etc.
            iteration: Current iteration number
            
        Returns:
            Metrics dict
        """
        iteration_metrics = {}
        
        # Get ground truth next images from batch
        # Try different possible keys
        gt_next_imgs = batch.get('pixel_values')  # [B, T, C, H, W]
        if gt_next_imgs is None:
            gt_next_imgs = batch.get('observation.images.image0_target')
        if gt_next_imgs is None:
            gt_next_imgs = batch.get('observation.images.image0_history')
            if gt_next_imgs is not None and gt_next_imgs.dim() == 5:
                # Take last frame as "target"
                gt_next_imgs = gt_next_imgs[:, -1:]  # [B, 1, C, H, W]
        
        if gt_next_imgs is None:
            logger.warning("No image data found in batch, skipping")
            return iteration_metrics
        
        # Make sure it's on the right device
        if isinstance(gt_next_imgs, torch.Tensor):
            gt_next_imgs = gt_next_imgs.to(self.device)
        
        # Phase 1: Train WM on batch
        wm_losses = []
        for _ in range(self.config.wm_updates_per_iter):
            loss_dict = self.wm_trainer.train_step(batch, gt_next_imgs)
            wm_losses.append(loss_dict['wm_loss'])
        
        if wm_losses:
            iteration_metrics['wm_loss'] = np.mean(wm_losses)
            self.metrics['wm_loss'].append(iteration_metrics['wm_loss'])
        
        # Phase 2: Train Explorer adversarially (after warmup)
        if iteration >= self.config.warmup_iterations:
            # Explorer tries to fool the WM
            # Compute adversarial reward based on WM prediction error
            adv_reward = self.explorer_trainer.compute_adversarial_reward(
                batch, gt_next_imgs
            )
            mean_adv_reward = adv_reward.mean().item()
            iteration_metrics['adversarial_reward'] = mean_adv_reward
            self.metrics['adversarial_reward'].append(mean_adv_reward)
            
            # Generate explorer actions and train with PPO
            with torch.no_grad():
                # Forward through explorer to get actions
                explorer_output = self.policy.forward_with_actor(
                    batch,
                    actor_name='explorer',
                    return_action_stats=True,
                )
                
                action_mean = explorer_output['action']
                # Sample actions
                std = torch.exp(self.explorer_trainer.log_std[:action_mean.shape[-1]])
                dist = torch.distributions.Normal(action_mean, std)
                actions = dist.sample()
                log_probs = dist.log_prob(actions).sum(dim=-1)
                
                # Compute value estimate from state embedding
                state_emb = explorer_output.get('state_emb')
                if state_emb is None:
                    state_emb = self.policy.model.state_proj(batch['observation.state'].to(self.device))
                values = self.explorer_trainer.value_head(state_emb).squeeze(-1)
            
            # For offline training, we use single-step update
            batch_size = gt_next_imgs.shape[0]
            
            # Prepare PPO batch
            old_log_probs = log_probs.detach()
            old_values = values.detach()
            
            # Compute simple advantages (single step)
            rewards = adv_reward.detach()
            advantages = rewards - old_values
            returns = rewards
            
            # Normalize advantages only if batch is large enough
            # For very small batches, normalization removes useful signal
            if advantages.numel() >= 8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif advantages.numel() > 1 and advantages.std() > 1e-6:
                # For small batches, just center without scaling
                advantages = advantages - advantages.mean()
            
            # Train explorer with PPO
            for _ in range(self.config.explorer_updates_per_iter):
                for _ in range(self.config.ppo_epochs):
                    loss_dict = self.explorer_trainer.train_step(
                        batch=batch,
                        old_log_probs=old_log_probs,
                        old_values=old_values,
                        advantages=advantages,
                        returns=returns,
                        actions=actions,
                    )
            
            iteration_metrics.update({
                'explorer_policy_loss': loss_dict['policy_loss'],
                'explorer_value_loss': loss_dict['value_loss'],
                'explorer_entropy': loss_dict.get('entropy', 0.0),
            })
            
            self.metrics['explorer_policy_loss'].append(loss_dict['policy_loss'])
            self.metrics['explorer_value_loss'].append(loss_dict['value_loss'])
        
        self.iteration = iteration
        self.global_step += 1
        
        return iteration_metrics
    
    def _build_ppo_batch(
        self,
        episode_data: List[Dict[str, Any]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build PPO batch from episode data."""
        if not episode_data:
            return None
        
        # Extract data
        actions = []
        log_probs = []
        values = []
        rewards = []
        
        for t in episode_data:
            if 'action' in t:
                actions.append(t['action'])
                log_probs.append(t.get('log_prob', 0.0))
                values.append(t.get('value', 0.0))
                # Use adversarial reward if available, else regular reward
                reward = t.get('adversarial_reward', t.get('reward', 0.0))
                rewards.append(reward)
        
        if not actions:
            return None
        
        # Convert to tensors
        actions = torch.stack([
            torch.tensor(a) if not isinstance(a, torch.Tensor) else a
            for a in actions
        ]).to(self.device)
        
        old_log_probs = torch.tensor(log_probs, device=self.device)
        old_values = torch.tensor(values, device=self.device)
        
        # Compute advantages
        advantages, returns = self._compute_gae(rewards, values)
        
        # Get observation batch from first transition
        batch = episode_data[0].get('batch', {})
        
        return {
            'batch': batch,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': torch.tensor(advantages, device=self.device),
            'returns': torch.tensor(returns, device=self.device),
            'actions': actions,
        }
    
    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_val - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        return advantages, returns
    
    def save_checkpoint(self, iteration: int):
        """Save checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint_{iteration:06d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save explorer
        self.policy.save_actor('explorer', str(checkpoint_dir / 'explorer.pt'))
        
        # Save WM optimizer
        torch.save(self.wm_trainer.optimizer.state_dict(), checkpoint_dir / 'wm_optimizer.pt')
        
        # Save explorer optimizer
        torch.save(self.explorer_trainer.optimizer.state_dict(), checkpoint_dir / 'explorer_optimizer.pt')
        
        # Save value head
        torch.save(self.explorer_trainer.value_head.state_dict(), checkpoint_dir / 'value_head.pt')
        
        # Save training state
        state = {
            'iteration': iteration,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
        }
        with open(checkpoint_dir / 'training_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved adversarial checkpoint at iteration {iteration}")
    
    def load_checkpoint(self, checkpoint_dir: str) -> int:
        """Load checkpoint and return starting iteration."""
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load explorer
        explorer_path = checkpoint_dir / 'explorer.pt'
        if explorer_path.exists():
            self.policy.load_actor('explorer', str(explorer_path))
        
        # Load optimizers
        wm_opt_path = checkpoint_dir / 'wm_optimizer.pt'
        if wm_opt_path.exists():
            self.wm_trainer.optimizer.load_state_dict(
                torch.load(wm_opt_path, map_location=self.device)
            )
        
        exp_opt_path = checkpoint_dir / 'explorer_optimizer.pt'
        if exp_opt_path.exists():
            self.explorer_trainer.optimizer.load_state_dict(
                torch.load(exp_opt_path, map_location=self.device)
            )
        
        # Load value head
        value_path = checkpoint_dir / 'value_head.pt'
        if value_path.exists():
            self.explorer_trainer.value_head.load_state_dict(
                torch.load(value_path, map_location=self.device)
            )
        
        # Load state
        state_path = checkpoint_dir / 'training_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
            self.global_step = state.get('global_step', 0)
            return state.get('iteration', 0)
        
        return 0
    
    def log_metrics(self, iteration: int):
        """Log training metrics."""
        wm_stats = self.wm_trainer.get_statistics()
        exp_stats = self.explorer_trainer.get_statistics()
        
        logger.info(f"Iteration {iteration}:")
        logger.info(f"  WM Loss: {wm_stats['mean_wm_loss']:.6f}")
        logger.info(f"  Explorer Policy Loss: {exp_stats['mean_policy_loss']:.6f}")
        logger.info(f"  Explorer Value Loss: {exp_stats['mean_value_loss']:.6f}")
        logger.info(f"  Adversarial Reward: {exp_stats['mean_adversarial_reward']:.6f}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get all training statistics."""
        stats = {
            'iteration': self.iteration,
            'global_step': self.global_step,
        }
        stats.update(self.wm_trainer.get_statistics())
        stats.update(self.explorer_trainer.get_statistics())
        return stats


def setup_adversarial_training(
    policy,
    vae,
    config: Optional[AdversarialTrainingConfig] = None,
    device: str = "cuda",
) -> AdversarialTrainingManager:
    """
    Convenience function to setup adversarial training.
    
    Args:
        policy: F1_VLA policy
        vae: VAE model
        config: Training configuration
        device: Device
        
    Returns:
        AdversarialTrainingManager
    """
    from .explorer import initialize_explorer, ExplorerConfig
    
    # Initialize explorer if not exists
    if 'explorer' not in policy.list_actors():
        explorer_config = ExplorerConfig(random_init=True)
        initialize_explorer(policy, explorer_config, device)
    
    # Create manager
    manager = AdversarialTrainingManager(
        policy=policy,
        vae=vae,
        config=config,
        device=device,
    )
    
    return manager
