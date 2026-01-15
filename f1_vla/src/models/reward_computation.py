"""
Reward Computation Module for Explorer RL Training

This module implements the reward computation for the Explorer actor.
The reward encourages actions that lead to high WM uncertainty and
information gain (MSE improvement).

Reward Formula:
    r1 = uncertainty_{t+1}  (immediate)
    r2 = MSE(pred_emb_{t+1}, emb_{t+1})  (immediate)
    r3 = MSE_{t+1} - MSE_{t+2}  (delayed 1 step)
    r4 = unc_{t+1} - unc_{t+2}  (delayed 1 step, optional)
    
    reward = alpha*r1 + beta*r2 + gamma*r3 + epsilon*r4 - delta*|a_t|

Key Features:
- Supports both immediate and delayed reward components
- Uses VAE embedding space for MSE computation
- Handles temporal alignment between predictions and ground truth
- Supports reward normalization and clipping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for individual reward components."""
    r1_uncertainty: torch.Tensor  # uncertainty_{t+1}
    r2_mse: torch.Tensor  # MSE(pred_emb_{t+1}, emb_{t+1})
    r3_mse_improvement: Optional[torch.Tensor] = None  # MSE_{t+1} - MSE_{t+2}
    r4_uncertainty_improvement: Optional[torch.Tensor] = None  # unc_{t+1} - unc_{t+2}
    action_penalty: torch.Tensor = None  # |a_t|
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary for logging."""
        result = {
            'r1_uncertainty': self.r1_uncertainty,
            'r2_mse': self.r2_mse,
        }
        if self.r3_mse_improvement is not None:
            result['r3_mse_improvement'] = self.r3_mse_improvement
        if self.r4_uncertainty_improvement is not None:
            result['r4_uncertainty_improvement'] = self.r4_uncertainty_improvement
        if self.action_penalty is not None:
            result['action_penalty'] = self.action_penalty
        return result


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    # Reward weights (alpha, beta, gamma, epsilon, delta)
    uncertainty_weight: float = 1.0  # alpha: weight for r1
    mse_weight: float = 1.0  # beta: weight for r2
    mse_improvement_weight: float = 1.0  # gamma: weight for r3
    uncertainty_improvement_weight: float = 0.5  # epsilon: weight for r4
    action_penalty_weight: float = 0.01  # delta: weight for action penalty
    
    # Normalization
    normalize_reward: bool = True
    reward_scale: float = 1.0
    reward_clip: Optional[float] = 10.0
    
    # MSE options
    mse_reduction: str = 'mean'  # 'mean' or 'sum'
    
    # Uncertainty options
    uncertainty_type: str = 'entropy'  # 'entropy', 'max_entropy', 'top_k_entropy'
    uncertainty_top_k: int = 10
    
    @classmethod
    def from_explorer_config(cls, explorer_config) -> 'RewardConfig':
        """Create RewardConfig from ExplorerConfig."""
        return cls(
            uncertainty_weight=getattr(explorer_config, 'reward_uncertainty_weight', 1.0),
            mse_weight=getattr(explorer_config, 'reward_mse_weight', 1.0),
            mse_improvement_weight=getattr(explorer_config, 'reward_mse_improvement_weight', 1.0),
            uncertainty_improvement_weight=getattr(explorer_config, 'reward_uncertainty_improvement_weight', 0.5),
            action_penalty_weight=getattr(explorer_config, 'reward_action_penalty_weight', 0.01),
        )


class RewardComputer:
    """
    Computes rewards for Explorer RL training.
    
    The reward encourages the Explorer to find actions that:
    1. Lead to high WM uncertainty (exploration of unknown regions)
    2. Lead to high MSE between prediction and GT (novel states)
    3. Improve WM prediction accuracy after seeing new GT (information gain)
    4. Reduce WM uncertainty after seeing new GT (learning signal)
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward computer.
        
        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        
        # Running stats for normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0
        
    def compute_mse(
        self,
        pred_emb: torch.Tensor,
        gt_emb: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute MSE between predicted and ground truth embeddings.
        
        Args:
            pred_emb: Predicted embeddings (B, D) or (B, T, D)
            gt_emb: Ground truth embeddings (B, D) or (B, T, D)
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            MSE values (scalar or per-sample)
        """
        # Ensure same shape
        assert pred_emb.shape == gt_emb.shape, \
            f"Shape mismatch: {pred_emb.shape} vs {gt_emb.shape}"
        
        # Compute MSE
        mse = F.mse_loss(pred_emb, gt_emb, reduction='none')
        
        # Reduce over feature dimension
        if mse.dim() == 2:  # (B, D)
            mse = mse.mean(dim=-1)  # (B,)
        elif mse.dim() == 3:  # (B, T, D)
            mse = mse.mean(dim=-1).mean(dim=-1)  # (B,)
        
        # Final reduction
        if reduction == 'mean':
            return mse.mean()
        elif reduction == 'sum':
            return mse.sum()
        else:
            return mse
    
    def compute_uncertainty(
        self,
        logits: torch.Tensor,
        method: str = 'entropy',
        top_k: int = 10,
    ) -> torch.Tensor:
        """
        Compute uncertainty from WM logits.
        
        Args:
            logits: WM generation logits (B, num_tokens, vocab_size)
            method: 'entropy', 'max_entropy', 'top_k_entropy'
            top_k: K for top-k entropy
            
        Returns:
            Uncertainty values (B,)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        if method == 'entropy':
            # Full entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, num_tokens)
            return entropy.mean(dim=-1)  # (B,)
            
        elif method == 'max_entropy':
            # Max entropy across tokens
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, num_tokens)
            return entropy.max(dim=-1)[0]  # (B,)
            
        elif method == 'top_k_entropy':
            # Average of top-k highest entropy tokens
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (B, num_tokens)
            k = min(top_k, entropy.size(-1))
            top_k_values, _ = torch.topk(entropy, k, dim=-1)
            return top_k_values.mean(dim=-1)  # (B,)
        
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    def compute_action_penalty(
        self,
        action: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute action magnitude penalty.
        
        Args:
            action: Action tensor (B, action_dim)
            reduction: 'mean', 'sum', or 'none'
            
        Returns:
            Action penalty (scalar or per-sample)
        """
        # L1 norm of action
        penalty = action.abs().mean(dim=-1)  # (B,)
        
        if reduction == 'mean':
            return penalty.mean()
        elif reduction == 'sum':
            return penalty.sum()
        else:
            return penalty
    
    def compute_immediate_reward(
        self,
        pred_emb_t1: torch.Tensor,  # pred_emb_{t+1}
        gt_emb_t1: torch.Tensor,    # emb_{t+1}
        uncertainty_t1: torch.Tensor,  # uncertainty_{t+1} (precomputed or logits)
        action_t: Optional[torch.Tensor] = None,  # a_t
        is_logits: bool = False,  # If True, uncertainty_t1 is logits
    ) -> Tuple[torch.Tensor, RewardComponents]:
        """
        Compute immediate reward components (r1, r2, action_penalty).
        
        This can be computed as soon as we get gt_{t+1}.
        
        Args:
            pred_emb_t1: WM predicted embedding for t+1
            gt_emb_t1: Ground truth embedding for t+1
            uncertainty_t1: Uncertainty at t+1 (precomputed scalar or logits)
            action_t: Action executed at t (for penalty)
            is_logits: If True, compute uncertainty from logits
            
        Returns:
            immediate_reward: Combined r1 + r2 - action_penalty
            components: Individual reward components
        """
        # r1: Uncertainty
        if is_logits:
            r1 = self.compute_uncertainty(
                uncertainty_t1,
                method=self.config.uncertainty_type,
                top_k=self.config.uncertainty_top_k,
            )
        else:
            r1 = uncertainty_t1
        
        # r2: MSE
        r2 = self.compute_mse(
            pred_emb_t1, gt_emb_t1,
            reduction='none' if pred_emb_t1.dim() == 2 else self.config.mse_reduction,
        )
        
        # Action penalty
        if action_t is not None:
            action_penalty = self.compute_action_penalty(action_t, reduction='none')
        else:
            action_penalty = torch.zeros_like(r1)
        
        # Combine immediate components
        immediate_reward = (
            self.config.uncertainty_weight * r1
            + self.config.mse_weight * r2
            - self.config.action_penalty_weight * action_penalty
        )
        
        components = RewardComponents(
            r1_uncertainty=r1,
            r2_mse=r2,
            action_penalty=action_penalty,
        )
        
        return immediate_reward, components
    
    def compute_delayed_reward(
        self,
        mse_t1: torch.Tensor,  # MSE_{t+1}
        mse_t2: torch.Tensor,  # MSE_{t+2}
        uncertainty_t1: Optional[torch.Tensor] = None,  # unc_{t+1}
        uncertainty_t2: Optional[torch.Tensor] = None,  # unc_{t+2}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute delayed reward components (r3, r4).
        
        This requires gt_{t+2}, so it's computed 1 step later.
        
        Args:
            mse_t1: MSE at t+1
            mse_t2: MSE at t+2
            uncertainty_t1: Uncertainty at t+1
            uncertainty_t2: Uncertainty at t+2
            
        Returns:
            delayed_reward: Combined gamma*r3 + epsilon*r4
            r3: MSE improvement
            r4: Uncertainty improvement (or zeros)
        """
        # r3: MSE improvement (positive if prediction improved)
        r3 = mse_t1 - mse_t2
        
        # r4: Uncertainty improvement (positive if WM became more confident)
        if uncertainty_t1 is not None and uncertainty_t2 is not None:
            r4 = uncertainty_t1 - uncertainty_t2
        else:
            r4 = torch.zeros_like(r3)
        
        # Combine delayed components
        delayed_reward = (
            self.config.mse_improvement_weight * r3
            + self.config.uncertainty_improvement_weight * r4
        )
        
        return delayed_reward, r3, r4
    
    def compute_full_reward(
        self,
        pred_emb_t1: torch.Tensor,
        gt_emb_t1: torch.Tensor,
        uncertainty_t1: torch.Tensor,
        pred_emb_t2: torch.Tensor,
        gt_emb_t2: torch.Tensor,
        uncertainty_t2: torch.Tensor,
        action_t: Optional[torch.Tensor] = None,
        is_logits: bool = False,
    ) -> Tuple[torch.Tensor, RewardComponents]:
        """
        Compute full reward with all components.
        
        This requires data from two consecutive steps (t+1 and t+2).
        
        Args:
            pred_emb_t1: WM predicted embedding for t+1
            gt_emb_t1: Ground truth embedding for t+1
            uncertainty_t1: Uncertainty at t+1
            pred_emb_t2: WM predicted embedding for t+2
            gt_emb_t2: Ground truth embedding for t+2
            uncertainty_t2: Uncertainty at t+2
            action_t: Action executed at t
            is_logits: If True, uncertainty is logits
            
        Returns:
            total_reward: Full reward value
            components: All reward components
        """
        # Compute immediate components
        immediate_reward, components = self.compute_immediate_reward(
            pred_emb_t1, gt_emb_t1, uncertainty_t1, action_t, is_logits
        )
        
        # Compute delayed components
        mse_t1 = components.r2_mse
        
        # Compute MSE for t+2
        mse_t2 = self.compute_mse(pred_emb_t2, gt_emb_t2, reduction='none')
        
        # Get uncertainty values
        if is_logits:
            unc_t1 = self.compute_uncertainty(
                uncertainty_t1,
                method=self.config.uncertainty_type,
                top_k=self.config.uncertainty_top_k,
            )
            unc_t2 = self.compute_uncertainty(
                uncertainty_t2,
                method=self.config.uncertainty_type,
                top_k=self.config.uncertainty_top_k,
            )
        else:
            unc_t1 = uncertainty_t1
            unc_t2 = uncertainty_t2
        
        delayed_reward, r3, r4 = self.compute_delayed_reward(
            mse_t1, mse_t2, unc_t1, unc_t2
        )
        
        # Update components
        components.r3_mse_improvement = r3
        components.r4_uncertainty_improvement = r4
        
        # Total reward
        total_reward = immediate_reward + delayed_reward
        
        # Apply normalization and clipping
        total_reward = self._normalize_and_clip(total_reward)
        
        return total_reward, components
    
    def _normalize_and_clip(self, reward: torch.Tensor) -> torch.Tensor:
        """Apply reward normalization and clipping."""
        # Scale
        reward = reward * self.config.reward_scale
        
        # Clip
        if self.config.reward_clip is not None:
            reward = torch.clamp(
                reward,
                -self.config.reward_clip,
                self.config.reward_clip
            )
        
        # Running normalization (optional, for stable training)
        if self.config.normalize_reward and self.reward_count > 0:
            reward = (reward - self.reward_mean) / (self.reward_var ** 0.5 + 1e-8)
        
        return reward
    
    def update_running_stats(self, reward: torch.Tensor):
        """Update running mean and variance for normalization."""
        batch_mean = reward.mean().item()
        batch_var = reward.var().item() if reward.numel() > 1 else 0.0
        batch_count = reward.numel()
        
        # Welford's online algorithm
        delta = batch_mean - self.reward_mean
        tot_count = self.reward_count + batch_count
        
        self.reward_mean = self.reward_mean + delta * batch_count / tot_count
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.reward_count * batch_count / tot_count
        self.reward_var = M2 / tot_count if tot_count > 0 else 1.0
        self.reward_count = tot_count
    
    def reset_running_stats(self):
        """Reset running statistics."""
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0


class RewardBuffer:
    """
    Buffer for storing reward-related data across steps.
    
    Since r3 and r4 require data from t+2, we need to buffer
    data from previous steps.
    """
    
    def __init__(self, max_length: int = 2):
        """
        Initialize reward buffer.
        
        Args:
            max_length: Maximum number of steps to store
        """
        self.max_length = max_length
        
        # Store embeddings and uncertainties
        self.pred_embeddings: List[torch.Tensor] = []
        self.gt_embeddings: List[torch.Tensor] = []
        self.uncertainties: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.mse_values: List[torch.Tensor] = []
        
        # Store immediate rewards
        self.immediate_rewards: List[torch.Tensor] = []
    
    def add(
        self,
        pred_emb: torch.Tensor,
        gt_emb: torch.Tensor,
        uncertainty: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        mse: Optional[torch.Tensor] = None,
        immediate_reward: Optional[torch.Tensor] = None,
    ):
        """Add data for a new step."""
        self.pred_embeddings.append(pred_emb.detach())
        self.gt_embeddings.append(gt_emb.detach())
        self.uncertainties.append(uncertainty.detach())
        
        if action is not None:
            self.actions.append(action.detach())
        if mse is not None:
            self.mse_values.append(mse.detach())
        if immediate_reward is not None:
            self.immediate_rewards.append(immediate_reward.detach())
        
        # Maintain max length
        self._truncate()
    
    def _truncate(self):
        """Truncate buffer to max length."""
        while len(self.pred_embeddings) > self.max_length:
            self.pred_embeddings.pop(0)
        while len(self.gt_embeddings) > self.max_length:
            self.gt_embeddings.pop(0)
        while len(self.uncertainties) > self.max_length:
            self.uncertainties.pop(0)
        while len(self.actions) > self.max_length:
            self.actions.pop(0)
        while len(self.mse_values) > self.max_length:
            self.mse_values.pop(0)
        while len(self.immediate_rewards) > self.max_length:
            self.immediate_rewards.pop(0)
    
    def can_compute_delayed_reward(self) -> bool:
        """Check if we have enough data for delayed reward."""
        return len(self.mse_values) >= 2 and len(self.uncertainties) >= 2
    
    def get_delayed_reward_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get data for computing delayed reward.
        
        Returns:
            mse_t1: MSE at t+1 (second-to-last)
            mse_t2: MSE at t+2 (last)
            unc_t1: Uncertainty at t+1
            unc_t2: Uncertainty at t+2
        """
        if not self.can_compute_delayed_reward():
            raise ValueError("Not enough data for delayed reward")
        
        return (
            self.mse_values[-2],  # MSE_{t+1}
            self.mse_values[-1],  # MSE_{t+2}
            self.uncertainties[-2],  # unc_{t+1}
            self.uncertainties[-1],  # unc_{t+2}
        )
    
    def reset(self):
        """Clear the buffer."""
        self.pred_embeddings.clear()
        self.gt_embeddings.clear()
        self.uncertainties.clear()
        self.actions.clear()
        self.mse_values.clear()
        self.immediate_rewards.clear()
    
    def __len__(self) -> int:
        return len(self.pred_embeddings)


class ExplorerRewardManager:
    """
    High-level manager for Explorer reward computation.
    
    Handles the temporal complexity of immediate vs delayed rewards
    and maintains the reward buffer.
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """
        Initialize reward manager.
        
        Args:
            config: Reward configuration
        """
        self.config = config or RewardConfig()
        self.reward_computer = RewardComputer(config)
        self.reward_buffer = RewardBuffer(max_length=2)
        
        # Track pending rewards (rewards waiting for delayed components)
        self.pending_action_idx: Optional[int] = None
    
    def step(
        self,
        pred_emb: torch.Tensor,
        gt_emb: torch.Tensor,
        uncertainty: torch.Tensor,
        action: torch.Tensor,
        is_logits: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process one step and compute available rewards.
        
        Args:
            pred_emb: WM predicted embedding (pred_emb_{t+1})
            gt_emb: Ground truth embedding (emb_{t+1})
            uncertainty: Uncertainty at t+1 (scalar or logits)
            action: Action executed at t (a_t)
            is_logits: If True, uncertainty is logits
            
        Returns:
            reward: Reward for the previous action (if delayed components available)
            info: Dictionary with reward components and metadata
        """
        info = {}
        
        # Compute uncertainty value if logits provided
        if is_logits:
            unc_value = self.reward_computer.compute_uncertainty(
                uncertainty,
                method=self.config.uncertainty_type,
                top_k=self.config.uncertainty_top_k,
            )
        else:
            unc_value = uncertainty
        
        # Compute MSE for this step
        mse = self.reward_computer.compute_mse(pred_emb, gt_emb, reduction='none')
        
        # Compute immediate reward for current action
        immediate_reward, components = self.reward_computer.compute_immediate_reward(
            pred_emb, gt_emb, unc_value, action, is_logits=False
        )
        
        # Add to buffer
        self.reward_buffer.add(
            pred_emb=pred_emb,
            gt_emb=gt_emb,
            uncertainty=unc_value,
            action=action,
            mse=mse,
            immediate_reward=immediate_reward,
        )
        
        # Store immediate components in info
        info['r1_uncertainty'] = components.r1_uncertainty.mean().item()
        info['r2_mse'] = components.r2_mse.mean().item()
        info['action_penalty'] = components.action_penalty.mean().item()
        info['immediate_reward'] = immediate_reward.mean().item()
        
        # Try to compute delayed reward for previous action
        reward = None
        if self.reward_buffer.can_compute_delayed_reward():
            mse_t1, mse_t2, unc_t1, unc_t2 = self.reward_buffer.get_delayed_reward_data()
            
            delayed_reward, r3, r4 = self.reward_computer.compute_delayed_reward(
                mse_t1, mse_t2, unc_t1, unc_t2
            )
            
            # Get the immediate reward from buffer
            prev_immediate = self.reward_buffer.immediate_rewards[-2]
            
            # Full reward for previous action
            reward = prev_immediate + delayed_reward
            reward = self.reward_computer._normalize_and_clip(reward)
            
            # Update running stats
            self.reward_computer.update_running_stats(reward)
            
            # Add delayed components to info
            info['r3_mse_improvement'] = r3.mean().item()
            info['r4_uncertainty_improvement'] = r4.mean().item()
            info['delayed_reward'] = delayed_reward.mean().item()
            info['full_reward'] = reward.mean().item()
        
        return reward, info
    
    def finalize_episode(self) -> Optional[torch.Tensor]:
        """
        Finalize rewards at end of episode.
        
        Returns immediate reward only for the last action (no delayed component).
        
        Returns:
            reward: Immediate-only reward for the last action
        """
        if len(self.reward_buffer.immediate_rewards) > 0:
            # Last action only gets immediate reward
            return self.reward_buffer.immediate_rewards[-1]
        return None
    
    def reset(self):
        """Reset for new episode."""
        self.reward_buffer.reset()
        self.pending_action_idx = None
    
    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        return {
            'reward_mean': self.reward_computer.reward_mean,
            'reward_std': self.reward_computer.reward_var ** 0.5,
            'reward_count': self.reward_computer.reward_count,
        }


def compute_batch_rewards(
    pred_embeddings: torch.Tensor,  # (B, T, D)
    gt_embeddings: torch.Tensor,  # (B, T, D)
    uncertainties: torch.Tensor,  # (B, T) or (B, T, num_tokens, vocab_size)
    actions: torch.Tensor,  # (B, T, action_dim)
    config: Optional[RewardConfig] = None,
    is_logits: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute rewards for a batch of trajectories.
    
    This is useful for offline RL or batch processing.
    
    Args:
        pred_embeddings: WM predictions (B, T, D)
        gt_embeddings: Ground truth (B, T, D)
        uncertainties: Uncertainty values (B, T) or logits (B, T, num_tokens, vocab_size)
        actions: Actions (B, T, action_dim)
        config: Reward configuration
        is_logits: If True, uncertainties are logits
        
    Returns:
        rewards: Reward values (B, T-1) - one less than sequence length
        info: Dictionary with reward components
    """
    config = config or RewardConfig()
    reward_computer = RewardComputer(config)
    
    B, T, D = pred_embeddings.shape
    device = pred_embeddings.device
    
    # Compute MSE for all timesteps
    mse_all = []
    for t in range(T):
        mse = reward_computer.compute_mse(
            pred_embeddings[:, t], gt_embeddings[:, t], reduction='none'
        )
        mse_all.append(mse)
    mse_all = torch.stack(mse_all, dim=1)  # (B, T)
    
    # Compute uncertainties for all timesteps
    if is_logits:
        unc_all = []
        for t in range(T):
            unc = reward_computer.compute_uncertainty(
                uncertainties[:, t],
                method=config.uncertainty_type,
                top_k=config.uncertainty_top_k,
            )
            unc_all.append(unc)
        unc_all = torch.stack(unc_all, dim=1)  # (B, T)
    else:
        unc_all = uncertainties  # (B, T)
    
    # Compute action penalties
    action_penalty_all = []
    for t in range(T):
        penalty = reward_computer.compute_action_penalty(
            actions[:, t], reduction='none'
        )
        action_penalty_all.append(penalty)
    action_penalty_all = torch.stack(action_penalty_all, dim=1)  # (B, T)
    
    # Compute rewards for t = 0 to T-2 (need t+1 and t+2)
    # Reward for action a_t is based on state s_{t+1}
    rewards = []
    r1_all, r2_all, r3_all, r4_all = [], [], [], []
    
    for t in range(T - 1):
        # r1: uncertainty_{t+1} - reward for action a_t based on next state
        r1 = unc_all[:, t + 1] if t + 1 < T else unc_all[:, t]
        
        # r2: MSE_{t+1} - prediction error at next state
        r2 = mse_all[:, t + 1] if t + 1 < T else mse_all[:, t]
        
        # r3: MSE_{t+1} - MSE_{t+2} (improvement, if t+2 available)
        if t + 2 < T:
            r3 = mse_all[:, t + 1] - mse_all[:, t + 2]
        else:
            r3 = torch.zeros_like(r1)
        
        # r4: unc_{t+1} - unc_{t+2} (uncertainty reduction, if t+2 available)
        if t + 2 < T:
            r4 = unc_all[:, t + 1] - unc_all[:, t + 2]
        else:
            r4 = torch.zeros_like(r1)
        
        # Action penalty for a_t
        action_penalty = action_penalty_all[:, t]
        
        # Combine
        reward = (
            config.uncertainty_weight * r1
            + config.mse_weight * r2
            + config.mse_improvement_weight * r3
            + config.uncertainty_improvement_weight * r4
            - config.action_penalty_weight * action_penalty
        )
        
        rewards.append(reward)
        r1_all.append(r1)
        r2_all.append(r2)
        r3_all.append(r3)
        r4_all.append(r4)
    
    rewards = torch.stack(rewards, dim=1)  # (B, T-1)
    
    info = {
        'r1_uncertainty': torch.stack(r1_all, dim=1),  # (B, T-1)
        'r2_mse': torch.stack(r2_all, dim=1),
        'r3_mse_improvement': torch.stack(r3_all, dim=1),
        'r4_uncertainty_improvement': torch.stack(r4_all, dim=1),
        'action_penalty': action_penalty_all[:, :-1],
    }
    
    return rewards, info
