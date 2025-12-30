"""
Explorer Actor Module

This module provides the Explorer actor for RL-based exploration training.
The Explorer learns to find actions that maximize World Model uncertainty
and information gain.

Key features:
- Random initialization (no pretrained weights)
- Same architecture as standard actor expert
- Compatible with multi-actor framework
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ExplorerConfig:
    """Configuration for Explorer actor."""
    
    def __init__(
        self,
        # Actor initialization
        random_init: bool = True,  # Random init vs copy from actor
        actor_checkpoint: Optional[str] = None,  # Optional checkpoint to load
        
        # Reward weights
        reward_uncertainty_weight: float = 1.0,  # r1: uncertainty_{t+1}
        reward_mse_weight: float = 1.0,  # r2: MSE(pred_emb_{t+1}, emb_{t+1})
        reward_mse_improvement_weight: float = 1.0,  # r3: MSE_{t+1} - MSE_{t+2}
        reward_uncertainty_improvement_weight: float = 0.5,  # r4: unc_{t+1} - unc_{t+2}
        reward_action_penalty_weight: float = 0.01,  # -|a_t|
        
        # Training
        freeze_world_model: bool = True,  # Phase 1: freeze WM
        freeze_actor: bool = True,  # Freeze the policy actor
        
        **kwargs,
    ):
        self.random_init = random_init
        self.actor_checkpoint = actor_checkpoint
        
        # Reward weights
        self.reward_uncertainty_weight = reward_uncertainty_weight
        self.reward_mse_weight = reward_mse_weight
        self.reward_mse_improvement_weight = reward_mse_improvement_weight
        self.reward_uncertainty_improvement_weight = reward_uncertainty_improvement_weight
        self.reward_action_penalty_weight = reward_action_penalty_weight
        
        # Training flags
        self.freeze_world_model = freeze_world_model
        self.freeze_actor = freeze_actor
        
        # Store additional kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'random_init': self.random_init,
            'actor_checkpoint': self.actor_checkpoint,
            'reward_uncertainty_weight': self.reward_uncertainty_weight,
            'reward_mse_weight': self.reward_mse_weight,
            'reward_mse_improvement_weight': self.reward_mse_improvement_weight,
            'reward_uncertainty_improvement_weight': self.reward_uncertainty_improvement_weight,
            'reward_action_penalty_weight': self.reward_action_penalty_weight,
            'freeze_world_model': self.freeze_world_model,
            'freeze_actor': self.freeze_actor,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplorerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


def initialize_explorer(
    policy,
    explorer_config: Optional[ExplorerConfig] = None,
    device: Optional[torch.device] = None,
) -> None:
    """
    Initialize the Explorer actor in the policy model.
    
    Args:
        policy: F1_VLA policy instance
        explorer_config: Explorer configuration (default: random init)
        device: Device to move the explorer to
    """
    if explorer_config is None:
        explorer_config = ExplorerConfig()
    
    # Check if explorer already exists
    if 'explorer' in policy.list_actors():
        logger.warning("Explorer already exists. Skipping initialization.")
        return
    
    # Add explorer with random or copied init
    policy.add_actor('explorer', random_init=explorer_config.random_init)
    
    # Load checkpoint if provided
    if explorer_config.actor_checkpoint is not None:
        policy.load_actor('explorer', explorer_config.actor_checkpoint)
        logger.info(f"Loaded explorer checkpoint from {explorer_config.actor_checkpoint}")
    
    # Move to device if specified
    if device is not None:
        policy.get_actor('explorer').to(device)
    
    logger.info(f"Initialized Explorer actor (random_init={explorer_config.random_init})")


def setup_explorer_training(
    policy,
    explorer_config: Optional[ExplorerConfig] = None,
) -> None:
    """
    Configure the policy for Explorer training.
    
    This freezes the World Model and regular actor, and makes only
    the Explorer trainable.
    
    Args:
        policy: F1_VLA policy instance
        explorer_config: Explorer configuration
    """
    if explorer_config is None:
        explorer_config = ExplorerConfig()
    
    # Ensure explorer exists
    if 'explorer' not in policy.list_actors():
        initialize_explorer(policy, explorer_config)
    
    # Set explorer as active
    policy.active_actor = 'explorer'
    
    # Configure trainable modules
    # Only explorer should be trainable
    policy.set_trainable_actors(['explorer'])
    
    # Freeze World Model if specified
    if explorer_config.freeze_world_model and hasattr(policy.model, 'paligemma_with_expert'):
        if hasattr(policy.model.paligemma_with_expert, 'gemma_wm_expert'):
            for param in policy.model.paligemma_with_expert.gemma_wm_expert.parameters():
                param.requires_grad = False
            logger.info("Froze World Model parameters")
    
    # Freeze vision encoder
    if hasattr(policy.model, 'paligemma_with_expert'):
        policy.model.paligemma_with_expert.paligemma.vision_tower.eval()
        for param in policy.model.paligemma_with_expert.paligemma.vision_tower.parameters():
            param.requires_grad = False
    
    # Freeze PaliGemma language model
    if hasattr(policy.model, 'paligemma_with_expert'):
        for param in policy.model.paligemma_with_expert.paligemma.language_model.parameters():
            param.requires_grad = False
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Explorer training setup complete:")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params:.2%}")


def get_explorer_parameters(policy) -> list:
    """
    Get the parameters of the Explorer actor for optimization.
    
    Args:
        policy: F1_VLA policy instance
        
    Returns:
        List of trainable parameters from the Explorer
    """
    if 'explorer' not in policy.list_actors():
        raise ValueError("Explorer not initialized. Call initialize_explorer first.")
    
    explorer = policy.get_actor('explorer')
    return list(explorer.parameters())
