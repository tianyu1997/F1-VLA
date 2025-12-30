"""
Explorer Actor RL Training Script

集成训练脚本，整合所有模块：
- VAE Embedding提取 (vae_embedding.py)
- Reward计算 (reward_computation.py)  
- Rollout收集 (explorer_rollout.py)
- Phase 1 RL训练 (explorer_trainer.py)
- Phase 2 对抗训练 (adversarial_trainer.py)

Usage:
    python train_explorer.py --config f1_vla/config/explorer_train_config.yaml
    python train_explorer.py --config f1_vla/config/explorer_train_config.yaml --phase 1
    python train_explorer.py --config f1_vla/config/explorer_train_config.yaml --phase 2 --resume
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import training modules
from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor, EmbeddingBuffer
from f1_vla.src.models.reward_computation import (
    RewardComputer, RewardConfig, RewardBuffer, ExplorerRewardManager
)
from f1_vla.src.models.explorer_rollout import (
    ExplorerRolloutCollector, RolloutConfig, EpisodeBuffer
)
from f1_vla.src.models.explorer_trainer import (
    ExplorerRLTrainer, ExplorerTrainingConfig, PPOValueHead
)
from f1_vla.src.models.adversarial_trainer import (
    AdversarialTrainingManager, AdversarialTrainingConfig
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExplorerTrainConfig:
    """Complete training configuration."""
    
    # Model paths
    pretrained_path: str = ""
    vae_checkpoint: str = ""
    wm_checkpoint: str = ""
    
    # VAE config
    vae_vocab_size: int = 4096
    vae_z_channels: int = 32
    
    # Environment config
    env_type: str = "robotwin"
    image_size: int = 224
    history_length: int = 4
    action_dim: int = 14
    max_episode_steps: int = 200
    
    # Reward config
    reward_alpha: float = 1.0
    reward_beta: float = 1.0
    reward_gamma: float = 0.5
    reward_epsilon: float = 0.1
    reward_delta: float = 0.01
    
    # Phase 1 config
    phase1_enabled: bool = True
    phase1_lr: float = 3e-4
    phase1_gamma: float = 0.99
    phase1_gae_lambda: float = 0.95
    phase1_clip_epsilon: float = 0.2
    phase1_value_coef: float = 0.5
    phase1_entropy_coef: float = 0.01
    phase1_max_grad_norm: float = 0.5
    phase1_total_timesteps: int = 100000
    phase1_steps_per_rollout: int = 256
    phase1_num_epochs: int = 4
    phase1_batch_size: int = 64
    
    # Phase 2 config
    phase2_enabled: bool = True
    phase2_wm_lr: float = 1e-4
    phase2_explorer_lr: float = 1e-4
    phase2_wm_updates: int = 10
    phase2_explorer_updates: int = 1
    phase2_warmup_iterations: int = 100
    phase2_collapse_threshold: float = 0.1
    phase2_total_iterations: int = 1000
    
    # Output config
    output_dir: str = "./outputs/explorer_training"
    log_freq: int = 100
    save_freq: int = 10000
    
    # Hardware config
    device: str = "cuda"
    mixed_precision: bool = True
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExplorerTrainConfig":
        """Load config from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Flatten nested config
        flat_config = {}
        
        # Model config
        if 'model' in config_dict:
            model = config_dict['model']
            flat_config['pretrained_path'] = model.get('pretrained_path', '')
            if 'vae' in model:
                flat_config['vae_checkpoint'] = model['vae'].get('checkpoint_path', '')
                flat_config['vae_vocab_size'] = model['vae'].get('vocab_size', 4096)
                flat_config['vae_z_channels'] = model['vae'].get('z_channels', 32)
            if 'world_model' in model:
                flat_config['wm_checkpoint'] = model['world_model'].get('checkpoint_path', '')
        
        # Environment config
        if 'environment' in config_dict:
            env = config_dict['environment']
            flat_config['env_type'] = env.get('type', 'robotwin')
            if 'observation' in env:
                flat_config['image_size'] = env['observation'].get('image_size', 224)
                flat_config['history_length'] = env['observation'].get('history_length', 4)
            if 'action' in env:
                flat_config['action_dim'] = env['action'].get('dim', 14)
                flat_config['max_episode_steps'] = env['action'].get('max_episode_steps', 200)
        
        # Reward config
        if 'reward' in config_dict:
            reward = config_dict['reward']
            flat_config['reward_alpha'] = reward.get('alpha', 1.0)
            flat_config['reward_beta'] = reward.get('beta', 1.0)
            flat_config['reward_gamma'] = reward.get('gamma', 0.5)
            flat_config['reward_epsilon'] = reward.get('epsilon', 0.1)
            flat_config['reward_delta'] = reward.get('delta', 0.01)
        
        # Phase 1 config
        if 'phase1' in config_dict:
            p1 = config_dict['phase1']
            flat_config['phase1_enabled'] = p1.get('enabled', True)
            if 'ppo' in p1:
                ppo = p1['ppo']
                flat_config['phase1_lr'] = ppo.get('learning_rate', 3e-4)
                flat_config['phase1_gamma'] = ppo.get('gamma', 0.99)
                flat_config['phase1_gae_lambda'] = ppo.get('gae_lambda', 0.95)
                flat_config['phase1_clip_epsilon'] = ppo.get('clip_epsilon', 0.2)
                flat_config['phase1_value_coef'] = ppo.get('value_coef', 0.5)
                flat_config['phase1_entropy_coef'] = ppo.get('entropy_coef', 0.01)
                flat_config['phase1_max_grad_norm'] = ppo.get('max_grad_norm', 0.5)
            if 'training' in p1:
                train = p1['training']
                flat_config['phase1_total_timesteps'] = train.get('total_timesteps', 100000)
                flat_config['phase1_steps_per_rollout'] = train.get('steps_per_rollout', 256)
                flat_config['phase1_num_epochs'] = train.get('num_epochs', 4)
                flat_config['phase1_batch_size'] = train.get('batch_size', 64)
        
        # Phase 2 config
        if 'phase2' in config_dict:
            p2 = config_dict['phase2']
            flat_config['phase2_enabled'] = p2.get('enabled', True)
            if 'adversarial' in p2:
                adv = p2['adversarial']
                flat_config['phase2_wm_lr'] = adv.get('wm_learning_rate', 1e-4)
                flat_config['phase2_explorer_lr'] = adv.get('explorer_learning_rate', 1e-4)
                flat_config['phase2_wm_updates'] = adv.get('wm_updates_per_iter', 10)
                flat_config['phase2_explorer_updates'] = adv.get('explorer_updates_per_iter', 1)
                flat_config['phase2_warmup_iterations'] = adv.get('warmup_iterations', 100)
                flat_config['phase2_collapse_threshold'] = adv.get('collapse_threshold', 0.1)
            if 'training' in p2:
                train = p2['training']
                flat_config['phase2_total_iterations'] = train.get('total_iterations', 1000)
        
        # Logging config
        if 'logging' in config_dict:
            log = config_dict['logging']
            flat_config['output_dir'] = log.get('output_dir', './outputs/explorer_training')
            flat_config['log_freq'] = log.get('log_freq', 100)
        
        # Hardware config
        if 'hardware' in config_dict:
            hw = config_dict['hardware']
            flat_config['device'] = hw.get('device', 'cuda')
            flat_config['mixed_precision'] = hw.get('mixed_precision', True)
        
        flat_config['seed'] = config_dict.get('seed', 42)
        
        return cls(**flat_config)


# =============================================================================
# Training Pipeline
# =============================================================================

class ExplorerTrainingPipeline:
    """
    Complete training pipeline for Explorer actor.
    
    Integrates all components:
    - Model loading (F1-VLA with Explorer actor)
    - VAE embedding extraction
    - Reward computation
    - Rollout collection
    - Phase 1: RL training with frozen WM
    - Phase 2: Adversarial training
    """
    
    def __init__(self, config: ExplorerTrainConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Set random seed
        self._set_seed(config.seed)
        
        # Initialize components (lazy loading)
        self.policy = None
        self.vae = None
        self.env = None
        self.embedding_extractor = None
        self.reward_manager = None
        self.rollout_collector = None
        self.phase1_trainer = None
        self.phase2_trainer = None
        self.tensorboard_writer = None
        
        self.logger.info(f"Training pipeline initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.logger.info(f"Random seed set to {seed}")
    
    def load_models(self):
        """Load F1-VLA policy with Explorer actor and VAE."""
        self.logger.info("Loading models...")
        
        # Load policy (placeholder - actual implementation depends on F1-VLA)
        # TODO: Load actual F1-VLA policy
        self.logger.info(f"  Policy path: {self.config.pretrained_path}")
        
        # For now, create a mock policy for testing
        # In production, replace with actual F1-VLA loading
        self.policy = self._create_mock_policy()
        
        # Load VAE
        self.logger.info(f"  VAE path: {self.config.vae_checkpoint}")
        self.vae = self._load_vae()
        
        # Initialize embedding extractor
        self.embedding_extractor = VAEEmbeddingExtractor(
            vae=self.vae,
            vocab_size=self.config.vae_vocab_size,
            device=self.device
        )
        
        self.logger.info("Models loaded successfully")
    
    def _create_mock_policy(self) -> nn.Module:
        """Create mock policy for testing."""
        # This should be replaced with actual F1-VLA policy loading
        
        class MockPolicy(nn.Module):
            def __init__(self, action_dim, hidden_dim=256):
                super().__init__()
                self.action_dim = action_dim
                self.hidden_dim = hidden_dim
                
                # Mock explorer actor
                self.actors = nn.ModuleDict({
                    'explorer': nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, action_dim)
                    )
                })
                
                # Mock world model (for embedding generation)
                self.world_model = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                
            def forward(self, x, actor_name='explorer'):
                return self.actors[actor_name](x)
            
            def get_actor(self, name):
                return self.actors.get(name)
        
        policy = MockPolicy(self.config.action_dim)
        policy.to(self.device)
        return policy
    
    def _load_vae(self) -> Optional[nn.Module]:
        """Load VAE model."""
        # Placeholder for VAE loading
        # In production, load actual VAE
        
        class MockVAE(nn.Module):
            def __init__(self, embed_dim=1280, vocab_size=4096):
                super().__init__()
                self.embed_dim = embed_dim
                self.vocab_size = vocab_size
                self.codebook = nn.Embedding(vocab_size, embed_dim)
                
            def encode(self, x):
                # Mock encoding: return random embedding
                B = x.shape[0] if len(x.shape) > 1 else 1
                return torch.randn(B, self.embed_dim, device=x.device)
            
            def get_codebook_embedding(self):
                return self.codebook.weight
        
        vae = MockVAE(
            embed_dim=1280,
            vocab_size=self.config.vae_vocab_size
        )
        vae.to(self.device)
        
        # Load checkpoint if exists
        if self.config.vae_checkpoint and os.path.exists(self.config.vae_checkpoint):
            self.logger.info(f"  Loading VAE checkpoint: {self.config.vae_checkpoint}")
            # checkpoint = torch.load(self.config.vae_checkpoint, map_location=self.device)
            # vae.load_state_dict(checkpoint)
        
        return vae
    
    def setup_environment(self):
        """Setup training environment."""
        self.logger.info(f"Setting up environment: {self.config.env_type}")
        
        # Placeholder for environment setup
        # In production, create actual environment
        
        class MockEnvironment:
            def __init__(self, image_size, action_dim, max_steps):
                self.image_size = image_size
                self.action_dim = action_dim
                self.max_steps = max_steps
                self.step_count = 0
                
            def reset(self):
                self.step_count = 0
                obs = {
                    'image': torch.randn(3, self.image_size, self.image_size),
                    'state': torch.randn(32)
                }
                return obs
            
            def step(self, action):
                self.step_count += 1
                obs = {
                    'image': torch.randn(3, self.image_size, self.image_size),
                    'state': torch.randn(32)
                }
                reward = 0.0  # Explorer computes its own reward
                done = self.step_count >= self.max_steps
                info = {}
                return obs, reward, done, info
        
        self.env = MockEnvironment(
            image_size=self.config.image_size,
            action_dim=self.config.action_dim,
            max_steps=self.config.max_episode_steps
        )
        
        self.logger.info("Environment setup complete")
    
    def setup_reward_system(self):
        """Setup reward computation system."""
        self.logger.info("Setting up reward system...")
        
        reward_config = RewardConfig(
            alpha=self.config.reward_alpha,
            beta=self.config.reward_beta,
            gamma=self.config.reward_gamma,
            epsilon=self.config.reward_epsilon,
            delta=self.config.reward_delta
        )
        
        self.reward_manager = ExplorerRewardManager(
            config=reward_config,
            buffer_size=self.config.phase1_steps_per_rollout * 2
        )
        
        self.logger.info(f"  Reward weights: α={reward_config.alpha}, β={reward_config.beta}, "
                        f"γ={reward_config.gamma}, ε={reward_config.epsilon}, δ={reward_config.delta}")
    
    def setup_rollout_collector(self):
        """Setup rollout collector."""
        self.logger.info("Setting up rollout collector...")
        
        rollout_config = RolloutConfig(
            history_length=self.config.history_length,
            max_episode_steps=self.config.max_episode_steps,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda
        )
        
        self.rollout_collector = ExplorerRolloutCollector(
            policy=self.policy,
            env=self.env,
            vae=self.vae,
            reward_manager=self.reward_manager,
            config=rollout_config,
            device=self.device
        )
        
        self.logger.info(f"  History length: {rollout_config.history_length}")
        self.logger.info(f"  Max episode steps: {rollout_config.max_episode_steps}")
    
    def setup_tensorboard(self):
        """Setup TensorBoard writer."""
        if not HAS_TENSORBOARD:
            self.logger.warning("TensorBoard not available, skipping...")
            return
        
        tb_dir = self.output_dir / "tensorboard"
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
        self.logger.info(f"TensorBoard logging to: {tb_dir}")
    
    def run_phase1(self, resume_from: Optional[str] = None):
        """
        Run Phase 1: RL training with frozen World Model.
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        if not self.config.phase1_enabled:
            self.logger.info("Phase 1 disabled, skipping...")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Phase 1: RL Training (Frozen WM)")
        self.logger.info("=" * 60)
        
        # Create training config
        train_config = ExplorerTrainingConfig(
            learning_rate=self.config.phase1_lr,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda,
            clip_epsilon=self.config.phase1_clip_epsilon,
            value_coef=self.config.phase1_value_coef,
            entropy_coef=self.config.phase1_entropy_coef,
            max_grad_norm=self.config.phase1_max_grad_norm,
            num_epochs=self.config.phase1_num_epochs,
            batch_size=self.config.phase1_batch_size,
            total_timesteps=self.config.phase1_total_timesteps,
        )
        
        # Create trainer
        self.phase1_trainer = ExplorerRLTrainer(
            policy=self.policy,
            config=train_config,
            device=self.device
        )
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            self.phase1_trainer.load_checkpoint(resume_from)
        
        # Training loop
        total_steps = 0
        num_updates = 0
        
        while total_steps < self.config.phase1_total_timesteps:
            # Collect rollout
            self.logger.info(f"Collecting rollout (step {total_steps})...")
            transitions = self._collect_rollout(self.config.phase1_steps_per_rollout)
            
            if len(transitions) == 0:
                self.logger.warning("No transitions collected, skipping update")
                continue
            
            # Convert to batch
            batch = self.rollout_collector.transitions_to_batch(transitions)
            
            # Update policy
            metrics = self.phase1_trainer.update(batch)
            
            # Log metrics
            total_steps += len(transitions)
            num_updates += 1
            
            if num_updates % self.config.log_freq == 0:
                self._log_metrics("phase1", metrics, total_steps)
            
            # Save checkpoint
            if total_steps % self.config.save_freq == 0:
                self._save_checkpoint("phase1", total_steps)
        
        # Save final checkpoint
        self._save_checkpoint("phase1", total_steps, is_final=True)
        self.logger.info("Phase 1 training complete!")
    
    def run_phase2(self, resume_from: Optional[str] = None):
        """
        Run Phase 2: Adversarial training (WM vs Explorer).
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        if not self.config.phase2_enabled:
            self.logger.info("Phase 2 disabled, skipping...")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Phase 2: Adversarial Training")
        self.logger.info("=" * 60)
        
        # Create adversarial training config
        adv_config = AdversarialTrainingConfig(
            wm_learning_rate=self.config.phase2_wm_lr,
            explorer_learning_rate=self.config.phase2_explorer_lr,
            wm_updates_per_iteration=self.config.phase2_wm_updates,
            explorer_updates_per_iteration=self.config.phase2_explorer_updates,
            warmup_iterations=self.config.phase2_warmup_iterations,
            collapse_threshold=self.config.phase2_collapse_threshold,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda,
            clip_epsilon=self.config.phase1_clip_epsilon,
            value_coef=self.config.phase1_value_coef,
            entropy_coef=self.config.phase1_entropy_coef,
            max_grad_norm=self.config.phase1_max_grad_norm,
        )
        
        # Create adversarial trainer
        self.phase2_trainer = AdversarialTrainingManager(
            policy=self.policy,
            vae=self.vae,
            config=adv_config,
            device=self.device
        )
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            self.phase2_trainer.load_checkpoint(resume_from)
        
        # Training loop
        for iteration in range(self.config.phase2_total_iterations):
            # Collect rollout
            transitions = self._collect_rollout(self.config.phase1_steps_per_rollout)
            
            if len(transitions) == 0:
                continue
            
            # Convert to batch
            batch = self.rollout_collector.transitions_to_batch(transitions)
            
            # Run adversarial training step
            metrics = self.phase2_trainer.train_step(batch, iteration)
            
            # Log metrics
            if iteration % self.config.log_freq == 0:
                self._log_metrics("phase2", metrics, iteration)
            
            # Save checkpoint
            if iteration % (self.config.save_freq // 10) == 0:
                self._save_checkpoint("phase2", iteration)
        
        # Save final checkpoint
        self._save_checkpoint("phase2", iteration, is_final=True)
        self.logger.info("Phase 2 training complete!")
    
    def _collect_rollout(self, num_steps: int) -> List:
        """Collect rollout from environment."""
        # Simplified rollout collection
        # In production, use proper rollout collector
        
        transitions = []
        obs = self.env.reset()
        
        for _ in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                state = torch.randn(1, 256, device=self.device)  # Mock state
                action = self.policy(state, actor_name='explorer')
                action = action.squeeze(0)
            
            # Step environment
            next_obs, _, done, info = self.env.step(action.cpu().numpy())
            
            # Create mock transition
            transition = {
                'observation': obs,
                'action': action.cpu(),
                'next_observation': next_obs,
                'done': done,
                'value': torch.tensor(0.0),
                'log_prob': torch.tensor(0.0),
                'reward': torch.tensor(0.0),  # Reward computed separately
            }
            transitions.append(transition)
            
            if done:
                obs = self.env.reset()
            else:
                obs = next_obs
        
        return transitions
    
    def _log_metrics(self, phase: str, metrics: Dict[str, float], step: int):
        """Log metrics to console and TensorBoard."""
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[{phase}] Step {step}: {metrics_str}")
        
        # TensorBoard logging
        if self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{phase}/{key}", value, step)
    
    def _save_checkpoint(self, phase: str, step: int, is_final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints" / phase
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_final:
            checkpoint_path = checkpoint_dir / "final.pth"
        else:
            checkpoint_path = checkpoint_dir / f"step_{step}.pth"
        
        if phase == "phase1" and self.phase1_trainer:
            self.phase1_trainer.save_checkpoint(str(checkpoint_path))
        elif phase == "phase2" and self.phase2_trainer:
            self.phase2_trainer.save_checkpoint(str(checkpoint_dir))
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def run(self, phase: Optional[int] = None, resume_from: Optional[str] = None):
        """
        Run complete training pipeline.
        
        Args:
            phase: Specific phase to run (1 or 2), or None for both
            resume_from: Path to checkpoint to resume from
        """
        self.logger.info("=" * 60)
        self.logger.info("Explorer Actor Training Pipeline")
        self.logger.info("=" * 60)
        
        # Load models
        self.load_models()
        
        # Setup components
        self.setup_environment()
        self.setup_reward_system()
        self.setup_rollout_collector()
        self.setup_tensorboard()
        
        # Run training
        if phase is None or phase == 1:
            self.run_phase1(resume_from if phase == 1 else None)
        
        if phase is None or phase == 2:
            # For phase 2, load phase 1 checkpoint if available
            phase1_checkpoint = None
            if phase == 2 and resume_from:
                phase1_checkpoint = resume_from
            elif self.config.phase1_enabled:
                phase1_final = self.output_dir / "checkpoints/phase1/final.pth"
                if phase1_final.exists():
                    phase1_checkpoint = str(phase1_final)
            
            self.run_phase2(phase1_checkpoint if phase == 2 else None)
        
        # Cleanup
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        self.logger.info("Training complete!")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Explorer Actor RL Training")
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML file"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help="Specific phase to run (1 or 2). Default: run both"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = ExplorerTrainConfig.from_yaml(args.config)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    
    # Create and run training pipeline
    pipeline = ExplorerTrainingPipeline(config)
    pipeline.run(phase=args.phase, resume_from=args.resume)


if __name__ == "__main__":
    main()
