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

# Add project root to path (F1-VLA directory)
project_root = Path(__file__).parent.parent.parent.parent  # scripts -> src -> f1_vla -> F1-VLA
sys.path.insert(0, str(project_root))

# Import training modules
from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor, EmbeddingBuffer
from f1_vla.src.models.reward_computation import (
    RewardComputer, RewardConfig, RewardBuffer, ExplorerRewardManager
)
from f1_vla.src.models.explorer_rollout import (
    ExplorerRolloutCollector, RolloutConfig, EpisodeBuffer, transitions_to_batch
)
from f1_vla.src.models.explorer_trainer import (
    ExplorerRLTrainer, ExplorerTrainingConfig, PPOValueHead
)
from f1_vla.src.models.adversarial_trainer import (
    AdversarialTrainingManager, AdversarialTrainingConfig
)

# Import integration module for actual F1-VLA and RoboTwin
from f1_vla.src.models.f1_integration import (
    load_f1_vla_policy,
    load_vae,
    create_robotwin_env,
    create_mock_env,
    ExplorerEnvWrapper,
    create_explorer_training_env,
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
    
    # Dataset config (for sequential data loading)
    data_dirs: List[str] = None  # List of directories containing episode_*.pt files
    task_descriptions: List[str] = None
    n_obs_img_steps: int = 4
    n_pred_img_steps: int = 1
    chunk_size: int = 4
    
    # Environment config (deprecated - use dataset instead)
    env_type: str = "dataset"  # Changed from "robotwin" to "dataset"
    image_size: int = 256  # Changed from 224 to 256 for VAE compatibility
    history_length: int = 4
    action_dim: int = 7  # Standard robot action dimension
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
    phase2_batch_size: int = 2  # Small batch for adversarial training to avoid OOM
    
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
            flat_config['env_type'] = env.get('type', 'dataset')
            if 'observation' in env:
                flat_config['image_size'] = env['observation'].get('image_size', 256)
                flat_config['history_length'] = env['observation'].get('history_length', 4)
            if 'action' in env:
                flat_config['action_dim'] = env['action'].get('dim', 7)
                flat_config['max_episode_steps'] = env['action'].get('max_episode_steps', 200)
        
        # Dataset config
        if 'dataset' in config_dict:
            ds = config_dict['dataset']
            flat_config['data_dirs'] = ds.get('data_dirs', [])
            flat_config['task_descriptions'] = ds.get('task_descriptions', None)
            flat_config['n_obs_img_steps'] = ds.get('n_obs_img_steps', 4)
            flat_config['n_pred_img_steps'] = ds.get('n_pred_img_steps', 1)
            flat_config['chunk_size'] = ds.get('chunk_size', 4)
        
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
                flat_config['phase2_batch_size'] = train.get('batch_size', 2)  # Small default
        
        # Dataset config - also check for batch_size
        if 'dataset' in config_dict:
            ds = config_dict['dataset']
            if 'batch_size' in ds:
                flat_config['phase2_batch_size'] = ds.get('batch_size', 2)
        
        # Logging config (can be either 'logging' or 'output')
        if 'logging' in config_dict:
            log = config_dict['logging']
            flat_config['output_dir'] = log.get('output_dir', './outputs/explorer_training')
            flat_config['log_freq'] = log.get('log_freq', 100)
            flat_config['save_freq'] = log.get('save_freq', 10000)
        elif 'output' in config_dict:
            out = config_dict['output']
            flat_config['output_dir'] = out.get('dir', './outputs/explorer_training')
            flat_config['log_freq'] = out.get('log_freq', 100)
            flat_config['save_freq'] = out.get('save_freq', 10000)
        
        # Hardware config
        if 'hardware' in config_dict:
            hw = config_dict['hardware']
            flat_config['device'] = hw.get('device', 'cuda')
            flat_config['mixed_precision'] = hw.get('mixed_precision', True)
            flat_config['seed'] = hw.get('seed', 42)
        
        flat_config['seed'] = config_dict.get('seed', flat_config.get('seed', 42))
        
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
        self.explorer_env = None  # ExplorerEnvWrapper
        self.vae_extractor = None
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
        self.logger.info(f"  Policy path: {self.config.pretrained_path}")
        self.logger.info(f"  VAE path: {self.config.vae_checkpoint}")
        
        # Try to load actual F1-VLA policy
        use_mock = False
        
        if self.config.pretrained_path and os.path.exists(self.config.pretrained_path):
            try:
                self.policy, self.vae = load_f1_vla_policy(
                    config_path=self.config.pretrained_path,
                    checkpoint_path=self.config.pretrained_path,
                    vae_path=self.config.vae_checkpoint,
                    device=str(self.device),
                    add_explorer=True,
                    train_act_expert_only=True,  # Only train explorer actor
                )
                self.logger.info("  Loaded actual F1-VLA policy")
                self.logger.info(f"  Available actors: {self.policy.list_actors()}")
            except Exception as e:
                import traceback
                self.logger.warning(f"  Failed to load F1-VLA: {e}")
                self.logger.warning(f"  Traceback: {traceback.format_exc()}")
                use_mock = True
        else:
            self.logger.info("  No pretrained path provided, using mock policy")
            use_mock = True
        
        if use_mock:
            self.logger.info("  Using mock policy and VAE for testing")
            self.policy = self._create_mock_policy()
            self.vae = self._load_mock_vae()
        
        # Initialize embedding extractor (use vae_extractor for consistency with rest of code)
        self.vae_extractor = VAEEmbeddingExtractor(
            vae=self.vae,
            embedding_dim=self.config.vae_z_channels,
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
                self.active_actor = 'explorer'
                
                # Mock state projection
                self.model = nn.Module()
                self.model.state_proj = nn.Linear(14, hidden_dim)  # state_dim -> hidden_dim
                
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
                if name in self.actors:
                    return self.actors[name]
                return None
            
            def list_actors(self):
                return list(self.actors.keys())
            
            def set_trainable_actors(self, actor_names):
                # For mock, just pass
                pass
            
            def forward_with_actor(self, batch, actor_name='explorer', **kwargs):
                """Forward pass with specific actor."""
                if 'state_emb' in batch:
                    state_emb = batch['state_emb']
                elif 'observation.state' in batch:
                    state = batch['observation.state']
                    state_emb = self.model.state_proj(state)
                else:
                    raise ValueError(f"batch missing 'state_emb' or 'observation.state'. Got keys: {list(batch.keys())}")
                
                action = self.actors[actor_name](state_emb)
                return {'action': action, 'state_emb': state_emb}
        
        policy = MockPolicy(self.config.action_dim)
        policy.to(self.device)
        return policy
    
    def _load_mock_vae(self) -> Optional[nn.Module]:
        """Load mock VAE model for testing."""
        
        class MockVAE(nn.Module):
            def __init__(self, embed_dim=1280, vocab_size=4096, z_channels=32):
                super().__init__()
                self.embed_dim = embed_dim
                self.vocab_size = vocab_size
                self.z_channels = z_channels
                self.codebook = nn.Embedding(vocab_size, embed_dim)
                
                # Mock encoder and quant_conv for VAEEmbeddingExtractor compatibility
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128, z_channels, 4, stride=2, padding=1),
                )
                self.quant_conv = nn.Conv2d(z_channels, z_channels, 1)
                
            def encode(self, x):
                # Mock encoding: return random embedding
                B = x.shape[0] if len(x.shape) > 1 else 1
                return torch.randn(B, self.embed_dim, device=x.device)
            
            def get_codebook_embedding(self):
                return self.codebook.weight
        
        vae = MockVAE(
            embed_dim=1280,
            vocab_size=self.config.vae_vocab_size,
            z_channels=self.config.vae_z_channels
        )
        vae.to(self.device)
        
        # Load checkpoint if exists
        if self.config.vae_checkpoint and os.path.exists(self.config.vae_checkpoint):
            self.logger.info(f"  Loading VAE checkpoint: {self.config.vae_checkpoint}")
            # Note: mock VAE won't match actual checkpoint structure
        
        return vae
    
    def setup_environment(self):
        """Setup training environment and data source."""
        self.logger.info(f"Setting up environment: {self.config.env_type}")
        
        if self.config.env_type == "dataset":
            # Offline RL: Use sequential dataset for training data
            self._setup_sequential_dataset()
        elif self.config.env_type in ["robotwin", "mock"]:
            # Online RL: Use environment interaction with sequential buffer
            self._setup_environment_with_sequential_buffer()
        else:
            raise ValueError(
                f"Unknown env_type: {self.config.env_type}. "
                f"Supported: 'dataset' (offline RL), 'robotwin' (online RL), 'mock' (testing)"
            )
        
        self.logger.info("Environment setup complete")
    
    def _setup_environment_with_sequential_buffer(self):
        """Setup environment for online RL with sequential rollout buffer."""
        from f1_vla.src.models.sequential_rollout_buffer import (
            SequentialRolloutBuffer, SequentialRolloutConfig, SequentialRolloutCollector
        )
        
        # Create environment
        if self.config.env_type == "robotwin":
            self.logger.info("  Creating RoboTwin environment...")
            self.env = create_robotwin_env(
                image_size=self.config.image_size,
            )
        else:
            self.logger.info("  Creating mock environment for testing...")
            # Convert image_size to tuple if needed
            img_size = (self.config.image_size, self.config.image_size) if isinstance(self.config.image_size, int) else self.config.image_size
            self.env = create_mock_env(
                image_size=img_size,
                action_dim=self.config.action_dim,
            )
        
        # Initialize VAE extractor if not already done
        if self.vae_extractor is None:
            from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
            self.vae_extractor = VAEEmbeddingExtractor(
                vae=self.vae,
                embedding_dim=self.config.vae_z_channels,
            )
        
        # Initialize reward manager if not already done
        if self.reward_manager is None:
            from f1_vla.src.models.reward_computation import (
                RewardComputer, RewardConfig, RewardBuffer, ExplorerRewardManager
            )
            reward_config = RewardConfig(
                uncertainty_weight=self.config.reward_alpha,
                mse_weight=self.config.reward_beta,
                mse_improvement_weight=self.config.reward_gamma,
                uncertainty_improvement_weight=self.config.reward_epsilon,
                action_penalty_weight=self.config.reward_delta,
            )
            self.reward_manager = ExplorerRewardManager(config=reward_config)
        
        # Create sequential rollout buffer
        buffer_config = SequentialRolloutConfig(
            max_episodes=100,
            max_steps_total=50000,
            n_obs_img_steps=self.config.n_obs_img_steps,
            n_pred_img_steps=self.config.n_pred_img_steps,
            chunk_size=self.config.chunk_size,
            image_size=self.config.image_size,
            camera_keys=["head_rgb", "wrist_rgb"],
            wm_camera_key="head_rgb",
            state_dim=14,
            action_dim=self.config.action_dim,
        )
        self.rollout_buffer = SequentialRolloutBuffer(config=buffer_config)
        
        # Create rollout collector
        self.sequential_collector = SequentialRolloutCollector(
            policy=self.policy,
            vae_extractor=self.vae_extractor,
            reward_manager=self.reward_manager,
            buffer=self.rollout_buffer,
            config=buffer_config,
            device=self.device,
        )
        
        self.logger.info(f"  Buffer config: max_episodes={buffer_config.max_episodes}, "
                        f"n_obs_img_steps={buffer_config.n_obs_img_steps}")
        self.logger.info(f"  Environment action_dim={self.config.action_dim}")
    
    def _setup_sequential_dataset(self):
        """Setup sequential dataset for training."""
        from f1_vla.src.processors.data_processors.sequential_dataset import (
            SequentialMEKVMDataset, SequentialBatchSampler, SequentialCollateFn
        )
        
        # Validate data_dirs
        if not self.config.data_dirs:
            raise ValueError(
                "data_dirs must be specified in config for Explorer training. "
                "Example: data_dirs: ['/path/to/episodes/']"
            )
        
        # Check if directories exist
        valid_dirs = []
        for data_dir in self.config.data_dirs:
            if os.path.exists(data_dir):
                valid_dirs.append(data_dir)
            else:
                self.logger.warning(f"Data directory not found: {data_dir}")
        
        if not valid_dirs:
            raise ValueError(
                f"No valid data directories found. Checked: {self.config.data_dirs}"
            )
        
        self.logger.info(f"  Using data directories: {valid_dirs}")
        
        # Create sequential dataset
        self.train_dataset = SequentialMEKVMDataset(
            data_dirs=valid_dirs,
            dataset_idx=0,
            n_obs_img_steps=self.config.n_obs_img_steps,
            n_pred_img_steps=self.config.n_pred_img_steps,
            chunk_size=self.config.chunk_size,
            task_descriptions=self.config.task_descriptions,
            rank=0,
            world_size=1,
            camera_config={
                "und_camera_keys": ["head_rgb", "wrist_rgb"],
                "wm_camera_key": "head_rgb",
            },
        )
        
        # Create batch sampler
        self.train_sampler = SequentialBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.config.phase1_batch_size,
            shuffle_episodes=True,
            drop_last=True,
        )
        
        # Create collate function
        self.collate_fn = SequentialCollateFn(
            max_state_dim=32,  # Will be padded
            max_action_dim=32,
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=0,  # Sequential access, no workers needed
            pin_memory=True,
        )
        
        self.logger.info(f"  Dataset size: {len(self.train_dataset)} samples")
        self.logger.info(f"  Num episodes: {self.train_dataset.get_num_episodes()}")
    
    def setup_reward_system(self):
        """Setup reward computation system."""
        self.logger.info("Setting up reward system...")
        
        reward_config = RewardConfig(
            uncertainty_weight=self.config.reward_alpha,
            mse_weight=self.config.reward_beta,
            mse_improvement_weight=self.config.reward_gamma,
            uncertainty_improvement_weight=self.config.reward_epsilon,
            action_penalty_weight=self.config.reward_delta
        )
        
        self.reward_manager = ExplorerRewardManager(config=reward_config)
        
        self.logger.info(f"  Reward weights: α={reward_config.uncertainty_weight}, "
                        f"β={reward_config.mse_weight}, γ={reward_config.mse_improvement_weight}, "
                        f"ε={reward_config.uncertainty_improvement_weight}, "
                        f"δ={reward_config.action_penalty_weight}")
    
    def setup_rollout_collector(self):
        """Setup rollout collector."""
        self.logger.info("Setting up rollout collector...")
        
        rollout_config = RolloutConfig(
            history_length=self.config.history_length,
            max_steps_per_episode=self.config.max_episode_steps,
        )
        
        # Create VAE embedding extractor
        from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
        self.vae_extractor = VAEEmbeddingExtractor(
            vae=self.vae,
            embedding_dim=self.config.vae_z_channels,
        )
        
        self.rollout_collector = ExplorerRolloutCollector(
            policy=self.policy,
            vae_extractor=self.vae_extractor,
            reward_manager=self.reward_manager,
            config=rollout_config,
            device=self.device
        )
        
        self.logger.info(f"  History length: {rollout_config.history_length}")
        self.logger.info(f"  Max episode steps: {rollout_config.max_steps_per_episode}")
    
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
        
        Supports two modes:
        1. Online RL (env_type='robotwin' or 'mock'): 
           - Explorer interacts with environment
           - Collects rollouts in sequential format
           - Updates policy with PPO
           
        2. Offline RL (env_type='dataset'):
           - Uses pre-collected dataset
           - Computes WM uncertainty as reward signal
           - Updates policy with PPO
        
        Args:
            resume_from: Path to checkpoint to resume from
        """
        if not self.config.phase1_enabled:
            self.logger.info("Phase 1 disabled, skipping...")
            return
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Phase 1: RL Training (Frozen WM)")
        self.logger.info("=" * 60)
        
        # Determine training mode
        if self.config.env_type in ["robotwin", "mock"]:
            self._run_phase1_online_rl(resume_from)
        else:
            self._run_phase1_offline_rl(resume_from)
    
    def _run_phase1_online_rl(self, resume_from: Optional[str] = None):
        """
        Phase 1 with online RL: Environment interaction with sequential rollout buffer.
        
        Training loop:
        1. Collect rollouts from environment (Explorer interacts)
        2. Store in sequential buffer
        3. Sample batches and update policy with PPO
        """
        self.logger.info("Running online RL mode (environment interaction)")
        
        # Validate environment is setup
        if not hasattr(self, 'env') or self.env is None:
            raise RuntimeError(
                "Environment not initialized. "
                "Call setup_environment() with env_type='robotwin' or 'mock'."
            )
        
        # Create training config
        train_config = ExplorerTrainingConfig(
            learning_rate=self.config.phase1_lr,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda,
            clip_epsilon=self.config.phase1_clip_epsilon,
            value_loss_coef=self.config.phase1_value_coef,
            entropy_coef=self.config.phase1_entropy_coef,
            max_grad_norm=self.config.phase1_max_grad_norm,
            ppo_epochs=self.config.phase1_num_epochs,
            mini_batch_size=self.config.phase1_batch_size,
            num_episodes=100,  # Will update dynamically
        )
        
        # Create trainer
        self.phase1_trainer = ExplorerRLTrainer(
            policy=self.policy,
            vae_extractor=self.vae_extractor,
            reward_manager=self.reward_manager,
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
        rollout_steps = self.config.phase1_steps_per_rollout
        
        # Exploration schedule
        epsilon_start = 0.3
        epsilon_end = 0.05
        epsilon_decay_steps = self.config.phase1_total_timesteps // 2
        
        while total_steps < self.config.phase1_total_timesteps:
            # Compute exploration epsilon
            epsilon = max(
                epsilon_end,
                epsilon_start - (epsilon_start - epsilon_end) * total_steps / epsilon_decay_steps
            )
            
            # Collect rollouts
            self.logger.info(f"Collecting rollouts (step {total_steps}, epsilon={epsilon:.3f})...")
            collect_stats = self.sequential_collector.collect_rollouts(
                env=self.env,
                num_steps=rollout_steps,
                use_explorer=True,
                epsilon=epsilon,
            )
            
            total_steps += collect_stats['steps_collected']
            
            self.logger.info(f"  Collected {collect_stats['steps_collected']} steps, "
                           f"mean_reward={collect_stats['mean_reward']:.3f}")
            
            # Check if buffer has enough samples
            if len(self.rollout_buffer) < self.config.phase1_batch_size:
                self.logger.info(f"  Buffer has {len(self.rollout_buffer)} samples, need more...")
                continue
            
            # PPO updates
            for ppo_epoch in range(self.config.phase1_num_epochs):
                # Sample batch from buffer
                batch = self.rollout_buffer.sample_batch(
                    batch_size=self.config.phase1_batch_size,
                    sequential=True,
                )
                
                # Move to device
                batch = self._move_batch_to_device(batch)
                
                # Build PPO batch
                ppo_batch = self._build_ppo_batch_from_buffer(batch)
                
                # Update policy
                metrics = self.phase1_trainer.train_step(ppo_batch)
                num_updates += 1
                
                if num_updates % self.config.log_freq == 0:
                    metrics['epsilon'] = epsilon
                    metrics['buffer_size'] = len(self.rollout_buffer)
                    self._log_metrics("phase1", metrics, total_steps)
            
            # Save checkpoint
            if total_steps % self.config.save_freq == 0:
                self._save_checkpoint("phase1", total_steps)
            
            # Log buffer statistics
            buf_stats = self.rollout_buffer.get_statistics()
            self.logger.info(f"  Buffer: {buf_stats['total_episodes']} episodes, "
                           f"{buf_stats['buffer_samples']} samples, "
                           f"mean_reward={buf_stats['mean_reward']:.3f}")
        
        # Save final checkpoint
        self._save_checkpoint("phase1", total_steps, is_final=True)
        self.logger.info("Phase 1 (online RL) training complete!")
    
    def _run_phase1_offline_rl(self, resume_from: Optional[str] = None):
        """
        Phase 1 with offline RL: Use pre-collected dataset.
        
        Training loop:
        1. Collect multiple batches to form rollout buffer
        2. Compute WM uncertainty as reward signal
        3. Update policy with PPO (multiple epochs over collected data)
        """
        self.logger.info("Running offline RL mode (dataset)")
        
        # Validate dataset is setup
        if not hasattr(self, 'train_dataloader') or self.train_dataloader is None:
            raise RuntimeError(
                "Training dataloader not initialized. "
                "Call setup_environment() with env_type='dataset'."
            )
        
        # Create training config
        train_config = ExplorerTrainingConfig(
            learning_rate=self.config.phase1_lr,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda,
            clip_epsilon=self.config.phase1_clip_epsilon,
            value_loss_coef=self.config.phase1_value_coef,
            entropy_coef=self.config.phase1_entropy_coef,
            max_grad_norm=self.config.phase1_max_grad_norm,
            ppo_epochs=self.config.phase1_num_epochs,
            mini_batch_size=self.config.phase1_batch_size,
            num_episodes=self.train_dataset.get_num_episodes(),
        )
        
        # Create trainer
        self.phase1_trainer = ExplorerRLTrainer(
            policy=self.policy,
            vae_extractor=self.vae_extractor,
            reward_manager=self.reward_manager,
            config=train_config,
            device=self.device
        )
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            self.phase1_trainer.load_checkpoint(resume_from)
        
        # PPO training: collect rollout buffer, then update for multiple epochs
        total_steps = 0
        num_updates = 0
        rollout_iteration = 0
        
        # Use steps_per_rollout from config (how many samples to collect before updating)
        rollout_buffer_size = self.config.phase1_steps_per_rollout
        ppo_epochs = self.config.phase1_num_epochs  # Number of epochs per rollout
        
        while total_steps < self.config.phase1_total_timesteps:
            rollout_iteration += 1
            self.logger.info(f"Rollout {rollout_iteration}, total_steps={total_steps}")
            
            # Phase 1: Collect rollout buffer (with old policy)
            rollout_buffer = []
            buffer_steps = 0
            
            data_iter = iter(self.train_dataloader)
            while buffer_steps < rollout_buffer_size:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_dataloader)
                    batch = next(data_iter)
                
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Compute WM prediction and uncertainty for reward
                batch = self._compute_wm_rewards(batch)
                
                # Build PPO batch from sequential data (old_log_probs computed here with current policy)
                ppo_batch = self._build_ppo_batch_from_dataset(batch)
                
                # Detach old_log_probs to prevent gradient flow
                ppo_batch['old_log_probs'] = ppo_batch['old_log_probs'].detach()
                ppo_batch['old_values'] = ppo_batch['old_values'].detach()
                
                rollout_buffer.append(ppo_batch)
                buffer_steps += batch['observation.state'].shape[0]
                
                # Clear intermediate tensors and cache to prevent OOM
                del batch
                torch.cuda.empty_cache()
            
            # Concatenate all batches in rollout buffer
            combined_batch = self._concatenate_ppo_batches(rollout_buffer)
            
            # Free rollout buffer memory
            del rollout_buffer
            torch.cuda.empty_cache()
            
            self.logger.info(f"Collected {buffer_steps} samples, running {ppo_epochs} PPO epochs")
            
            # Phase 2: Update policy for multiple epochs over collected data
            for ppo_epoch in range(ppo_epochs):
                metrics = self.phase1_trainer.train_step(combined_batch)
                num_updates += 1
                
                if ppo_epoch == ppo_epochs - 1:  # Log at last epoch
                    progress_pct = 100.0 * total_steps / self.config.phase1_total_timesteps
                    self.logger.info(
                        f"[phase1] Step {total_steps}/{self.config.phase1_total_timesteps} ({progress_pct:.1f}%) | "
                        f"π_loss: {metrics.get('policy_loss', 0):.4f} | "
                        f"v_loss: {metrics.get('value_loss', 0):.2f} | "
                        f"H: {metrics.get('entropy', 0):.3f} | "
                        f"ratio: {metrics.get('ratio', 1):.3f} | "
                        f"clip%: {metrics.get('clip_fraction', 0)*100:.1f} | "
                        f"adv: {metrics.get('advantage_mean', 0):.2f} | "
                        f"std: {metrics.get('std', 0):.3f}"
                    )
                    if self.tensorboard_writer:
                        for k, v in metrics.items():
                            self.tensorboard_writer.add_scalar(f"phase1/{k}", v, total_steps)
            
            total_steps += buffer_steps
            
            # Save checkpoint
            if total_steps % self.config.save_freq == 0:
                self._save_checkpoint("phase1", total_steps)
        
        # Save final checkpoint
        self._save_checkpoint("phase1", total_steps, is_final=True)
        self.logger.info("Phase 1 (offline RL) training complete!")
    
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
        
        # Check if we should use offline mode (dataset) or online mode (env)
        if self.config.env_type == 'dataset':
            self._run_phase2_offline(resume_from)
        else:
            self._run_phase2_online(resume_from)
    
    def _run_phase2_offline(self, resume_from: Optional[str] = None):
        """
        Run Phase 2 in offline mode using dataset.
        
        WM and Explorer are trained adversarially using dataset batches.
        """
        self.logger.info("Running Phase 2 in offline mode (dataset)")
        
        # Create adversarial training config
        adv_config = AdversarialTrainingConfig(
            wm_learning_rate=self.config.phase2_wm_lr,
            explorer_learning_rate=self.config.phase2_explorer_lr,
            wm_updates_per_iter=self.config.phase2_wm_updates,
            explorer_updates_per_iter=self.config.phase2_explorer_updates,
            warmup_iterations=self.config.phase2_warmup_iterations,
            wm_loss_threshold=self.config.phase2_collapse_threshold,
            gamma=self.config.phase1_gamma,
            gae_lambda=self.config.phase1_gae_lambda,
            clip_epsilon=self.config.phase1_clip_epsilon,
            value_loss_coef=self.config.phase1_value_coef,
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
        
        # Create dedicated dataloader with smaller batch size for Phase 2
        from torch.utils.data import DataLoader
        from f1_vla.src.processors.data_processors.sequential_dataset import SequentialBatchSampler
        
        phase2_sampler = SequentialBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.config.phase2_batch_size,  # Smaller batch size
            shuffle_episodes=True,
            drop_last=True,
        )
        phase2_dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=phase2_sampler,
            collate_fn=self.collate_fn,
            num_workers=0,  # Single worker to avoid issues
            pin_memory=True,
        )
        
        # Resume from checkpoint if provided
        if resume_from and os.path.exists(resume_from):
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            self.phase2_trainer.load_checkpoint(resume_from)
        
        # Training loop using dataset
        iteration = 0
        data_iter = iter(phase2_dataloader)
        
        while iteration < self.config.phase2_total_iterations:
            # Get batch from dataset
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(phase2_dataloader)
                batch = next(data_iter)
            
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Run adversarial training step
            metrics = self.phase2_trainer.train_step_offline(batch, iteration)
            
            iteration += 1
            
            # Log metrics
            if iteration % self.config.log_freq == 0:
                progress_pct = 100.0 * iteration / self.config.phase2_total_iterations
                self.logger.info(
                    f"[phase2] Iter {iteration}/{self.config.phase2_total_iterations} ({progress_pct:.1f}%) | "
                    f"wm_loss: {metrics.get('wm_loss', 0):.4f} | "
                    f"exp_π_loss: {metrics.get('explorer_policy_loss', 0):.4f} | "
                    f"exp_v_loss: {metrics.get('explorer_value_loss', 0):.2f} | "
                    f"adv_reward: {metrics.get('adversarial_reward', 0):.4f}"
                )
                if self.tensorboard_writer:
                    for k, v in metrics.items():
                        self.tensorboard_writer.add_scalar(f"phase2/{k}", v, iteration)
            
            # Save checkpoint
            if iteration % self.config.save_freq == 0:
                self._save_checkpoint("phase2", iteration)
        
        # Save final checkpoint
        self._save_checkpoint("phase2", iteration, is_final=True)
        self.logger.info("Phase 2 (offline adversarial) training complete!")
    
    def _run_phase2_online(self, resume_from: Optional[str] = None):
        """
        Run Phase 2 in online mode using environment rollouts.
        """
        self.logger.info("Running Phase 2 in online mode (environment)")
        
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
        """Collect rollout from environment using ExplorerEnvWrapper."""
        transitions = []
        
        # Reset explorer environment
        obs, info = self.explorer_env.reset()
        
        for step in range(num_steps):
            # Get action from Explorer policy
            with torch.no_grad():
                # Prepare input for Explorer
                # Use gt_img_emb as the main feature
                if 'gt_img_emb' in obs:
                    state_feat = obs['gt_img_emb'].flatten().unsqueeze(0)
                else:
                    state_feat = torch.randn(1, 256, device=self.device)
                
                # Build batch for forward_with_actor
                # Get proj_width from policy config
                proj_width = getattr(self.policy.model.config, 'proj_width', 2048)
                
                # Pad or truncate state_feat to proj_width
                if state_feat.numel() < proj_width:
                    state_emb = torch.zeros(1, proj_width, device=self.device)
                    state_emb[0, :state_feat.numel()] = state_feat.flatten()
                else:
                    state_emb = state_feat.flatten()[:proj_width].unsqueeze(0)
                
                batch = {'state_emb': state_emb}
                
                # Get action from policy using forward_with_actor
                output = self.policy.forward_with_actor(
                    batch,
                    actor_name='explorer',
                    return_action_stats=True,
                )
                
                action_mean = output['action']  # (1, action_dim) should be (1, 7)
                
                # Sample from distribution using learned log_std
                if hasattr(self, 'phase1_trainer') and self.phase1_trainer is not None:
                    log_std = self.phase1_trainer.log_std
                else:
                    # Fallback to fixed std if trainer not initialized
                    log_std = torch.zeros(action_mean.shape[-1], device=self.device) - 1.0  # std ≈ 0.37
                
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(action_mean, std)
                action = dist.rsample().squeeze(0)  # (action_dim,) should be (7,)
                
                # Compute log probability
                log_prob = dist.log_prob(action.unsqueeze(0)).sum(dim=-1).squeeze(0)
                
                # Get value estimate
                state_emb_out = output.get('state_emb')
                if state_emb_out is not None and hasattr(self, 'phase1_trainer') and self.phase1_trainer is not None:
                    value = self.phase1_trainer.value_head(state_emb_out).squeeze()
                else:
                    value = torch.tensor(0.0, device=self.device)
            
            # Step environment
            next_obs, reward_info, terminated, truncated, step_info = self.explorer_env.step(action)
            done = terminated or truncated
            
            # Compute reward using reward manager
            if self.reward_manager is not None:
                reward, reward_components = self.reward_manager.step(
                    pred_emb=reward_info['pred_emb'],
                    gt_emb=reward_info['gt_emb'],
                    uncertainty=reward_info['uncertainty'],
                    action=reward_info['action'],
                )
                if reward is None:
                    reward = torch.tensor(0.0, device=self.device)
            else:
                reward = torch.tensor(0.0, device=self.device)
            
            # Create transition
            transition = {
                'observation': obs,
                'action': action.cpu(),
                'next_observation': next_obs,
                'done': done,
                'value': value.cpu(),
                'log_prob': log_prob.cpu(),
                'reward': reward.cpu() if isinstance(reward, torch.Tensor) else torch.tensor(reward),
                'reward_info': reward_info,  # Store for delayed reward computation
            }
            transitions.append(transition)
            
            if done:
                obs, _ = self.explorer_env.reset()
            else:
                obs = next_obs
        
        return transitions
    
    def _build_ppo_batch(self, transitions: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Build PPO training batch from transitions.
        
        Args:
            transitions: List of transition dicts
            
        Returns:
            Batch dict with tensors for PPO training
        """
        # Extract data
        actions = torch.stack([t['action'] for t in transitions]).to(self.device)
        old_log_probs = torch.tensor([t['log_prob'].item() if isinstance(t['log_prob'], torch.Tensor) else t['log_prob'] 
                                      for t in transitions], device=self.device)
        values = [t['value'].item() if isinstance(t['value'], torch.Tensor) else t['value'] for t in transitions]
        rewards = [t['reward'].item() if isinstance(t['reward'], torch.Tensor) else t['reward'] for t in transitions]
        dones = [t['done'] for t in transitions]
        
        # Extract state embeddings from observations if available
        state_embs = []
        for t in transitions:
            obs = t.get('observation', {})
            if isinstance(obs, dict):
                if 'gt_img_emb' in obs:
                    # Use the image embedding as state representation
                    emb = obs['gt_img_emb']
                    if isinstance(emb, torch.Tensor):
                        state_embs.append(emb.flatten())
                    else:
                        state_embs.append(torch.tensor(emb, device=self.device).flatten())
                elif 'state' in obs:
                    state = obs['state']
                    if isinstance(state, torch.Tensor):
                        state_embs.append(state)
                    else:
                        state_embs.append(torch.tensor(state, device=self.device))
                else:
                    raise ValueError(
                        f"Transition observation missing 'gt_img_emb' or 'state'. "
                        f"Got keys: {list(obs.keys()) if isinstance(obs, dict) else type(obs)}"
                    )
            else:
                raise ValueError(
                    f"Transition observation must be a dict, got: {type(obs)}"
                )
        
        # Stack state embeddings - ensure consistent size
        if state_embs:
            # Pad/truncate to proj_width - get from policy if available
            if hasattr(self, 'policy') and hasattr(self.policy, 'model') and hasattr(self.policy.model, 'config'):
                proj_width = getattr(self.policy.model.config, 'proj_width', 2048)
            else:
                proj_width = 2048
            padded_embs = []
            for emb in state_embs:
                if emb.numel() < proj_width:
                    # Pad with zeros
                    padded = torch.zeros(proj_width, device=self.device)
                    padded[:emb.numel()] = emb.flatten()
                    padded_embs.append(padded)
                else:
                    # Truncate
                    padded_embs.append(emb.flatten()[:proj_width])
            state_emb_tensor = torch.stack(padded_embs)
        else:
            raise ValueError(
                f"No state embeddings collected from transitions. "
                f"Transitions count: {len(transitions)}"
            )
        
        # Normalize rewards
        if len(rewards) > 1:
            rewards_tensor = torch.tensor(rewards)
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std() + 1e-8
            rewards = ((rewards_tensor - rewards_mean) / rewards_std).tolist()
        
        # Compute GAE
        advantages = []
        gae = 0.0
        gamma = self.config.phase1_gamma
        gae_lambda = self.config.phase1_gae_lambda
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t])) - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        old_values = torch.tensor(values, device=self.device)
        
        batch = {
            'actions': actions,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': advantages,
            'returns': returns,
            'state_emb': state_emb_tensor,  # Add state embeddings for forward_with_actor
        }
        
        return batch
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], torch.Tensor):
                result[key] = [v.to(self.device) for v in value]
            else:
                result[key] = value
        return result
    
    def _compute_wm_rewards(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute World Model prediction and uncertainty for reward calculation.
        
        The Explorer is rewarded for finding states where WM has high prediction error.
        
        Args:
            batch: Sequential batch from dataset with images and states
            
        Returns:
            batch with added reward-related fields
        """
        with torch.no_grad():
            B = batch['observation.state'].shape[0]
            
            # Get WM prediction using the policy's world model
            # Use the image history for WM input
            wm_history = batch.get('observation.images.image0_history')  # (B, T+pred, C, H, W)
            
            if wm_history is None:
                raise ValueError(
                    "batch missing 'observation.images.image0_history' for WM prediction. "
                    f"Got keys: {list(batch.keys())}"
                )
            
            # Encode history images to VAE embeddings
            # wm_history shape: (B, n_obs + n_pred, C, H, W)
            T = wm_history.shape[1]
            wm_history_flat = wm_history.view(B * T, *wm_history.shape[2:])  # (B*T, C, H, W)
            
            # Get VAE embeddings (ensure float32 for VAE quantization)
            wm_history_flat = wm_history_flat.float()  # Ensure float32
            
            # Process VAE encoding in smaller batches to prevent OOM
            vae_batch_size = 32  # Process 32 images at a time
            total_frames = wm_history_flat.shape[0]
            
            # img_to_idxBl returns a list of tensors (one per scale)
            # We need to collect indices for each scale separately
            all_scale_indices = None  # Will be list of lists
            
            for i in range(0, total_frames, vae_batch_size):
                batch_imgs = wm_history_flat[i:i + vae_batch_size]
                batch_indices_list = self.vae.img_to_idxBl(batch_imgs)  # List of (B, pn*pn)
                
                if all_scale_indices is None:
                    # Initialize: one list per scale
                    all_scale_indices = [[] for _ in range(len(batch_indices_list))]
                
                for scale_idx, scale_indices in enumerate(batch_indices_list):
                    all_scale_indices[scale_idx].append(scale_indices)
                
                # Clear intermediate tensors
                del batch_imgs, batch_indices_list
            
            # Concatenate indices for each scale
            gt_indices_per_scale = [torch.cat(scale_list, dim=0) for scale_list in all_scale_indices]
            del all_scale_indices
            
            gt_emb = self.vae.quantize.idxBl_to_var_input(gt_indices_per_scale)  # (B*T, ...)
            
            # Get prediction target (last n_pred images)
            n_pred = self.config.n_pred_img_steps
            n_obs = self.config.n_obs_img_steps
            target_emb = gt_emb.view(B, T, *gt_emb.shape[1:])[:, -n_pred:]  # (B, n_pred, ...)
            
            # Compute embedding-based uncertainty for WM reward
            # Use VAE embedding variance/diversity as proxy for prediction difficulty
            # States with higher embedding variance/magnitude indicate more complex/uncertain regions
            if target_emb.dim() > 2:
                target_flat = target_emb.view(B, -1)
            else:
                target_flat = target_emb
            
            # Method 1: Embedding magnitude (states with larger embeddings may be harder to predict)
            emb_magnitude = target_flat.norm(dim=-1)
            
            # Method 2: Embedding variance across channels (more variance = more complex)
            if target_emb.dim() >= 3:
                emb_var = target_emb.var(dim=-1).mean(dim=tuple(range(1, target_emb.dim()-1)))
            else:
                emb_var = target_emb.var(dim=-1)
            
            # Combine signals: normalize both to [0, 1] and average
            mag_normalized = (emb_magnitude - emb_magnitude.min()) / (emb_magnitude.max() - emb_magnitude.min() + 1e-8)
            var_normalized = (emb_var - emb_var.min()) / (emb_var.max() - emb_var.min() + 1e-8)
            
            # Weight: 0.6 magnitude + 0.4 variance
            uncertainty = 0.6 * mag_normalized + 0.4 * var_normalized
            
            # Add small random noise to encourage exploration
            uncertainty = uncertainty + 0.1 * torch.rand_like(uncertainty)
            
            # Cleanup
            del gt_indices_per_scale, gt_emb
            
            # Compute reward: Explorer should find high-uncertainty states
            # reward = alpha * uncertainty
            rewards = self.config.reward_alpha * uncertainty
            
            # Add to batch
            batch['wm_uncertainty'] = uncertainty
            batch['wm_reward'] = rewards
            batch['gt_emb'] = target_emb
            
        return batch
    
    def _build_ppo_batch_from_dataset(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Build PPO training batch from sequential dataset batch.
        
        Args:
            batch: Sequential batch from dataset
            
        Returns:
            PPO batch dict
        """
        B = batch['observation.state'].shape[0]
        
        # Get state embeddings from observation.state
        state = batch['observation.state']  # (B, state_dim)
        state_emb = self.policy.model.state_proj(state)  # (B, proj_width)
        
        # Get actions from dataset (ground truth actions)
        actions = batch['action'][:, 0, :]  # (B, chunk, action_dim) -> (B, action_dim) first action
        
        # Truncate to actual action dim
        action_dim = self.config.action_dim
        if actions.shape[-1] > action_dim:
            actions = actions[..., :action_dim]
        
        # Build full batch for forward_with_actor (with images for vision understanding)
        # Extract current frame from history for image input
        if 'observation.images.image0_history' in batch:
            # History is (B, T+pred, C, H, W), take the last observation frame (before prediction)
            img_history = batch['observation.images.image0_history']
            n_obs = self.config.history_length  # number of observation frames
            current_img = img_history[:, n_obs - 1]  # (B, C, H, W) - last observation frame
            
            # Create image mask (all ones for valid images)
            img_mask = torch.ones(B, dtype=torch.bool, device=self.device)
            
            # Build full batch with images for forward_with_actor
            forward_batch = {
                'observation.images.image0': current_img,
                'observation.images.image0_mask': img_mask,
                'observation.state': state,
                'task': batch.get('task', ["perform the task"] * B),
            }
        else:
            # Fallback to simplified mode if no images
            forward_batch = {'state_emb': state_emb}
        
        # Get Explorer's predicted actions and log probs
        with torch.no_grad():
            explorer_output = self.policy.forward_with_actor(
                forward_batch,
                actor_name='explorer',
                return_action_stats=True,
            )
            action_mean = explorer_output['action']  # (B, action_dim)
            # Get state_emb from output (computed in forward if using full path)
            if 'state_emb' in explorer_output:
                state_emb = explorer_output['state_emb']
            
            # Compute log prob under Explorer's distribution
            log_std = self.phase1_trainer.log_std
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(action_mean, std)
            old_log_probs = dist.log_prob(actions).sum(dim=-1)  # (B,)
            
            # Get value estimates
            values = self.phase1_trainer.value_head(state_emb).squeeze(-1)  # (B,)
        
        # Get rewards from WM uncertainty
        rewards = batch['wm_reward']  # (B,)
        
        # Detect episode boundaries based on frame_idx and episode_idx
        dones = torch.zeros(B, dtype=torch.bool, device=self.device)
        
        if 'frame_idx' in batch and 'episode_idx' in batch:
            frame_idx = batch['frame_idx']  # (B,) or list
            episode_idx = batch['episode_idx']  # (B,) or list
            
            # Convert to tensor if needed
            if isinstance(frame_idx, list):
                frame_idx = torch.tensor(frame_idx, device=self.device)
            if isinstance(episode_idx, list):
                episode_idx = torch.tensor(episode_idx, device=self.device)
            
            # Detect episode boundaries: mark as done when episode_idx changes
            # For sequential data, check if next sample is from different episode
            for i in range(B - 1):
                if episode_idx[i] != episode_idx[i + 1]:
                    dones[i] = True
            
            # Last sample in batch: check if it's the last frame of the episode
            # We estimate max episode length from frame_idx distribution
            if 'episode_length' in batch:
                max_frames = batch['episode_length']
                if isinstance(max_frames, (list, torch.Tensor)):
                    max_frames = max_frames[-1] if isinstance(max_frames, list) else max_frames[-1].item()
                if frame_idx[-1].item() >= max_frames - 1:
                    dones[-1] = True
            else:
                # Use heuristic: typical episode length is ~50 frames
                # Mark as done if frame_idx is high (>40) and at batch end
                if frame_idx[-1].item() > 40:
                    dones[-1] = True
        
        # Compute GAE
        rewards_list = rewards.tolist()
        values_list = values.tolist()
        dones_list = dones.tolist()
        
        advantages = []
        gae = 0.0
        gamma = self.config.phase1_gamma
        gae_lambda = self.config.phase1_gae_lambda
        
        for t in reversed(range(B)):
            if t == B - 1:
                next_val = 0.0
            else:
                next_val = values_list[t + 1]
            
            delta = rewards_list[t] + gamma * next_val * (1.0 - float(dones_list[t])) - values_list[t]
            gae = delta + gamma * gae_lambda * (1.0 - float(dones_list[t])) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values_list)]
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        
        # Detach all tensors to allow multiple PPO epochs over same data
        ppo_batch = {
            'actions': actions.detach(),
            'old_log_probs': old_log_probs.detach(),
            'old_values': values.detach(),
            'advantages': advantages.detach(),
            'returns': returns.detach(),
            'state_emb': state_emb.detach(),  # Important: detach to allow multiple backward passes
            # Pass through original batch data for forward pass
            'observation.state': state.detach(),
        }
        
        # Include image data for full vision path during PPO updates
        if 'observation.images.image0_history' in batch:
            img_history = batch['observation.images.image0_history']
            n_obs = self.config.history_length
            current_img = img_history[:, n_obs - 1]  # (B, C, H, W)
            ppo_batch['observation.images.image0'] = current_img.detach()
            ppo_batch['observation.images.image0_mask'] = torch.ones(B, dtype=torch.bool, device=self.device)
            ppo_batch['task'] = batch.get('task', ["perform the task"] * B)
        
        return ppo_batch
    
    def _concatenate_ppo_batches(self, batches: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Concatenate multiple PPO batches into one large batch.
        
        Args:
            batches: List of PPO batch dicts
            
        Returns:
            Combined PPO batch dict
        """
        if not batches:
            raise ValueError("Cannot concatenate empty batch list")
        
        combined = {}
        keys = batches[0].keys()
        
        for key in keys:
            values = [batch[key] for batch in batches]
            # Handle task field (list of strings) differently
            if key == 'task':
                # Concatenate string lists
                combined[key] = []
                for v in values:
                    if isinstance(v, list):
                        combined[key].extend(v)
                    else:
                        combined[key].append(v)
            else:
                # Tensor concatenation
                combined[key] = torch.cat(values, dim=0)
        
        return combined

    def _build_ppo_batch_from_buffer(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Build PPO training batch from sequential rollout buffer.
        
        Used for online RL mode where data comes from environment interaction.
        
        Args:
            batch: Sequential batch from rollout buffer
            
        Returns:
            PPO batch dict
        """
        B = batch['observation.state'].shape[0]
        
        # Get state embeddings from observation.state
        state = batch['observation.state']  # (B, state_dim)
        state_emb = self.policy.model.state_proj(state)  # (B, proj_width)
        
        # Get actions from buffer (actions actually taken)
        actions = batch['action'][:, 0, :]  # (B, chunk, action_dim) -> (B, action_dim)
        
        # Truncate to actual action dim
        action_dim = self.config.action_dim
        if actions.shape[-1] > action_dim:
            actions = actions[..., :action_dim]
        
        # Get stored log probs and values from buffer
        old_log_probs = batch['log_prob'][:, 0]  # (B, chunk) -> (B,)
        old_values = batch['value'][:, 0]  # (B, chunk) -> (B,)
        
        # Get rewards from buffer
        rewards = batch['reward'][:, 0]  # (B, chunk) -> (B,)
        
        # Get done flags
        dones = batch.get('done', torch.zeros(B, dtype=torch.bool, device=self.device))
        if dones.dim() > 1:
            dones = dones[:, 0] if dones.shape[1] > 0 else dones.squeeze(-1)
        
        # Compute GAE
        rewards_list = rewards.tolist()
        values_list = old_values.tolist()
        dones_list = dones.tolist() if isinstance(dones, torch.Tensor) else [False] * B
        
        advantages = []
        gae = 0.0
        gamma = self.config.phase1_gamma
        gae_lambda = self.config.phase1_gae_lambda
        
        for t in reversed(range(B)):
            if t == B - 1:
                next_val = 0.0
            else:
                next_val = values_list[t + 1]
            
            delta = rewards_list[t] + gamma * next_val * (1.0 - float(dones_list[t])) - values_list[t]
            gae = delta + gamma * gae_lambda * (1.0 - float(dones_list[t])) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values_list)]
        
        advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ppo_batch = {
            'actions': actions,
            'old_log_probs': old_log_probs,
            'old_values': old_values,
            'advantages': advantages,
            'returns': returns,
            'state_emb': state_emb,
            'observation.state': state,
        }
        
        return ppo_batch

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
            # Phase 2 trainer's save_checkpoint expects iteration number
            self.phase2_trainer.save_checkpoint(step)
        
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
