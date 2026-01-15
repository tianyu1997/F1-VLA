"""
F1-VLA and RoboTwin Integration for Explorer Training

This module provides the integration between the Explorer training modules
and the actual F1-VLA model with RoboTwin environment.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# F1-VLA Model Loading (参考 train_hf.py 和 f1_policy.py)
# =============================================================================

def load_f1_vla_policy(
    ckpt_path: str,
    device: str = "cuda",
    add_explorer: bool = True,
    train_act_expert_only: bool = False,
    explorer_checkpoint_path: Optional[str] = None,
    actor_checkpoint_path: Optional[str] = None,
    explorer_init_from: str = "auto",
) -> Tuple[nn.Module, nn.Module]:
    """
    Load F1-VLA policy with Explorer actor.
    
    模仿 train_hf.py 的加载方式：
    1. 直接从一个 checkpoint 加载完整模型（包含 VAE 和 WM）
    2. 不分块加载
    3. Explorer actor 加载顺序（可配置）：
       - "explorer": 优先加载已保存的 explorer checkpoint
       - "actor": 从 actor checkpoint 初始化
       - "gemma_expert": 从预训练的 gemma_expert 复制 (random_init=False)
       - "random": 随机初始化 (random_init=True)
       - "auto": 自动选择（explorer -> actor -> gemma_expert）
    
    Args:
        ckpt_path: Path to F1-VLA checkpoint directory (包含完整模型)
        device: Device to load model on
        add_explorer: Whether to add Explorer actor
        train_act_expert_only: Whether to train only the action expert (for explorer training)
        explorer_checkpoint_path: Path to saved explorer weights (optional)
        actor_checkpoint_path: Path to saved actor weights (optional)
        explorer_init_from: How to initialize explorer ["auto", "explorer", "actor", "gemma_expert", "random"]
        
    Returns:
        policy: F1_VLA policy with Explorer actor
        vae: VAE model (from policy.vae)
    """
    from f1_vla.src.models.configuration_f1 import F1Config
    from f1_vla.src.policies.f1_policy import F1_VLA
    from f1_vla.src.processors.train_processors.policy_trainer import PolicyTrainingArguments
    from f1_vla.src.utils.utils import load_ckpt
    from omegaconf import OmegaConf
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading F1-VLA from {ckpt_path} (train_hf.py style)")
    
    # Step 1: Load config from checkpoint path (same as train_hf.py)
    config = F1Config.from_pretrained(ckpt_path)
    
    # Fix tokenizer path to local path if the original doesn't exist
    local_tokenizer_path = str(project_root / "paligemma-3b-pt-224")
    if not os.path.exists(config.language_tokenizer_path) and os.path.exists(local_tokenizer_path):
        config.language_tokenizer_path = local_tokenizer_path
        logger.info(f"Using local tokenizer: {local_tokenizer_path}")
    
    # Step 2: Create training_args to control freezing behavior
    training_args = PolicyTrainingArguments(
        output_dir="./outputs",  # Dummy path, not used for inference
        train_act_expert_only=train_act_expert_only,
        freeze_vision_encoder=True,  # Always freeze vision encoder for explorer training
        freeze_gen_expert=True,  # Freeze generation expert for explorer training
    )
    
    # Step 3: Create policy model (same as train_hf.py: policy = F1_VLA(**kwargs))
    kwargs = {"config": config, "training_args": training_args}
    policy = F1_VLA(**kwargs)
    
    # Step 4: Load weights from checkpoint using load_ckpt (same as train_hf.py)
    # Create a minimal OmegaConf config with load_ckpt path
    load_config = OmegaConf.create({
        "exp": {
            "load_ckpt": ckpt_path  # Load from checkpoint directory
        }
    })
    policy = load_ckpt(policy, load_config)
    
    logger.info("  Loaded F1-VLA weights from checkpoint")
    
    # Step 5: Add Explorer actor with flexible initialization options
    if add_explorer:
        explorer_loaded = False
        
        # Option 1: Try to load existing explorer checkpoint
        if explorer_init_from in ["auto", "explorer"] and explorer_checkpoint_path:
            if os.path.exists(explorer_checkpoint_path):
                try:
                    # First add actor with random init, then load weights
                    logger.info(f"Loading Explorer from checkpoint: {explorer_checkpoint_path}")
                    policy.add_actor('explorer', random_init=True)
                    policy.load_actor('explorer', explorer_checkpoint_path)
                    explorer_loaded = True
                    logger.info("  ✓ Explorer loaded from saved checkpoint")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load explorer checkpoint: {e}")
                    # Remove the failed actor
                    if 'explorer' in policy.list_actors():
                        del policy.model.gemma_experts['explorer']
        
        # Option 2: Try to load from actor checkpoint
        if not explorer_loaded and explorer_init_from in ["auto", "actor"] and actor_checkpoint_path:
            if os.path.exists(actor_checkpoint_path):
                try:
                    logger.info(f"Initializing Explorer from actor checkpoint: {actor_checkpoint_path}")
                    policy.add_actor('explorer', random_init=True)
                    policy.load_actor('explorer', actor_checkpoint_path)
                    explorer_loaded = True
                    logger.info("  ✓ Explorer initialized from actor checkpoint")
                except Exception as e:
                    logger.warning(f"  ✗ Failed to load actor checkpoint: {e}")
                    if 'explorer' in policy.list_actors():
                        del policy.model.gemma_experts['explorer']
        
        # Option 3: Copy from gemma_expert (default fallback)
        if not explorer_loaded and explorer_init_from in ["auto", "gemma_expert"]:
            logger.info("Initializing Explorer by copying from gemma_expert weights")
            policy.add_actor('explorer', random_init=False)
            explorer_loaded = True
            logger.info("  ✓ Explorer initialized from gemma_expert (random_init=False)")
        
        # Option 4: Random initialization
        if not explorer_loaded and explorer_init_from == "random":
            logger.info("Initializing Explorer with random weights")
            policy.add_actor('explorer', random_init=True)
            explorer_loaded = True
            logger.info("  ✓ Explorer initialized randomly")
        
        # Final check
        if not explorer_loaded:
            raise ValueError(
                f"Failed to initialize Explorer actor. "
                f"explorer_init_from={explorer_init_from}, "
                f"explorer_checkpoint_path={explorer_checkpoint_path}, "
                f"actor_checkpoint_path={actor_checkpoint_path}"
            )
        
        policy.active_actor = 'explorer'
        logger.info(f"  Available actors: {policy.list_actors()}")
    
    # Step 6: Move to device
    policy.to(device)
    
    # Return policy and its VAE
    return policy, policy.vae


def load_vae(
    vae_path: str,
    config: Any,
    device: str = "cuda",
) -> nn.Module:
    """
    Load VAE model for image encoding.
    
    Args:
        vae_path: Path to VAE checkpoint
        config: F1Config with VAE configuration
        device: Device to load on
        
    Returns:
        vae: VAE model
    """
    logger = logging.getLogger(__name__)
    
    # Try to import VAR VAE
    try:
        # Import from f1_vla models
        from f1_vla.src.models.wm.vqvae import VQVAE
        
        # Get VAE config from F1Config
        vae_config = config.gen_expert_config.vae if hasattr(config, 'gen_expert_config') else None
        
        # Determine test_mode and freeze_encoder from config
        # Priority: vae_test_mode > infer from pixel_loss_weight
        pixel_loss_weight = getattr(config, 'pixel_loss_weight', 0.0)
        test_mode = getattr(config, 'vae_test_mode', pixel_loss_weight == 0)  # Use explicit config or infer
        freeze_encoder = getattr(config, 'vae_freeze_encoder', True)  # Default: only train decoder
        
        if vae_config:
            vae = VQVAE(
                vocab_size=getattr(vae_config, 'vocab_size', 4096),
                z_channels=getattr(vae_config, 'z_channels', 32),
                ch=getattr(vae_config, 'ch', 160),
                dropout=getattr(vae_config, 'dropout', 0.0),
                test_mode=test_mode,
                freeze_encoder=freeze_encoder,
                share_quant_resi=4,
                v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            )
        else:
            # Default VAE config
            vae = VQVAE(
                vocab_size=4096,
                z_channels=32,
                ch=160,
                dropout=0.0,
                test_mode=test_mode,
                freeze_encoder=freeze_encoder,
                share_quant_resi=4,
                v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
            )
        
        # Load weights
        if vae_path and os.path.exists(vae_path):
            logger.info(f"Loading VAE weights from {vae_path}")
            state_dict = torch.load(vae_path, map_location=device)
            vae.load_state_dict(state_dict, strict=False)
        
        vae.to(device)
        vae.eval()
        
        return vae
        
    except ImportError as e:
        logger.warning(f"Could not import VAR VQVAE: {e}")
        logger.info("Using mock VAE")
        return create_mock_vae(device)


def create_mock_vae(device: str = "cuda") -> nn.Module:
    """Create a mock VAE for testing when actual VAE is not available."""
    
    class MockVAE(nn.Module):
        def __init__(self, embed_dim=1280, vocab_size=4096, z_channels=32):
            super().__init__()
            self.embed_dim = embed_dim
            self.vocab_size = vocab_size
            self.V = vocab_size
            self.Cvae = z_channels
            self.z_channels = z_channels
            self.patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
            
            # Codebook for embeddings
            self.codebook = nn.Embedding(vocab_size, embed_dim)
            
            # Simple encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, z_channels, 4, stride=2, padding=1),
            )
            
        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode image to embedding."""
            if len(x.shape) == 3:
                x = x.unsqueeze(0)
            # Simple encoding
            z = self.encoder(x)
            z = z.flatten(1)  # (B, z_channels * H * W)
            # Project to embed_dim
            z = F.adaptive_avg_pool1d(z.unsqueeze(1), self.embed_dim).squeeze(1)
            return z
        
        def get_codebook_embedding(self) -> torch.Tensor:
            return self.codebook.weight
        
        def forward(self, x):
            return self.encode(x)
    
    vae = MockVAE()
    vae.to(device)
    vae.eval()
    return vae


# =============================================================================
# RoboTwin Environment Integration
# =============================================================================

def create_robotwin_env(
    task_name: str = "random_exploration",
    task_config_name: str = "demo_randomized",
    history_length: int = 4,
    max_steps: int = 200,
    image_size: Tuple[int, int] | int = (224, 224),
    device: str = "cuda",
    render_mode: str = "rasterize",
    action_scale: float = 0.5,
    single_arm: bool = False,
    **kwargs
) -> Any:
    """
    Create RoboTwin environment for Explorer training.
    
    Args:
        task_name: Name of the task
        task_config_name: Configuration name
        history_length: Number of history frames (L)
        max_steps: Maximum steps per episode
        image_size: Observation image size (int or tuple)
        device: Device for processing
        render_mode: Rendering mode
        action_scale: Scale for action bounds (smaller = safer)
        single_arm: Use single arm mode
        **kwargs: Additional arguments for F1RLEnv
        
    Returns:
        env: F1RLEnv environment
    """
    logger = logging.getLogger(__name__)
    
    # Convert image_size to tuple if needed
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    # Add RoboTwin to path
    robotwin_dir = project_root / "RoboTwin"
    sys.path.insert(0, str(robotwin_dir))
    
    try:
        from rl.f1_rl_env import F1RLEnv, load_embodiment_config
        import yaml
        
        # Load base config
        config_path = robotwin_dir / "task_config" / f"{task_config_name}.yml"
        if not config_path.exists():
            # Try default config
            config_path = robotwin_dir / "task_config" / "demo_randomized.yml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                task_config = yaml.safe_load(f)
        else:
            # Minimal default config
            task_config = {
                'task_name': task_name,
                'embodiment': ['franka-panda', 'franka-panda', 0.6],
                'render_freq': 0,
                'save_data': False,
                'need_plan': False,
                'need_topp': False,
                'need_planner': False,
            }
        
        # Update config
        task_config['task_name'] = task_name
        task_config['render_freq'] = 0  # Headless
        task_config['save_data'] = False
        task_config['need_plan'] = False
        task_config['need_topp'] = False  # Disable motion planner for RL
        task_config['need_planner'] = False
        
        # Load embodiment config
        task_config = load_embodiment_config(task_config, base_dir=str(robotwin_dir))
        
        # Create environment
        env = F1RLEnv(
            task_config=task_config,
            phase="student",  # Explorer uses student phase
            history_length=history_length,
            max_steps=max_steps,
            device=device,
            image_size=image_size,
            action_dim=32,  # Unified action dim
            state_dim=32,   # Unified state dim
            render_mode=render_mode,
            action_scale=action_scale,
            single_arm=single_arm,
            **kwargs
        )
        
        logger.info(f"Created RoboTwin environment: {task_name}")
        logger.info(f"  History length: {history_length}")
        logger.info(f"  Max steps: {max_steps}")
        logger.info(f"  Action scale: {action_scale}")
        
        return env
        
    except (ImportError, FileNotFoundError, Exception) as e:
        logger.warning(f"Could not create RoboTwin environment: {e}")
        logger.info("Using mock environment")
        return create_mock_env(
            history_length=history_length,
            max_steps=max_steps,
            image_size=image_size,
            device=device
        )


def create_mock_env(
    history_length: int = 4,
    max_steps: int = 200,
    image_size: Tuple[int, int] | int = (224, 224),
    device: str = "cuda",
    action_dim: int = 32,
) -> Any:
    """Create mock environment for testing."""
    
    # Convert image_size to tuple if it's an int
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    class MockEnv:
        def __init__(self):
            self.history_length = history_length
            self.max_steps = max_steps
            self.image_size = image_size
            self.device = device
            self.action_dim = action_dim
            self.state_dim = 32
            self.step_count = 0
            
            # Action/observation spaces (gym-like)
            self.action_space = type('Space', (), {
                'shape': (self.action_dim,),
                'sample': lambda self=self: np.random.uniform(-1, 1, self.action_dim).astype(np.float32)
            })()
            
        def reset(self, seed=None):
            self.step_count = 0
            # Generate random image with shape (C, H, W) - single frame, not history
            H, W = self.image_size
            obs = {
                'state': np.zeros(self.state_dim, dtype=np.float32),
                'action_history': np.zeros((self.history_length, self.action_dim), dtype=np.float32),
                'head_rgb': np.random.randint(0, 255, (3, H, W), dtype=np.uint8),  # WM camera
                'wrist_rgb': np.random.randint(0, 255, (3, H, W), dtype=np.uint8),
            }
            info = {'embodiment': 'mock', 'control_mode': 'delta_qpos'}
            return obs, info
        
        def step(self, action):
            self.step_count += 1
            H, W = self.image_size
            obs = {
                'state': np.random.randn(self.state_dim).astype(np.float32),
                'action_history': np.random.uniform(-1, 1, (self.history_length, self.action_dim)).astype(np.float32),
                'head_rgb': np.random.randint(0, 255, (3, H, W), dtype=np.uint8),
                'wrist_rgb': np.random.randint(0, 255, (3, H, W), dtype=np.uint8),
            }
            reward = 0.0  # Explorer computes its own reward
            terminated = False
            truncated = self.step_count >= self.max_steps
            info = {}
            return obs, reward, terminated, truncated, info
        
        def close(self):
            pass
    
    return MockEnv()


# =============================================================================
# Explorer Environment Wrapper
# =============================================================================

class ExplorerEnvWrapper:
    """
    Wrapper that combines F1-VLA policy, VAE, and RoboTwin environment
    for Explorer training.
    
    Handles:
    - Running PaliGemma + WM to get predictions
    - Computing VAE embeddings
    - Collecting observations for Explorer input
    """
    
    def __init__(
        self,
        policy: nn.Module,
        vae: nn.Module,
        env: Any,
        history_length: int = 4,
        device: str = "cuda",
    ):
        """
        Initialize Explorer environment wrapper.
        
        Args:
            policy: F1-VLA policy with Explorer actor
            vae: VAE model for embedding extraction
            env: Base environment (RoboTwin or mock)
            history_length: Number of history frames (L)
            device: Torch device
        """
        self.policy = policy
        self.vae = vae
        self.env = env
        self.history_length = history_length
        self.device = device
        
        # History buffers
        self.gt_emb_history = []      # L+1 frames
        self.pred_emb_history = []    # L frames
        self.uncertainty_history = [] # L frames
        self.action_history = []      # L frames
        self.state_history = []       # L+1 frames
        
        # Current step info
        self.last_wm_output = None
        
        # Import embedding extractor
        from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
        self.embedding_extractor = VAEEmbeddingExtractor(
            vae=vae,
            embedding_dim=getattr(vae, 'z_channels', 32),
        )
        
    def reset(self, seed=None):
        """Reset environment and initialize histories."""
        obs, info = self.env.reset(seed=seed)
        
        # Clear histories
        self.gt_emb_history = []
        self.pred_emb_history = []
        self.uncertainty_history = []
        self.action_history = []
        self.state_history = []
        self.last_wm_output = None
        
        # Extract initial embedding from observation
        if 'wrist_rgb' in obs:
            # Use last frame from history
            img = obs['wrist_rgb'][-1]  # (C, H, W)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            img_tensor = img_tensor / 255.0  # Normalize to [0, 1]
            
            with torch.no_grad():
                gt_emb = self.embedding_extractor.encode_image(img_tensor)
        else:
            gt_emb = torch.zeros(1, 1280, device=self.device)
        
        # Initialize histories with zeros/copies
        for _ in range(self.history_length + 1):
            self.gt_emb_history.append(gt_emb.clone())
            self.state_history.append(torch.from_numpy(obs['state']).float().to(self.device))
        
        for _ in range(self.history_length):
            self.pred_emb_history.append(torch.zeros_like(gt_emb))
            self.uncertainty_history.append(torch.zeros(1, device=self.device))
            self.action_history.append(torch.zeros(32, device=self.device))
        
        return self._build_explorer_input(obs), info
    
    def step(self, action: torch.Tensor):
        """
        Execute one step with Explorer action.
        
        Args:
            action: Action from Explorer (32-dim tensor)
            
        Returns:
            explorer_obs: Observation dict for Explorer
            reward_info: Dict with info for reward computation
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional info
        """
        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action_np = action.detach().cpu().numpy()
        else:
            action_np = action
        
        # Execute action in environment
        obs, env_reward, terminated, truncated, info = self.env.step(action_np)
        
        # Extract GT embedding from new observation
        if 'wrist_rgb' in obs:
            img = obs['wrist_rgb'][-1]  # (C, H, W)
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(self.device)
            img_tensor = img_tensor / 255.0
            
            with torch.no_grad():
                gt_emb = self.embedding_extractor.encode_image(img_tensor)
        else:
            gt_emb = torch.zeros(1, 1280, device=self.device)
        
        # Get WM prediction (if policy supports it)
        pred_emb, uncertainty = self._run_world_model(obs, action)
        
        # Update histories
        self.gt_emb_history.append(gt_emb)
        self.gt_emb_history = self.gt_emb_history[-(self.history_length + 1):]
        
        self.pred_emb_history.append(pred_emb)
        self.pred_emb_history = self.pred_emb_history[-self.history_length:]
        
        self.uncertainty_history.append(uncertainty)
        self.uncertainty_history = self.uncertainty_history[-self.history_length:]
        
        # Ensure action has consistent shape
        action_tensor = torch.from_numpy(action_np).float().to(self.device)
        if action_tensor.shape[0] != self.action_history[0].shape[0]:
            # Pad or truncate to match expected action dim
            expected_dim = self.action_history[0].shape[0]
            if action_tensor.shape[0] < expected_dim:
                action_tensor = F.pad(action_tensor, (0, expected_dim - action_tensor.shape[0]))
            else:
                action_tensor = action_tensor[:expected_dim]
        
        self.action_history.append(action_tensor)
        self.action_history = self.action_history[-self.history_length:]
        
        self.state_history.append(torch.from_numpy(obs['state']).float().to(self.device))
        self.state_history = self.state_history[-(self.history_length + 1):]
        
        # Build reward info
        reward_info = {
            'gt_emb': gt_emb,
            'pred_emb': pred_emb,
            'uncertainty': uncertainty,
            'action': torch.from_numpy(action_np).float().to(self.device),
            # Previous step info for delayed reward
            'prev_gt_emb': self.gt_emb_history[-2] if len(self.gt_emb_history) > 1 else gt_emb,
            'prev_pred_emb': self.pred_emb_history[-2] if len(self.pred_emb_history) > 1 else pred_emb,
            'prev_uncertainty': self.uncertainty_history[-2] if len(self.uncertainty_history) > 1 else uncertainty,
        }
        
        explorer_obs = self._build_explorer_input(obs)
        
        return explorer_obs, reward_info, terminated, truncated, info
    
    def _run_world_model(
        self, 
        obs: Dict[str, np.ndarray], 
        action: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run World Model to get prediction embedding and uncertainty.
        
        Returns:
            pred_emb: Predicted embedding (B, embed_dim)
            uncertainty: Prediction uncertainty (B,)
        """
        # Get embedding dim from first gt_emb in history
        if self.gt_emb_history:
            emb_dim = self.gt_emb_history[-1].shape[-1]
        else:
            emb_dim = 32  # Default to z_channels
        
        # Try to use actual WM forward pass if policy supports it
        if hasattr(self.policy, 'forward_with_world_model') and self.policy.config.use_world_model:
            try:
                # Build input batch for WM
                # Need: observation.images.image0_history, observation.state, action
                
                # Get image history from gt_emb history or raw images
                if len(self.gt_emb_history) >= self.history_length:
                    # Use stored observation images if available
                    if hasattr(self, '_img_history') and len(self._img_history) >= self.history_length:
                        img_history = torch.stack(self._img_history[-self.history_length:], dim=1)  # (1, T, C, H, W)
                    else:
                        # Fallback to embedding-based uncertainty
                        pred_emb = torch.randn(1, emb_dim, device=self.device) * 0.1
                        uncertainty = torch.rand(1, device=self.device)
                        return pred_emb, uncertainty
                    
                    # Build state history
                    state_history = torch.stack(self.state_history[-self.history_length:], dim=0).unsqueeze(0)  # (1, T, state_dim)
                    
                    # Build WM batch
                    wm_batch = {
                        'observation.images.image0_history': img_history,
                        'observation.state': state_history[:, -1],  # Current state
                        'action': torch.from_numpy(action).float().unsqueeze(0).unsqueeze(0).to(self.device),  # (1, 1, action_dim)
                    }
                    
                    with torch.no_grad():
                        loss_dict = self.policy.forward_with_world_model(
                            wm_batch,
                            train_gen_expert_only=True,
                            cur_n_obs_img_steps=self.history_length - 1,
                            cur_n_pred_img_steps=1,
                        )
                    
                    # Get uncertainty from WM accuracy
                    wm_acc = loss_dict.get('wm_acc_mean', torch.tensor(0.5))
                    uncertainty = (1.0 - wm_acc) * torch.ones(1, device=self.device)
                    
                    # Get predicted embedding (dummy for now, actual would decode from WM output)
                    pred_emb = torch.randn(1, emb_dim, device=self.device) * 0.1
                    
                    return pred_emb, uncertainty
                    
            except Exception as e:
                logging.getLogger(__name__).warning(f"WM forward failed in _run_world_model: {e}")
        
        # Fallback to mock WM output
        pred_emb = torch.randn(1, emb_dim, device=self.device) * 0.1
        uncertainty = torch.rand(1, device=self.device)
        
        return pred_emb, uncertainty
    
    def _build_explorer_input(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Build Explorer input from current observation and histories.
        
        Explorer input (at time t, after environment returns gt_{t+1}):
        - state_history: [s_{t-L+1}, ..., s_t, s_{t+1}] - L+1 frames
        - action_history: [a_{t-L+1}, ..., a_t] - L frames
        - gt_img_emb: [emb_{t-L+1}, ..., emb_t, emb_{t+1}] - L+1 frames
        - pred_img_emb: [pred_{t-L+2}, ..., pred_t, pred_{t+1}] - L frames
        - pred_uncertainty: [unc_{t-L+2}, ..., unc_t, unc_{t+1}] - L frames
        """
        # Stack histories (handling potential shape mismatches)
        def safe_stack(tensor_list, dim=0):
            """Stack tensors, handling shape mismatches."""
            if not tensor_list:
                return torch.tensor([])
            # Get reference shape
            ref_shape = tensor_list[0].shape
            aligned = []
            for t in tensor_list:
                if t.shape != ref_shape:
                    # Reshape to match reference
                    if t.numel() == ref_shape.numel():
                        t = t.view(ref_shape)
                    else:
                        # Pad or truncate
                        flat_ref = ref_shape.numel()
                        flat_t = t.numel()
                        t_flat = t.flatten()
                        if flat_t < flat_ref:
                            t_flat = F.pad(t_flat, (0, flat_ref - flat_t))
                        else:
                            t_flat = t_flat[:flat_ref]
                        t = t_flat.view(ref_shape)
                aligned.append(t)
            return torch.stack(aligned, dim=dim)
        
        explorer_input = {
            # State history: L+1 frames
            'state_history': safe_stack(self.state_history[-self.history_length-1:], dim=0),
            
            # Action history: L frames
            'action_history': safe_stack(self.action_history[-self.history_length:], dim=0),
            
            # GT image embeddings: L+1 frames - squeeze batch dim before stacking
            'gt_img_emb': safe_stack([e.squeeze(0) if e.dim() > 1 else e for e in self.gt_emb_history[-self.history_length-1:]], dim=0),
            
            # Predicted image embeddings: L frames
            'pred_img_emb': safe_stack([e.squeeze(0) if e.dim() > 1 else e for e in self.pred_emb_history[-self.history_length:]], dim=0),
            
            # Prediction uncertainty: L frames
            'pred_uncertainty': safe_stack([u.squeeze() if u.dim() > 0 else u for u in self.uncertainty_history[-self.history_length:]], dim=0),
            
            # Raw observation (for debugging)
            'raw_state': torch.from_numpy(obs['state']).float().to(self.device),
        }
        
        return explorer_input
    
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_explorer_training_env(
    config: Dict[str, Any],
    device: str = "cuda",
) -> ExplorerEnvWrapper:
    """
    Create complete Explorer training environment.
    
    Args:
        config: Configuration dict with:
            - model.pretrained_path: F1-VLA config/checkpoint path
            - model.vae.checkpoint_path: VAE checkpoint path
            - environment.type: Environment type (robotwin, mock)
            - environment.observation.history_length: L value
            - environment.action.max_episode_steps: Max steps
        device: Torch device
        
    Returns:
        env: ExplorerEnvWrapper ready for training
    """
    logger = logging.getLogger(__name__)
    
    # Extract config values
    model_config = config.get('model', {})
    env_config = config.get('environment', {})
    
    # Load policy and VAE
    pretrained_path = model_config.get('pretrained_path', '')
    vae_path = model_config.get('vae', {}).get('checkpoint_path', '')
    
    policy, vae = load_f1_vla_policy(
        config_path=pretrained_path,
        checkpoint_path=pretrained_path,
        vae_path=vae_path,
        device=device,
        add_explorer=True,
    )
    
    # Create base environment
    env_type = env_config.get('type', 'robotwin')
    obs_config = env_config.get('observation', {})
    action_config = env_config.get('action', {})
    
    history_length = obs_config.get('history_length', 4)
    max_steps = action_config.get('max_episode_steps', 200)
    image_size = (obs_config.get('image_size', 224), obs_config.get('image_size', 224))
    
    if env_type == 'robotwin':
        base_env = create_robotwin_env(
            history_length=history_length,
            max_steps=max_steps,
            image_size=image_size,
            device=device,
        )
    else:
        base_env = create_mock_env(
            history_length=history_length,
            max_steps=max_steps,
            image_size=image_size,
            device=device,
        )
    
    # Create wrapper
    env = ExplorerEnvWrapper(
        policy=policy,
        vae=vae,
        env=base_env,
        history_length=history_length,
        device=device,
    )
    
    logger.info("Created Explorer training environment")
    
    return env
