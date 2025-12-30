"""
Teacher-Student Policy Wrapper for World Model Distillation.

Teacher: Full observation (head + wrist cameras) - frozen
Student: Wrist camera only - trainable (gen expert only)

The student learns to predict next frames while distilling the teacher's
KV memory state. This enables efficient deployment with only wrist camera.

Design principles:
1. Incremental design - does not modify existing F1_VLA code
2. Both policies share the same architecture (F1_VLA)
3. Teacher is completely frozen, student trains gen expert only
4. Memory state distillation loss + GT prediction loss
"""

import copy
import logging
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from f1_vla.src.policies.f1_policy import F1_VLA
from f1_vla.src.models.configuration_f1 import F1Config

logger = logging.getLogger(__name__)


class TeacherStudentPolicy(nn.Module):
    """
    Teacher-Student wrapper for F1-VLA World Model distillation.
    
    Teacher Policy: Full observation (head + wrist camera), completely frozen.
    Student Policy: Wrist camera only, trains gen expert with:
        - GT loss: next frame prediction loss
        - Memory distillation loss: MSE between student and teacher memory states
    
    Args:
        config: F1Config for both teacher and student
        teacher_ckpt: Path to pretrained teacher checkpoint (if None, uses random init)
        student_ckpt: Path to pretrained student checkpoint (if None, copies from teacher)
        memory_loss_weight: Weight for memory distillation loss (relative to gt_loss)
        use_memory_distillation: Whether to use memory distillation loss
        training_args: Training arguments for setting requires_grad
    
    Checkpoint Loading Logic:
        - If both teacher_ckpt and student_ckpt are None: both use random init
        - If only teacher_ckpt is provided: teacher loads it, student copies from teacher
        - If only student_ckpt is provided: teacher uses random init, student loads it
        - If both provided: each loads its own checkpoint
    """
    
    config_class = F1Config
    
    def __init__(
        self,
        config: F1Config,
        teacher_ckpt: str = None,
        student_ckpt: str = None,
        memory_loss_weight: float = 0.5,
        use_memory_distillation: bool = True,
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.memory_loss_weight = memory_loss_weight
        self.use_memory_distillation = use_memory_distillation
        
        # Modify training_args to ensure gen_expert_only mode for student
        training_args = kwargs.get("training_args")
        if training_args is not None:
            # Force train_gen_expert_only for student
            training_args.train_gen_expert_only = True
            training_args.freeze_gen_expert = False
        
        # Create teacher policy (will be frozen)
        logger.info("[TeacherStudentPolicy] Creating teacher policy...")
        self.teacher = F1_VLA(config, **kwargs)
        
        # Create student policy (gen expert trainable)
        logger.info("[TeacherStudentPolicy] Creating student policy...")
        self.student = F1_VLA(config, **kwargs)
        
        # Load checkpoints based on provided paths
        # Logic: 
        #   1. Load teacher checkpoint if provided
        #   2. If student checkpoint provided, load it; otherwise copy from teacher
        if teacher_ckpt is not None:
            logger.info(f"[TeacherStudentPolicy] Loading teacher from {teacher_ckpt}")
            self._load_checkpoint(self.teacher, teacher_ckpt, "teacher")
        
        if student_ckpt is not None:
            # Student has its own checkpoint
            logger.info(f"[TeacherStudentPolicy] Loading student from {student_ckpt}")
            self._load_checkpoint(self.student, student_ckpt, "student")
        elif teacher_ckpt is not None:
            # No student checkpoint, copy weights from teacher
            logger.info("[TeacherStudentPolicy] Copying teacher weights to student (same initialization)")
            self._copy_weights_teacher_to_student()
        
        # Freeze teacher completely
        self._freeze_teacher()
        
        # Setup student for gen_expert_only training
        self._setup_student_training()
        
        # Share VAE between teacher and student (VAE is frozen anyway)
        self.student.vae = self.teacher.vae
        
        logger.info(f"[TeacherStudentPolicy] Initialized with memory_loss_weight={memory_loss_weight}, "
                   f"use_memory_distillation={use_memory_distillation}")
        
        # Report trainable parameters
        self._report_trainable_params()
    
    def _load_checkpoint(self, policy: F1_VLA, ckpt_path: str, name: str = "policy"):
        """Load checkpoint into a policy.
        
        Args:
            policy: The F1_VLA policy to load weights into
            ckpt_path: Path to checkpoint (directory or file)
            name: Name for logging (e.g., 'teacher' or 'student')
        """
        import os
        from safetensors.torch import load_file
        
        if os.path.isdir(ckpt_path):
            # Directory with safetensors
            safetensor_path = os.path.join(ckpt_path, "model.safetensors")
            if os.path.exists(safetensor_path):
                state_dict = load_file(safetensor_path)
                # Remove 'model.' prefix if present
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                missing, unexpected = policy.load_state_dict(state_dict, strict=False)
                logger.info(f"[TeacherStudentPolicy] Loaded {name} from {safetensor_path}")
                if missing:
                    logger.debug(f"[TeacherStudentPolicy] {name} missing keys: {len(missing)}")
                if unexpected:
                    logger.debug(f"[TeacherStudentPolicy] {name} unexpected keys: {len(unexpected)}")
            else:
                logger.warning(f"[TeacherStudentPolicy] No model.safetensors found in {ckpt_path}")
        else:
            # Single file
            state_dict = load_file(ckpt_path)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            policy.load_state_dict(state_dict, strict=False)
            logger.info(f"[TeacherStudentPolicy] Loaded {name} from {ckpt_path}")
    
    def _copy_weights_teacher_to_student(self):
        """Copy all weights from teacher to student for same initialization."""
        teacher_state = self.teacher.state_dict()
        missing, unexpected = self.student.load_state_dict(teacher_state, strict=False)
        logger.info(f"[TeacherStudentPolicy] Copied {len(teacher_state)} parameters from teacher to student")
        if missing:
            logger.debug(f"[TeacherStudentPolicy] Student missing keys after copy: {len(missing)}")
    
    def _freeze_teacher(self):
        """Completely freeze teacher policy."""
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        logger.info("[TeacherStudentPolicy] Teacher frozen (all parameters)")
    
    def _setup_student_training(self):
        """Setup student for gen_expert_only training (same as train_gen_expert_only mode)."""
        # The student's requires_grad should already be set by F1FlowMatching.set_requires_grad()
        # based on training_args.train_gen_expert_only = True
        # But let's verify and ensure gen expert is trainable
        
        if hasattr(self.student, 'model') and hasattr(self.student.model, 'gen_expert'):
            gen_expert = self.student.model.gen_expert
            for param in gen_expert.parameters():
                param.requires_grad = True
            logger.info("[TeacherStudentPolicy] Student gen_expert set to trainable")
        
        # Ensure memory module is trainable if using memory
        if self.config.use_memory and hasattr(self.student.model, 'memory_bank'):
            for param in self.student.model.memory_bank.parameters():
                param.requires_grad = True
            logger.info("[TeacherStudentPolicy] Student memory_bank set to trainable")
    
    def _report_trainable_params(self):
        """Report number of trainable parameters."""
        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        student_trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        logger.info(f"[TeacherStudentPolicy] Parameter counts:")
        logger.info(f"  Teacher trainable: {teacher_trainable:,} (should be 0)")
        logger.info(f"  Student trainable: {student_trainable:,}")
        logger.info(f"  Total parameters: {total_params:,}")
    
    def reset(self):
        """Reset action queue for both policies."""
        self.teacher.reset()
        self.student.reset()
    
    def _prepare_student_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Prepare batch for student - mask out head camera observation.
        
        The student only sees wrist camera, so we zero out head camera images.
        This simulates deployment with only wrist camera.
        
        Camera config:
            - head camera: observation.images.image0 (head_rgb)
            - wrist camera: observation.images.image1 (wrist_rgb)
        """
        student_batch = {}
        
        for key, value in batch.items():
            if key == "observation.images.image0":
                # Zero out head camera for student
                # Keep same shape but all zeros
                student_batch[key] = torch.zeros_like(value)
                logger.debug(f"[Student] Zeroed out {key}")
            elif key == "observation.images.image0_mask":
                # Set mask to False (invalid) for head camera
                student_batch[key] = torch.zeros_like(value, dtype=torch.bool)
                logger.debug(f"[Student] Masked {key}")
            elif key == "observation.images.image0_history":
                # Zero out head camera history for world model input
                student_batch[key] = torch.zeros_like(value)
                logger.debug(f"[Student] Zeroed out history {key}")
            elif key == "observation.images.image0_target":
                # Keep target same - student still predicts the same target
                student_batch[key] = value
            else:
                # Keep all other keys unchanged
                student_batch[key] = value
        
        return student_batch
    
    def _compute_memory_distillation_loss(
        self,
        teacher_memory: List[Tuple[Tensor, Tensor]],
        student_memory: List[Tuple[Tensor, Tensor]],
    ) -> Tensor:
        """
        Compute MSE loss between teacher and student memory states.
        
        Memory state format: List of (key, value) tuples per layer
        Each tensor shape: (batch, memory_len, num_kv_heads, head_dim)
        
        Returns:
            MSE loss averaged across all memory slots
        """
        if teacher_memory is None or student_memory is None:
            return torch.tensor(0.0, device=next(self.student.parameters()).device)
        
        total_loss = 0.0
        num_layers = len(teacher_memory)
        
        for layer_idx in range(num_layers):
            teacher_k, teacher_v = teacher_memory[layer_idx]
            student_k, student_v = student_memory[layer_idx]
            
            # Detach teacher (no gradient through teacher)
            teacher_k = teacher_k.detach()
            teacher_v = teacher_v.detach()
            
            # MSE loss for keys and values
            k_loss = F.mse_loss(student_k, teacher_k)
            v_loss = F.mse_loss(student_v, teacher_v)
            
            total_loss = total_loss + k_loss + v_loss
        
        # Average over layers and K/V
        avg_loss = total_loss / (num_layers * 2)
        
        return avg_loss
    
    def forward_with_world_model(
        self,
        batch: Dict[str, Tensor],
        noise: Tensor = None,
        time: Tensor = None,
        cur_n_obs_img_steps: int = None,
        cur_n_pred_img_steps: int = None,
        train_gen_expert_only: bool = True,
        gen_out_loss_ratio: float = 1.0,
        return_images: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for teacher-student training.
        
        1. Teacher forward (no grad) with full observation
        2. Student forward with wrist-only observation
        3. Compute GT loss for student
        4. Compute memory distillation loss if enabled
        5. Combine losses
        
        Returns:
            loss_dict with combined loss and metrics
        """
        device = batch["observation.state"].device
        
        # ==================== Teacher Forward (no grad) ====================
        with torch.no_grad():
            self.teacher.eval()
            teacher_outputs = self.teacher.forward_with_world_model(
                batch=batch,
                noise=noise,
                time=time,
                cur_n_obs_img_steps=cur_n_obs_img_steps,
                cur_n_pred_img_steps=cur_n_pred_img_steps,
                train_gen_expert_only=True,
                gen_out_loss_ratio=gen_out_loss_ratio,
                return_images=False,
            )
        
        # Get teacher's memory state after forward
        teacher_memory = None
        if self.use_memory_distillation and self.config.use_memory:
            if hasattr(self.teacher.model, 'memory_manager') and self.teacher.model.memory_manager is not None:
                # Get memory state from teacher
                # Note: Memory is stored in memory_bank after forward pass
                # We need to extract it from the batch indices
                dataset_indices = batch.get("dataset_idx")
                episode_indices = batch.get("episode_idx")
                if dataset_indices is not None and episode_indices is not None:
                    teacher_memory = self._get_memory_state_for_batch(
                        self.teacher, batch, device
                    )
        
        # ==================== Student Forward ====================
        # Prepare student batch (mask head camera)
        student_batch = self._prepare_student_batch(batch)
        
        # Forward student
        student_outputs = self.student.forward_with_world_model(
            batch=student_batch,
            noise=noise,
            time=time,
            cur_n_obs_img_steps=cur_n_obs_img_steps,
            cur_n_pred_img_steps=cur_n_pred_img_steps,
            train_gen_expert_only=True,
            gen_out_loss_ratio=gen_out_loss_ratio,
            return_images=return_images,
        )
        
        # Get student's memory state
        student_memory = None
        if self.use_memory_distillation and self.config.use_memory:
            if hasattr(self.student.model, 'memory_manager') and self.student.model.memory_manager is not None:
                student_memory = self._get_memory_state_for_batch(
                    self.student, batch, device
                )
        
        # ==================== Compute Losses ====================
        # GT loss from student
        gt_loss = student_outputs["loss"]
        
        # Memory distillation loss
        memory_loss = torch.tensor(0.0, device=device)
        if self.use_memory_distillation and teacher_memory is not None and student_memory is not None:
            memory_loss = self._compute_memory_distillation_loss(teacher_memory, student_memory)
        
        # Combined loss
        # total_loss = gt_loss + memory_loss_weight * memory_loss
        total_loss = gt_loss + self.memory_loss_weight * memory_loss
        
        # ==================== Prepare Output ====================
        loss_dict = {
            "loss": total_loss,
            "gt_loss": gt_loss.detach(),
            "memory_loss": memory_loss.detach() if isinstance(memory_loss, Tensor) else memory_loss,
            "wm_loss": student_outputs.get("wm_loss", gt_loss).detach(),
            "wm_acc_mean": student_outputs.get("wm_acc_mean", torch.tensor(0.0)),
            "wm_acc_tail": student_outputs.get("wm_acc_tail", torch.tensor(0.0)),
            "teacher_wm_acc": teacher_outputs.get("wm_acc_mean", torch.tensor(0.0)),
        }
        
        if return_images:
            if "wm_gt_img" in student_outputs:
                loss_dict["wm_gt_img"] = student_outputs["wm_gt_img"]
            if "wm_pred_img" in student_outputs:
                loss_dict["wm_pred_img"] = student_outputs["wm_pred_img"]
        
        return loss_dict
    
    def _get_memory_state_for_batch(
        self,
        policy: F1_VLA,
        batch: Dict[str, Tensor],
        device: torch.device
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Get current memory state for the batch from a policy's memory manager.
        
        This retrieves the memory state that was just computed/stored
        during the forward pass.
        """
        if not hasattr(policy.model, 'memory_manager') or policy.model.memory_manager is None:
            return None
        
        dtype = next(policy.model.parameters()).dtype
        memory_kv, _, _ = policy._get_memory_state(batch)
        
        return memory_kv
    
    def get_optim_params(self) -> dict:
        """Return only student's trainable parameters for optimizer."""
        return [p for p in self.student.parameters() if p.requires_grad]
    
    @property
    def use_world_model(self) -> bool:
        return self.config.use_world_model
    
    @property
    def model(self):
        """For compatibility with trainer that accesses policy.model."""
        return self.student.model


class StudentOnlyPolicy(nn.Module):
    """
    Control group: Student with GT loss only (no teacher distillation).
    
    This is a wrapper around F1_VLA that masks head camera input,
    training only with ground truth loss. Used for comparison with
    teacher-student distillation.
    
    Args:
        config: F1Config for student
        student_ckpt: Path to pretrained student checkpoint (optional)
        training_args: Training arguments
    """
    
    config_class = F1Config
    
    def __init__(
        self,
        config: F1Config,
        student_ckpt: str = None,
        **kwargs
    ):
        super().__init__()
        self.config = config
        
        # Modify training_args to ensure gen_expert_only mode
        training_args = kwargs.get("training_args")
        if training_args is not None:
            training_args.train_gen_expert_only = True
            training_args.freeze_gen_expert = False
        
        # Create student policy
        logger.info("[StudentOnlyPolicy] Creating student policy (no teacher)...")
        self.student = F1_VLA(config, **kwargs)
        
        # Load student checkpoint if provided
        if student_ckpt is not None:
            logger.info(f"[StudentOnlyPolicy] Loading student from {student_ckpt}")
            self._load_checkpoint(student_ckpt)
        
        # Report trainable parameters
        trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.student.parameters())
        logger.info(f"[StudentOnlyPolicy] Trainable: {trainable:,} / {total:,} parameters")
    
    def _load_checkpoint(self, ckpt_path: str):
        """Load checkpoint into student policy."""
        import os
        from safetensors.torch import load_file
        
        if os.path.isdir(ckpt_path):
            safetensor_path = os.path.join(ckpt_path, "model.safetensors")
            if os.path.exists(safetensor_path):
                state_dict = load_file(safetensor_path)
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                missing, unexpected = self.student.load_state_dict(state_dict, strict=False)
                logger.info(f"[StudentOnlyPolicy] Loaded student from {safetensor_path}")
                if missing:
                    logger.debug(f"[StudentOnlyPolicy] Missing keys: {len(missing)}")
            else:
                logger.warning(f"[StudentOnlyPolicy] No model.safetensors found in {ckpt_path}")
        else:
            state_dict = load_file(ckpt_path)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
            self.student.load_state_dict(state_dict, strict=False)
            logger.info(f"[StudentOnlyPolicy] Loaded student from {ckpt_path}")
    
    def reset(self):
        self.student.reset()
    
    def _prepare_student_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Mask out head camera for student (same as TeacherStudentPolicy)."""
        student_batch = {}
        
        for key, value in batch.items():
            if key == "observation.images.image0":
                student_batch[key] = torch.zeros_like(value)
            elif key == "observation.images.image0_mask":
                student_batch[key] = torch.zeros_like(value, dtype=torch.bool)
            elif key == "observation.images.image0_history":
                student_batch[key] = torch.zeros_like(value)
            else:
                student_batch[key] = value
        
        return student_batch
    
    def forward_with_world_model(
        self,
        batch: Dict[str, Tensor],
        noise: Tensor = None,
        time: Tensor = None,
        cur_n_obs_img_steps: int = None,
        cur_n_pred_img_steps: int = None,
        train_gen_expert_only: bool = True,
        gen_out_loss_ratio: float = 1.0,
        return_images: bool = False,
    ) -> Dict[str, Tensor]:
        """Forward with wrist-only observation and GT loss only."""
        # Prepare student batch (mask head camera)
        student_batch = self._prepare_student_batch(batch)
        
        # Forward student
        outputs = self.student.forward_with_world_model(
            batch=student_batch,
            noise=noise,
            time=time,
            cur_n_obs_img_steps=cur_n_obs_img_steps,
            cur_n_pred_img_steps=cur_n_pred_img_steps,
            train_gen_expert_only=True,
            gen_out_loss_ratio=gen_out_loss_ratio,
            return_images=return_images,
        )
        
        # Add gt_loss and memory_loss keys for compatibility
        outputs["gt_loss"] = outputs["loss"].detach()
        outputs["memory_loss"] = torch.tensor(0.0, device=outputs["loss"].device)
        
        return outputs
    
    def get_optim_params(self) -> dict:
        return [p for p in self.student.parameters() if p.requires_grad]
    
    @property
    def use_world_model(self) -> bool:
        return self.config.use_world_model
    
    @property
    def model(self):
        return self.student.model
