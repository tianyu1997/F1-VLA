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
            # Extract memory KV states from teacher's past_key_values
            past_kv = teacher_outputs.get("past_key_values")
            if past_kv is not None and isinstance(past_kv, dict) and len(past_kv) > 0:
                # Convert dict format to list of tuples and DETACH (teacher needs no gradient)
                teacher_memory = [(past_kv[i]["key_states"].detach(), past_kv[i]["value_states"].detach()) 
                                 for i in range(len(past_kv))]
                logger.debug(f"[TeacherStudent] Teacher memory retrieved: {len(teacher_memory)} layers")
            else:
                logger.warning(f"[TeacherStudent] Teacher memory is None or invalid format: past_kv={past_kv}")
        
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
            # Extract memory KV states from student's past_key_values
            past_kv = student_outputs.get("past_key_values")
            if past_kv is not None and isinstance(past_kv, dict):
                # Convert dict format to list of tuples: {layer_idx: {key_states, value_states}} -> [(k, v), ...]
                student_memory = [(past_kv[i]["key_states"], past_kv[i]["value_states"]) 
                                 for i in range(len(past_kv))]
                logger.debug(f"[TeacherStudent] Student memory retrieved: {len(student_memory)} layers")
            else:
                logger.warning(f"[TeacherStudent] Student memory is None or invalid format")
        
        # ==================== Compute Losses ====================
        # GT loss from student
        gt_loss = student_outputs["loss"]
        
        # Memory distillation loss
        memory_loss = torch.tensor(0.0, device=device)
        if self.use_memory_distillation and teacher_memory is not None and student_memory is not None:
            memory_loss = self._compute_memory_distillation_loss(teacher_memory, student_memory)
            logger.debug(f"[TeacherStudent] Memory loss: {memory_loss.item():.4f}")
        else:
            if self.use_memory_distillation:
                logger.warning(f"[TeacherStudent] Memory distillation enabled but memory not available: "
                             f"teacher_memory={teacher_memory is not None}, student_memory={student_memory is not None}")
        
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


class TeacherStudentSplitGPU(nn.Module):
    """
    Teacher-Student with teacher and student on SEPARATE GPUs.
    
    This avoids OOM by:
    - Teacher stays on a dedicated GPU (e.g., GPU 0), completely frozen
    - Student on another GPU (e.g., GPU 1), trainable
    - Batch data is moved between GPUs as needed
    
    Args:
        config: F1Config for both teacher and student
        teacher_ckpt: Path to teacher checkpoint
        student_ckpt: Path to student checkpoint (optional, will copy from teacher if None)
        memory_loss_weight: Weight for memory distillation loss
        use_memory_distillation: Whether to use memory distillation
        teacher_device: Device for teacher (e.g., 'cuda:0')
        student_device: Device for student (e.g., 'cuda:1')
    """
    
    config_class = F1Config
    
    def __init__(
        self,
        config: F1Config,
        teacher_ckpt: str = None,
        student_ckpt: str = None,
        memory_loss_weight: float = 0.5,
        use_memory_distillation: bool = True,
        teacher_device: str = "cuda:0",
        student_device: str = "cuda:1",
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.memory_loss_weight = memory_loss_weight
        self.use_memory_distillation = use_memory_distillation
        self.teacher_device = torch.device(teacher_device)
        self.student_device = torch.device(student_device)
        
        # Modify training_args for student
        training_args = kwargs.get("training_args")
        if training_args is not None:
            training_args.train_gen_expert_only = True
            training_args.freeze_gen_expert = False
        
        # ==================== Create Teacher on teacher_device ====================
        logger.info(f"[SplitGPU] Creating teacher on {teacher_device}...")
        self.teacher = F1_VLA(config, device=self.teacher_device, **kwargs)
        
        # Move teacher to its device first (before loading checkpoint)
        self.teacher = self.teacher.to(self.teacher_device)
        
        if teacher_ckpt is not None:
            logger.info(f"[SplitGPU] Loading teacher from {teacher_ckpt}")
            self._load_checkpoint(self.teacher, teacher_ckpt, "teacher")
            # Clean any NaN/Inf in memory parameters after loading checkpoint
            self._clean_memory_nan(self.teacher, "teacher")
            # Ensure all modules are on the correct device after checkpoint loading
            self.teacher = self.teacher.to(self.teacher_device)
            # Explicitly move VAE to correct device (it may not be moved by .to())
            if hasattr(self.teacher, 'vae') and self.teacher.vae is not None:
                self.teacher.vae = self.teacher.vae.to(self.teacher_device)
            if hasattr(self.teacher, 'model') and hasattr(self.teacher.model, 'vae') and self.teacher.model.vae is not None:
                self.teacher.model.vae = self.teacher.model.vae.to(self.teacher_device)

        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        
        logger.info(f"[SplitGPU] Teacher frozen and moved to {teacher_device}")
        
        # DEBUG: Check teacher VAE device
        if hasattr(self.teacher, 'vae') and self.teacher.vae is not None:
            vae_device = next(self.teacher.vae.parameters()).device
            logger.info(f"[SplitGPU] DEBUG: self.teacher.vae device: {vae_device}")
        if hasattr(self.teacher, 'model') and hasattr(self.teacher.model, 'vae') and self.teacher.model.vae is not None:
            vae_device = next(self.teacher.model.vae.parameters()).device
            logger.info(f"[SplitGPU] DEBUG: self.teacher.model.vae device: {vae_device}")
        
        # ==================== Create Student on student_device ====================
        logger.info(f"[SplitGPU] Creating student on {student_device}...")
        self.student = F1_VLA(config, device=self.student_device, **kwargs)
        
        # Move student to its device first (before loading checkpoint)
        self.student = self.student.to(self.student_device)
        
        if student_ckpt is not None:
            logger.info(f"[SplitGPU] Loading student from {student_ckpt}")
            self._load_checkpoint(self.student, student_ckpt, "student")
            # Clean any NaN/Inf in memory parameters after loading checkpoint
            self._clean_memory_nan(self.student, "student")
            # Ensure all modules are on the correct device after checkpoint loading
            self.student = self.student.to(self.student_device)
            # Explicitly move VAE to correct device (it may not be moved by .to())
            if hasattr(self.student, 'vae') and self.student.vae is not None:
                self.student.vae = self.student.vae.to(self.student_device)
            if hasattr(self.student, 'model') and hasattr(self.student.model, 'vae') and self.student.model.vae is not None:
                self.student.model.vae = self.student.model.vae.to(self.student_device)
        elif teacher_ckpt is not None:
            logger.info("[SplitGPU] Copying teacher weights to student")
            self._copy_weights_teacher_to_student()
            # Clean any NaN/Inf in memory parameters after copying
            self._clean_memory_nan(self.student, "student")
            self.student = self.student.to(self.student_device)
            # Explicitly move VAE to correct device
            if hasattr(self.student, 'vae') and self.student.vae is not None:
                self.student.vae = self.student.vae.to(self.student_device)
            if hasattr(self.student, 'model') and hasattr(self.student.model, 'vae') and self.student.model.vae is not None:
                self.student.model.vae = self.student.model.vae.to(self.student_device)
        
        logger.info(f"[SplitGPU] Student moved to {student_device}")
        
        # Report parameters
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        student_trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        student_total = sum(p.numel() for p in self.student.parameters())
        logger.info(f"[SplitGPU] Teacher: {teacher_params:,} params (frozen on {teacher_device})")

        logger.info(f"[SplitGPU] Student: {student_trainable:,} / {student_total:,} trainable (on {student_device})")
    
    def _clean_memory_nan(self, policy: "F1_VLA", name: str):
        """Clean NaN/Inf from memory module parameters after checkpoint loading."""
        # Memory is in policy.model.memory_bank (not policy.model.memory!)
        memory = None
        if hasattr(policy, 'model') and hasattr(policy.model, 'memory_bank') and policy.model.memory_bank is not None:
            memory = policy.model.memory_bank
        elif hasattr(policy, 'memory_bank') and policy.memory_bank is not None:
            memory = policy.memory_bank
        
        if memory is not None:
            
            # Clean init_memory parameter
            if hasattr(memory, 'init_memory'):
                init_mem = memory.init_memory.data
                if torch.isnan(init_mem).any() or torch.isinf(init_mem).any():
                    nan_count = torch.isnan(init_mem).sum().item()
                    inf_count = torch.isinf(init_mem).sum().item()
                    logger.warning(f"[SplitGPU] {name} init_memory has NaN/Inf! nan={nan_count}, inf={inf_count}. Reinitializing...")
                    # Reinitialize with small random values
                    nn.init.normal_(memory.init_memory, mean=0.0, std=0.02)
            
            # Clean memory_token parameter
            if hasattr(memory, 'memory_token'):
                mem_token = memory.memory_token.data
                if torch.isnan(mem_token).any() or torch.isinf(mem_token).any():
                    logger.warning(f"[SplitGPU] {name} memory_token has NaN/Inf! Reinitializing...")
                    nn.init.normal_(memory.memory_token, mean=0.0, std=0.02)
            
            # Clean GRU parameters
            if hasattr(memory, 'memory_gru'):
                for param_name, param in memory.memory_gru.named_parameters():
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        logger.warning(f"[SplitGPU] {name} memory_gru.{param_name} has NaN/Inf! Reinitializing...")
                        if 'weight' in param_name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'bias' in param_name:
                            nn.init.zeros_(param.data)
            
            # Clean memory_info_proj
            if hasattr(memory, 'memory_info_proj'):
                for param_name, param in memory.memory_info_proj.named_parameters():
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        logger.warning(f"[SplitGPU] {name} memory_info_proj.{param_name} has NaN/Inf! Reinitializing...")
                        if 'weight' in param_name:
                            nn.init.xavier_uniform_(param.data)
                        elif 'bias' in param_name:
                            nn.init.zeros_(param.data)
            
            # Clear memory bank to start fresh
            memory.clear_memory_bank()
            logger.info(f"[SplitGPU] {name} memory cleaned and memory bank cleared")
        else:
            logger.warning(f"[SplitGPU] {name} memory module not found! Cannot clean NaN.")
    def _load_checkpoint(self, policy: F1_VLA, ckpt_path: str, name: str):
        """Load checkpoint into policy."""
        import os
        from safetensors.torch import load_file
        
        # Determine the target device from the policy's current device
        device = next(policy.parameters()).device
        
        if os.path.isdir(ckpt_path):
            safetensor_path = os.path.join(ckpt_path, "model.safetensors")
            if os.path.exists(safetensor_path):
                # Load state dict with specified device
                state_dict = load_file(safetensor_path, device=str(device))
                state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                missing, unexpected = policy.load_state_dict(state_dict, strict=False)
                logger.info(f"[SplitGPU] Loaded {name} from {safetensor_path} to {device}")
    
    def _copy_weights_teacher_to_student(self):
        """Copy teacher weights to student (on CPU to avoid OOM)."""
        # Move both to CPU temporarily for copy
        teacher_state = {k: v.cpu() for k, v in self.teacher.state_dict().items()}
        missing, unexpected = self.student.load_state_dict(teacher_state, strict=False)
        logger.info(f"[SplitGPU] Copied {len(teacher_state)} params from teacher to student")
    
    def reset(self):
        self.teacher.reset()
        self.student.reset()
    
    def to(self, *args, **kwargs):
        """Override to() to prevent Trainer from moving models to wrong device.
        
        In SplitGPU mode, teacher and student are on their designated devices.
        We should NOT move them when Trainer calls .to(device).
        """
        # Don't move teacher and student - they're already on correct devices
        # Only move other attributes if needed
        logger.info(f"[SplitGPU] to() called with args={args}, kwargs={kwargs} - IGNORING for teacher/student")
        return self
    
    def cuda(self, device=None):
        """Override cuda() to prevent Trainer from moving models."""
        logger.info(f"[SplitGPU] cuda() called with device={device} - IGNORING for teacher/student")
        return self

    def _move_batch_to_device(self, batch: Dict[str, Tensor], device: torch.device) -> Dict[str, Tensor]:
        """Move batch tensors to specified device."""
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
    
    def _prepare_student_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Replace head camera (image0) with wrist camera (image1) for student.
        
        Instead of masking or removing image0, we copy image1 (wrist) data to image0 position.
        This way the student uses wrist camera for both image0 and image1 slots.
        
        Mapping:
        - image0 (head) -> replaced with image1 (wrist)
        - image0_history -> replaced with image1_history (if exists)
        - image0_target -> replaced with image1_target (if exists)
        - image0_mask -> set to True (valid)
        """
        # Debug: log input batch keys
        logger.info(f"[SplitGPU] _prepare_student_batch input keys: {list(batch.keys())}")
        
        student_batch = {}
        for key, value in batch.items():
            if key == "observation.images.image0":
                # Replace head camera with wrist camera
                wrist_key = "observation.images.image1"
                if wrist_key in batch:
                    student_batch[key] = batch[wrist_key].clone()
                    logger.info(f"[SplitGPU] Replaced {key} with {wrist_key}")
                else:
                    student_batch[key] = value  # Fallback to original
                    logger.warning(f"[SplitGPU] {wrist_key} not found, using original {key}")
            elif key == "observation.images.image0_mask":
                # Mark as valid since we have real wrist data
                student_batch[key] = torch.ones_like(value, dtype=torch.bool)
            elif key == "observation.images.image0_history":
                # Replace head history with wrist history
                wrist_key = "observation.images.image1_history"
                if wrist_key in batch:
                    student_batch[key] = batch[wrist_key].clone()
                else:
                    student_batch[key] = value  # Fallback
            elif key == "observation.images.image0_target":
                # Replace head target with wrist target
                wrist_key = "observation.images.image1_target"
                if wrist_key in batch:
                    student_batch[key] = batch[wrist_key].clone()
                else:
                    student_batch[key] = value  # Fallback
            else:
                student_batch[key] = value
        
        # Debug: log output batch keys
        logger.info(f"[SplitGPU] _prepare_student_batch output keys: {list(student_batch.keys())}")
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
        """
        Forward with teacher and student on separate GPUs.
        
        IMPORTANT: Teacher memory is maintained across frames within an episode (same as training).
        - frame_idx=0: Uses init_memory (learnable parameter)
        - frame_idx>0: Uses memory from previous frame (stored in memory_bank)
        - After forward: Memory is updated via GRU and stored for next frame
        
        Do NOT clear memory_bank before each forward - that would break temporal consistency!
        """
        # ==================== Teacher Forward (on teacher_device) ====================
        # NOTE: Do NOT clear teacher memory bank! Teacher needs memory state from previous frames.
        # Memory is updated inside forward_with_world_model and stored for next frame.
        
        teacher_batch = self._move_batch_to_device(batch, self.teacher_device)
        teacher_noise = noise.to(self.teacher_device) if noise is not None else None
        teacher_time = time.to(self.teacher_device) if time is not None else None
        
        with torch.no_grad():
            self.teacher.eval()
            
            teacher_outputs = self.teacher.forward_with_world_model(
                batch=teacher_batch,
                noise=teacher_noise,
                time=teacher_time,
                cur_n_obs_img_steps=cur_n_obs_img_steps,
                cur_n_pred_img_steps=cur_n_pred_img_steps,
                train_gen_expert_only=True,
                gen_out_loss_ratio=gen_out_loss_ratio,
                return_images=False,
            )
        
        # DEBUG: Check teacher outputs for NaN
        teacher_loss = teacher_outputs.get("loss", torch.tensor(0.0))
        if torch.isnan(teacher_loss).any():
            logger.error(f"[SplitGPU] Teacher output loss is NaN!")
            logger.error(f"  Teacher batch keys: {list(teacher_batch.keys())}")
            for key in ["observation.images.image0", "observation.images.image0_history"]:
                if key in teacher_batch:
                    t = teacher_batch[key]
                    logger.error(f"  {key}: shape={t.shape}, min={t.min():.3f}, max={t.max():.3f}, nan={torch.isnan(t).sum()}")
        
        # Extract teacher memory (move to student device)
        teacher_memory = None
        if self.use_memory_distillation and self.config.use_memory:
            past_kv = teacher_outputs.get("past_key_values")
            if past_kv is not None and isinstance(past_kv, dict) and len(past_kv) > 0:
                teacher_memory = [
                    (past_kv[i]["key_states"].detach().to(self.student_device),
                     past_kv[i]["value_states"].detach().to(self.student_device))
                    for i in range(len(past_kv))
                ]
        
        # Save teacher accuracy before clearing
        teacher_wm_acc = teacher_outputs.get("wm_acc_mean", torch.tensor(0.0))
        
        # Clear teacher outputs to free GPU memory (except what we need)
        del teacher_outputs
        torch.cuda.empty_cache()
        
        # ==================== Student Forward (on student_device) ====================
        student_batch = self._move_batch_to_device(batch, self.student_device)
        student_batch = self._prepare_student_batch(student_batch)
        student_noise = noise.to(self.student_device) if noise is not None else None
        student_time = time.to(self.student_device) if time is not None else None
        
        # Free teacher batch memory
        del teacher_batch, teacher_noise, teacher_time
        torch.cuda.empty_cache()
        
        student_outputs = self.student.forward_with_world_model(
            batch=student_batch,
            noise=student_noise,
            time=student_time,
            cur_n_obs_img_steps=cur_n_obs_img_steps,
            cur_n_pred_img_steps=cur_n_pred_img_steps,
            train_gen_expert_only=True,
            gen_out_loss_ratio=gen_out_loss_ratio,
            return_images=return_images,
        )
        
        # Get student memory
        student_memory = None
        if self.use_memory_distillation and self.config.use_memory:
            past_kv = student_outputs.get("past_key_values")
            if past_kv is not None and isinstance(past_kv, dict) and len(past_kv) > 0:
                student_memory = [(past_kv[i]["key_states"], past_kv[i]["value_states"]) 
                                 for i in range(len(past_kv))]
        
        # ==================== Compute Losses (on student_device) ====================
        gt_loss = student_outputs["loss"]
        
        # Check for nan in gt_loss
        if torch.isnan(gt_loss):
            logger.error(f"[SplitGPU] gt_loss is NaN!")
            # Print student outputs for debugging
            for key, val in student_outputs.items():
                if isinstance(val, torch.Tensor):
                    logger.error(f"  {key}: {val.item() if val.numel() == 1 else 'tensor'}, isnan={torch.isnan(val).any()}")
        
        # Memory distillation loss
        if teacher_memory is not None and student_memory is not None:
            memory_loss = self._compute_memory_loss(teacher_memory, student_memory)
            if torch.isnan(memory_loss):
                logger.error(f"[SplitGPU] memory_loss is NaN!")
        else:
            memory_loss = torch.tensor(0.0, device=self.student_device)
            if self.use_memory_distillation:
                logger.warning(f"[SplitGPU] Memory not available: teacher={teacher_memory is not None}, student={student_memory is not None}")
        
        # Combined loss
        combined_loss = gt_loss + self.memory_loss_weight * memory_loss
        
        # Move loss to trainer's expected device (cuda:0 when using split GPU)
        # Trainer expects loss on its original device
        trainer_device = torch.device("cuda:0")
        
        return {
            "loss": combined_loss.to(trainer_device),
            "gt_loss": gt_loss.detach().to(trainer_device),
            "memory_loss": memory_loss.detach().to(trainer_device),
            "wm_out_loss": student_outputs.get("wm_out_loss", gt_loss).detach().to(trainer_device),
            "wm_loss": student_outputs.get("wm_loss", gt_loss).detach().to(trainer_device),
            "wm_acc_mean": student_outputs.get("wm_acc_mean", torch.tensor(0.0)).to(trainer_device),
            "wm_acc_tail": student_outputs.get("wm_acc_tail", torch.tensor(0.0)).to(trainer_device),
            "teacher_wm_acc": teacher_wm_acc.to(trainer_device),
        }
    
    def _compute_memory_loss(self, teacher_memory, student_memory):
        """Compute MSE loss between teacher and student memory states."""
        total_loss = 0.0
        num_layers = len(teacher_memory)
        num_valid_layers = 0
        
        for layer_idx in range(num_layers):
            teacher_k, teacher_v = teacher_memory[layer_idx]
            student_k, student_v = student_memory[layer_idx]
            
            # Check for NaN/Inf in teacher or student memory before computing loss
            if torch.isnan(teacher_k).any() or torch.isinf(teacher_k).any():
                logger.warning(f"[SplitGPU] teacher_k has NaN/Inf at layer {layer_idx}, skipping this layer")
                continue
            if torch.isnan(teacher_v).any() or torch.isinf(teacher_v).any():
                logger.warning(f"[SplitGPU] teacher_v has NaN/Inf at layer {layer_idx}, skipping this layer")
                continue
            if torch.isnan(student_k).any() or torch.isinf(student_k).any():
                logger.warning(f"[SplitGPU] student_k has NaN/Inf at layer {layer_idx}, skipping this layer")
                continue
            if torch.isnan(student_v).any() or torch.isinf(student_v).any():
                logger.warning(f"[SplitGPU] student_v has NaN/Inf at layer {layer_idx}, skipping this layer")
                continue
            
            k_loss = F.mse_loss(student_k, teacher_k)
            v_loss = F.mse_loss(student_v, teacher_v)
            
            # Check computed losses for NaN
            if torch.isnan(k_loss).any() or torch.isinf(k_loss).any():
                logger.warning(f"[SplitGPU] k_loss is NaN/Inf at layer {layer_idx}, skipping")
                continue
            if torch.isnan(v_loss).any() or torch.isinf(v_loss).any():
                logger.warning(f"[SplitGPU] v_loss is NaN/Inf at layer {layer_idx}, skipping")
                continue
            
            total_loss = total_loss + k_loss + v_loss
            num_valid_layers += 1
        
        # If no valid layers, return zero loss
        if num_valid_layers == 0:
            logger.error(f"[SplitGPU] No valid layers for memory loss computation!")
            return torch.tensor(0.0, device=student_memory[0][0].device)
        
        return total_loss / (num_valid_layers * 2)
    
    def get_optim_params(self):
        """Return only student's trainable parameters."""
        return [p for p in self.student.parameters() if p.requires_grad]
    
    @property
    def use_world_model(self):
        return self.config.use_world_model
    
    @property
    def model(self):
        return self.student.model
    
    def parameters(self, recurse=True):
        """Only return student parameters for optimizer."""
        return self.student.parameters(recurse=recurse)
