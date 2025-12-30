"""
VAE Embedding Extractor Module

This module provides utilities for extracting VAE embeddings from images,
which are used for computing rewards in Explorer RL training.

Key features:
- Extract embeddings from GT images via VAE encoder
- Extract embeddings from WM predictions (already in embedding space)
- Compute uncertainty from VAE decoder logits
"""

import logging
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VAEEmbeddingExtractor(nn.Module):
    """
    Extracts VAE embeddings from images for reward computation.
    
    The embedding is the quantized latent representation from VQVAE.
    Both GT images and WM predictions use the same embedding space.
    """
    
    def __init__(
        self,
        vae: nn.Module,
        embedding_dim: int = 32,  # z_channels of VQVAE
        pool_method: str = 'mean',  # 'mean', 'max', or 'flatten'
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            vae: VQVAE model instance
            embedding_dim: Dimension of VAE embeddings (z_channels)
            pool_method: How to pool spatial dimensions ('mean', 'max', 'flatten')
        """
        super().__init__()
        self.vae = vae
        self.embedding_dim = embedding_dim
        self.pool_method = pool_method
        
        # Freeze VAE
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
    
    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image to VAE embedding.
        
        Args:
            image: Input image tensor, shape (B, C, H, W) or (B, T, C, H, W)
                   Values should be in range [-1, 1] (VAE convention)
        
        Returns:
            embedding: VAE embedding, shape (B, embed_dim) or (B, T, embed_dim)
        """
        has_time_dim = image.dim() == 5
        
        if has_time_dim:
            B, T, C, H, W = image.shape
            image = image.reshape(B * T, C, H, W)
        
        # Encode to latent space
        # VAE encoder: (B, C, H, W) -> (B, z_channels, h, w)
        z = self.vae.quant_conv(self.vae.encoder(image))
        
        # Pool spatial dimensions
        if self.pool_method == 'mean':
            embedding = z.mean(dim=[2, 3])  # (B, z_channels)
        elif self.pool_method == 'max':
            embedding = z.max(dim=3)[0].max(dim=2)[0]  # (B, z_channels)
        elif self.pool_method == 'flatten':
            embedding = z.flatten(start_dim=1)  # (B, z_channels * h * w)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        if has_time_dim:
            embedding = embedding.reshape(B, T, -1)
        
        return embedding
    
    @torch.no_grad()
    def encode_image_multiscale(self, image: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode an image to multi-scale VAE embeddings (for each resolution).
        
        Args:
            image: Input image tensor, shape (B, C, H, W)
                   Values should be in range [-1, 1]
        
        Returns:
            embeddings: List of embeddings for each scale
        """
        # Get multi-scale quantized indices
        idx_list = self.vae.img_to_idxBl(image)  # List of (B, pn*pn)
        
        # Convert indices to embeddings
        embeddings = []
        for idx in idx_list:
            # idx: (B, pn*pn)
            emb = self.vae.quantize.embedding(idx)  # (B, pn*pn, Cvae)
            emb = emb.mean(dim=1)  # (B, Cvae) - pool tokens
            embeddings.append(emb)
        
        return embeddings
    
    @torch.no_grad()
    def get_embedding_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert VAE token indices to embeddings.
        
        This is useful for WM predictions which output indices.
        
        Args:
            indices: Token indices from WM, shape (B, L) where L is total tokens
        
        Returns:
            embedding: VAE embedding, shape (B, embed_dim)
        """
        # indices: (B, L) where L = sum(pn^2 for pn in patch_nums)
        B, L = indices.shape
        
        # Get embeddings
        emb = self.vae.quantize.embedding(indices)  # (B, L, Cvae)
        
        # Pool
        if self.pool_method == 'mean':
            embedding = emb.mean(dim=1)  # (B, Cvae)
        elif self.pool_method == 'max':
            embedding = emb.max(dim=1)[0]
        elif self.pool_method == 'flatten':
            embedding = emb.flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        return embedding
    
    @torch.no_grad()
    def get_embedding_from_var_input(self, var_input: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding from VAR-style input (continuous embeddings).
        
        This is the format used by the World Model internally.
        
        Args:
            var_input: VAR input embedding, shape (B, L, Cvae)
        
        Returns:
            embedding: Pooled embedding, shape (B, embed_dim)
        """
        # var_input: (B, L, Cvae)
        if self.pool_method == 'mean':
            embedding = var_input.mean(dim=1)  # (B, Cvae)
        elif self.pool_method == 'max':
            embedding = var_input.max(dim=1)[0]
        elif self.pool_method == 'flatten':
            embedding = var_input.flatten(start_dim=1)
        else:
            raise ValueError(f"Unknown pool_method: {self.pool_method}")
        
        return embedding


class UncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty from World Model generation logits.
    
    Uses entropy of the token probability distribution as uncertainty measure.
    """
    
    def __init__(
        self,
        vocab_size: int = 4096,
        temperature: float = 1.0,
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            vocab_size: Size of VAE vocabulary
            temperature: Temperature for softmax (higher = smoother distribution)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of logits distribution.
        
        Args:
            logits: WM output logits, shape (B, L, vocab_size)
        
        Returns:
            entropy: Per-sample entropy, shape (B,)
        """
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Compute probabilities
        probs = F.softmax(scaled_logits, dim=-1)  # (B, L, vocab_size)
        
        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
        
        # Average over tokens
        entropy = entropy.mean(dim=-1)  # (B,)
        
        return entropy
    
    def compute_max_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute maximum entropy across tokens.
        
        Args:
            logits: WM output logits, shape (B, L, vocab_size)
        
        Returns:
            max_entropy: Per-sample max entropy, shape (B,)
        """
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
        
        max_entropy = entropy.max(dim=-1)[0]  # (B,)
        
        return max_entropy
    
    def compute_top_k_entropy(
        self, 
        logits: torch.Tensor, 
        k: int = 100,
    ) -> torch.Tensor:
        """
        Compute entropy using only top-k tokens.
        
        This focuses uncertainty on the most likely tokens.
        
        Args:
            logits: WM output logits, shape (B, L, vocab_size)
            k: Number of top tokens to consider
        
        Returns:
            entropy: Per-sample entropy, shape (B,)
        """
        B, L, V = logits.shape
        
        # Get top-k
        top_k_logits, _ = torch.topk(logits, k, dim=-1)  # (B, L, k)
        
        # Compute entropy on top-k
        scaled_logits = top_k_logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (B, L)
        
        entropy = entropy.mean(dim=-1)  # (B,)
        
        return entropy


def compute_embedding_mse(
    pred_emb: torch.Tensor,
    gt_emb: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute MSE between predicted and ground truth embeddings.
    
    Args:
        pred_emb: Predicted embedding, shape (B, embed_dim) or (B, T, embed_dim)
        gt_emb: Ground truth embedding, same shape as pred_emb
        reduction: 'mean', 'sum', or 'none'
    
    Returns:
        mse: MSE loss, shape depends on reduction
    """
    mse = F.mse_loss(pred_emb, gt_emb, reduction='none')
    
    # Sum over embedding dimension
    mse = mse.sum(dim=-1)  # (B,) or (B, T)
    
    if reduction == 'mean':
        return mse.mean()
    elif reduction == 'sum':
        return mse.sum()
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def compute_embedding_cosine_similarity(
    pred_emb: torch.Tensor,
    gt_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between predicted and ground truth embeddings.
    
    Args:
        pred_emb: Predicted embedding, shape (B, embed_dim)
        gt_emb: Ground truth embedding, shape (B, embed_dim)
    
    Returns:
        similarity: Cosine similarity, shape (B,)
    """
    pred_norm = F.normalize(pred_emb, dim=-1)
    gt_norm = F.normalize(gt_emb, dim=-1)
    
    similarity = (pred_norm * gt_norm).sum(dim=-1)  # (B,)
    
    return similarity


class EmbeddingBuffer:
    """
    Buffer for storing embedding history for Explorer reward computation.
    
    Stores:
    - gt_embeddings: Ground truth image embeddings [emb_{t-L+1}, ..., emb_t, emb_{t+1}]
    - pred_embeddings: WM predicted embeddings [pred_emb_{t-L+2}, ..., pred_emb_t, pred_emb_{t+1}]
    - uncertainties: WM uncertainty values [unc_{t-L+2}, ..., unc_t, unc_{t+1}]
    """
    
    def __init__(
        self,
        max_len: int = 8,
        device: torch.device = None,
    ):
        """
        Initialize the embedding buffer.
        
        Args:
            max_len: Maximum history length
            device: Device to store tensors on
        """
        self.max_len = max_len
        self.device = device or torch.device('cpu')
        
        self.gt_embeddings: List[torch.Tensor] = []
        self.pred_embeddings: List[torch.Tensor] = []
        self.uncertainties: List[torch.Tensor] = []
        self.mse_values: List[torch.Tensor] = []
    
    def reset(self):
        """Clear all buffers."""
        self.gt_embeddings.clear()
        self.pred_embeddings.clear()
        self.uncertainties.clear()
        self.mse_values.clear()
    
    def add(
        self,
        gt_emb: Optional[torch.Tensor] = None,
        pred_emb: Optional[torch.Tensor] = None,
        uncertainty: Optional[torch.Tensor] = None,
        mse: Optional[torch.Tensor] = None,
    ):
        """
        Add new embeddings to buffer.
        
        Args:
            gt_emb: Ground truth embedding for current frame
            pred_emb: Predicted embedding from WM
            uncertainty: Uncertainty value from WM
            mse: Pre-computed MSE between pred and gt
        """
        if gt_emb is not None:
            self.gt_embeddings.append(gt_emb.to(self.device))
            if len(self.gt_embeddings) > self.max_len + 1:  # +1 for L+1 frames
                self.gt_embeddings.pop(0)
        
        if pred_emb is not None:
            self.pred_embeddings.append(pred_emb.to(self.device))
            if len(self.pred_embeddings) > self.max_len:  # L frames
                self.pred_embeddings.pop(0)
        
        if uncertainty is not None:
            self.uncertainties.append(uncertainty.to(self.device))
            if len(self.uncertainties) > self.max_len:  # L frames
                self.uncertainties.pop(0)
        
        if mse is not None:
            self.mse_values.append(mse.to(self.device))
            if len(self.mse_values) > self.max_len:
                self.mse_values.pop(0)
    
    def get_latest(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get the latest values from buffer."""
        return {
            'gt_emb': self.gt_embeddings[-1] if self.gt_embeddings else None,
            'pred_emb': self.pred_embeddings[-1] if self.pred_embeddings else None,
            'uncertainty': self.uncertainties[-1] if self.uncertainties else None,
            'mse': self.mse_values[-1] if self.mse_values else None,
        }
    
    def get_history(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get stacked history tensors."""
        return {
            'gt_emb_history': torch.stack(self.gt_embeddings, dim=1) if self.gt_embeddings else None,
            'pred_emb_history': torch.stack(self.pred_embeddings, dim=1) if self.pred_embeddings else None,
            'uncertainty_history': torch.stack(self.uncertainties, dim=0) if self.uncertainties else None,
            'mse_history': torch.stack(self.mse_values, dim=0) if self.mse_values else None,
        }
    
    def can_compute_improvement_reward(self) -> bool:
        """Check if we have enough history to compute improvement rewards."""
        return len(self.mse_values) >= 2 and len(self.uncertainties) >= 2
    
    def get_improvement_signals(self) -> Dict[str, torch.Tensor]:
        """
        Get improvement signals for reward computation.
        
        Returns:
            dict with:
                - mse_improvement: MSE_{t} - MSE_{t+1} (positive = prediction improved)
                - unc_improvement: unc_{t} - unc_{t+1} (positive = more confident)
        """
        if not self.can_compute_improvement_reward():
            raise ValueError("Not enough history for improvement reward")
        
        mse_improvement = self.mse_values[-2] - self.mse_values[-1]
        unc_improvement = self.uncertainties[-2] - self.uncertainties[-1]
        
        return {
            'mse_improvement': mse_improvement,
            'unc_improvement': unc_improvement,
        }
