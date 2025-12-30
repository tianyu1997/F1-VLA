#!/usr/bin/env python3
"""
Unit tests for VAE embedding extraction.

Tests:
1. Embedding extraction from images
2. Uncertainty computation
3. MSE computation
4. Embedding buffer operations
"""

import os
import sys
import torch
import torch.nn as nn

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
f1_vla_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, f1_vla_dir)


class MockVAE(nn.Module):
    """Mock VAE for testing without loading full model."""
    
    def __init__(self, z_channels=32, vocab_size=4096):
        super().__init__()
        self.Cvae = z_channels
        self.V = vocab_size
        
        # Mock encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, z_channels, 4, 2, 1),
        )
        
        # Mock quant_conv
        self.quant_conv = nn.Conv2d(z_channels, z_channels, 3, 1, 1)
        
        # Mock quantizer
        self.quantize = MockQuantizer(vocab_size, z_channels)
    
    def img_to_idxBl(self, image):
        """Mock img_to_idxBl for testing."""
        B = image.shape[0]
        # Return fake indices for different scales
        patch_nums = [1, 2, 3, 4]
        return [torch.randint(0, self.V, (B, pn*pn)) for pn in patch_nums]


class MockQuantizer(nn.Module):
    """Mock quantizer for testing."""
    
    def __init__(self, vocab_size, z_channels):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, z_channels)


def test_vae_embedding_extractor_creation():
    """Test VAEEmbeddingExtractor creation."""
    from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
    
    vae = MockVAE()
    extractor = VAEEmbeddingExtractor(vae, embedding_dim=32)
    
    assert extractor.embedding_dim == 32
    assert extractor.pool_method == 'mean'
    print("  ✓ VAEEmbeddingExtractor created successfully")


def test_encode_image():
    """Test image encoding to embeddings."""
    from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
    
    vae = MockVAE(z_channels=32)
    extractor = VAEEmbeddingExtractor(vae, embedding_dim=32)
    
    # Test single batch
    image = torch.randn(2, 3, 64, 64)  # (B, C, H, W)
    embedding = extractor.encode_image(image)
    
    assert embedding.shape == (2, 32)  # (B, embed_dim)
    print("  ✓ encode_image works for single batch")
    
    # Test with time dimension
    image_seq = torch.randn(2, 4, 3, 64, 64)  # (B, T, C, H, W)
    embedding_seq = extractor.encode_image(image_seq)
    
    assert embedding_seq.shape == (2, 4, 32)  # (B, T, embed_dim)
    print("  ✓ encode_image works for sequence")


def test_get_embedding_from_indices():
    """Test embedding extraction from indices."""
    from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
    
    vae = MockVAE(z_channels=32, vocab_size=4096)
    extractor = VAEEmbeddingExtractor(vae, embedding_dim=32)
    
    # Create fake indices
    indices = torch.randint(0, 4096, (2, 680))  # (B, L)
    embedding = extractor.get_embedding_from_indices(indices)
    
    assert embedding.shape == (2, 32)
    print("  ✓ get_embedding_from_indices works")


def test_get_embedding_from_var_input():
    """Test embedding extraction from VAR input."""
    from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
    
    vae = MockVAE(z_channels=32)
    extractor = VAEEmbeddingExtractor(vae, embedding_dim=32)
    
    # Create fake VAR input
    var_input = torch.randn(2, 680, 32)  # (B, L, Cvae)
    embedding = extractor.get_embedding_from_var_input(var_input)
    
    assert embedding.shape == (2, 32)
    print("  ✓ get_embedding_from_var_input works")


def test_uncertainty_estimator():
    """Test uncertainty estimation from logits."""
    from f1_vla.src.models.vae_embedding import UncertaintyEstimator
    
    estimator = UncertaintyEstimator(vocab_size=4096)
    
    # Create fake logits
    logits = torch.randn(2, 100, 4096)  # (B, L, vocab_size)
    
    # Test entropy
    entropy = estimator.compute_entropy(logits)
    assert entropy.shape == (2,)
    assert (entropy >= 0).all()
    print("  ✓ compute_entropy works")
    
    # Test max entropy
    max_entropy = estimator.compute_max_entropy(logits)
    assert max_entropy.shape == (2,)
    assert (max_entropy >= entropy).all()  # max >= mean
    print("  ✓ compute_max_entropy works")
    
    # Test top-k entropy
    topk_entropy = estimator.compute_top_k_entropy(logits, k=100)
    assert topk_entropy.shape == (2,)
    print("  ✓ compute_top_k_entropy works")


def test_compute_embedding_mse():
    """Test MSE computation between embeddings."""
    from f1_vla.src.models.vae_embedding import compute_embedding_mse
    
    pred_emb = torch.randn(2, 32)
    gt_emb = torch.randn(2, 32)
    
    # Test different reductions
    mse_mean = compute_embedding_mse(pred_emb, gt_emb, reduction='mean')
    assert mse_mean.dim() == 0
    print("  ✓ compute_embedding_mse with mean reduction works")
    
    mse_none = compute_embedding_mse(pred_emb, gt_emb, reduction='none')
    assert mse_none.shape == (2,)
    print("  ✓ compute_embedding_mse with no reduction works")


def test_compute_embedding_cosine_similarity():
    """Test cosine similarity computation."""
    from f1_vla.src.models.vae_embedding import compute_embedding_cosine_similarity
    
    pred_emb = torch.randn(2, 32)
    gt_emb = pred_emb.clone()  # Same embedding
    
    similarity = compute_embedding_cosine_similarity(pred_emb, gt_emb)
    assert similarity.shape == (2,)
    assert torch.allclose(similarity, torch.ones(2), atol=1e-5)  # Should be 1.0
    print("  ✓ compute_embedding_cosine_similarity works")


def test_embedding_buffer():
    """Test EmbeddingBuffer operations."""
    from f1_vla.src.models.vae_embedding import EmbeddingBuffer
    
    buffer = EmbeddingBuffer(max_len=4)
    
    # Add some data
    for i in range(6):
        buffer.add(
            gt_emb=torch.randn(2, 32),
            pred_emb=torch.randn(2, 32),
            uncertainty=torch.rand(2),
            mse=torch.rand(2),
        )
    
    # Check max length is respected
    assert len(buffer.gt_embeddings) == 5  # max_len + 1
    assert len(buffer.pred_embeddings) == 4  # max_len
    assert len(buffer.uncertainties) == 4  # max_len
    print("  ✓ EmbeddingBuffer respects max length")
    
    # Test get_latest
    latest = buffer.get_latest()
    assert latest['gt_emb'] is not None
    assert latest['pred_emb'] is not None
    print("  ✓ EmbeddingBuffer get_latest works")
    
    # Test get_history
    history = buffer.get_history()
    assert history['gt_emb_history'].shape == (2, 5, 32)
    assert history['pred_emb_history'].shape == (2, 4, 32)
    print("  ✓ EmbeddingBuffer get_history works")
    
    # Test improvement signals
    assert buffer.can_compute_improvement_reward()
    signals = buffer.get_improvement_signals()
    assert signals['mse_improvement'].shape == (2,)
    assert signals['unc_improvement'].shape == (2,)
    print("  ✓ EmbeddingBuffer improvement signals work")
    
    # Test reset
    buffer.reset()
    assert len(buffer.gt_embeddings) == 0
    print("  ✓ EmbeddingBuffer reset works")


def test_pool_methods():
    """Test different pooling methods."""
    from f1_vla.src.models.vae_embedding import VAEEmbeddingExtractor
    
    vae = MockVAE(z_channels=32)
    
    # Test mean pooling
    extractor_mean = VAEEmbeddingExtractor(vae, pool_method='mean')
    image = torch.randn(2, 3, 64, 64)
    emb_mean = extractor_mean.encode_image(image)
    assert emb_mean.shape == (2, 32)
    print("  ✓ mean pooling works")
    
    # Test max pooling
    extractor_max = VAEEmbeddingExtractor(vae, pool_method='max')
    emb_max = extractor_max.encode_image(image)
    assert emb_max.shape == (2, 32)
    print("  ✓ max pooling works")


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("VAE Embedding Extractor Unit Tests")
    print("=" * 60)
    
    print("\n[Test 1] VAEEmbeddingExtractor creation...")
    test_vae_embedding_extractor_creation()
    
    print("\n[Test 2] Encode image...")
    test_encode_image()
    
    print("\n[Test 3] Get embedding from indices...")
    test_get_embedding_from_indices()
    
    print("\n[Test 4] Get embedding from VAR input...")
    test_get_embedding_from_var_input()
    
    print("\n[Test 5] Uncertainty estimator...")
    test_uncertainty_estimator()
    
    print("\n[Test 6] Compute embedding MSE...")
    test_compute_embedding_mse()
    
    print("\n[Test 7] Compute cosine similarity...")
    test_compute_embedding_cosine_similarity()
    
    print("\n[Test 8] Embedding buffer...")
    test_embedding_buffer()
    
    print("\n[Test 9] Pool methods...")
    test_pool_methods()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_all_tests()
