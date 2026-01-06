#!/usr/bin/env python3
"""
Estimate memory impact of unfreezing VAE decoder.
"""

import torch
from f1_vla.src.models.wm.vqvae import VQVAE


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_memory(num_params, dtype=torch.float32):
    """Estimate memory usage for parameters."""
    bytes_per_param = 4 if dtype == torch.float32 else 2  # bf16/fp16 = 2 bytes
    # Parameters + Gradients + Optimizer states (Adam: 2x for momentum + variance)
    memory_gb = (num_params * bytes_per_param * 4) / (1024**3)  # 4x = params + grad + 2 optimizer states
    return memory_gb


def main():
    print("\n" + "="*70)
    print("VAE Memory Usage Estimation")
    print("="*70)
    
    # Test 1: Frozen VAE (current state)
    print("\n[1] Frozen VAE (test_mode=True)")
    vae_frozen = VQVAE(test_mode=True, freeze_encoder=True)
    total, trainable = count_parameters(vae_frozen)
    memory = estimate_memory(trainable, torch.bfloat16)
    
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Estimated additional memory: {memory:.2f} GB (with bf16 + Adam)")
    
    # Test 2: Decoder unfrozen
    print("\n[2] Decoder Unfrozen (test_mode=False, freeze_encoder=True)")
    vae_partial = VQVAE(test_mode=False, freeze_encoder=True)
    total, trainable = count_parameters(vae_partial)
    memory = estimate_memory(trainable, torch.bfloat16)
    
    decoder_params = sum(p.numel() for p in vae_partial.decoder.parameters())
    post_conv_params = sum(p.numel() for p in vae_partial.post_quant_conv.parameters())
    
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"    - Decoder: {decoder_params:,}")
    print(f"    - Post-quant conv: {post_conv_params:,}")
    print(f"  Estimated additional memory: {memory:.2f} GB (with bf16 + Adam)")
    
    # Memory breakdown
    print("\n" + "="*70)
    print("Memory Breakdown (bf16 + Adam optimizer):")
    print("="*70)
    params_mem = trainable * 2 / (1024**3)  # bf16 = 2 bytes
    grads_mem = trainable * 2 / (1024**3)
    adam_mem = trainable * 2 * 2 / (1024**3)  # 2 states (momentum + variance) * 2 bytes
    
    print(f"  Parameters (bf16):        {params_mem:.3f} GB")
    print(f"  Gradients (bf16):         {grads_mem:.3f} GB")
    print(f"  Adam states (fp32):       {adam_mem:.3f} GB")
    print(f"  {'='*40}")
    print(f"  TOTAL:                    {params_mem + grads_mem + adam_mem:.3f} GB")
    
    # Current GPU status
    print("\n" + "="*70)
    print("Current GPU Memory Status:")
    print("="*70)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            print(f"  GPU {i}: {props.name}")
            print(f"    Total: {total_mem:.1f} GB | Allocated: {allocated:.1f} GB | Reserved: {reserved:.1f} GB")
    else:
        print("  CUDA not available")
    
    # Recommendation
    print("\n" + "="*70)
    print("Recommendation:")
    print("="*70)
    additional_mem = params_mem + grads_mem + adam_mem
    print(f"  Unfreezing VAE decoder will add ~{additional_mem:.2f} GB per GPU")
    print(f"  Current usage: ~47 GB per GPU (GPU 1-4)")
    print(f"  Estimated new usage: ~{47 + additional_mem:.1f} GB per GPU")
    
    if 47 + additional_mem < 48:
        print(f"  ✓ Should fit in 49GB A6000 memory")
    else:
        print(f"  ✗ May exceed 49GB A6000 memory!")
        print(f"  Consider:")
        print(f"    - Reduce batch_size from 2 to 1")
        print(f"    - Increase gradient_accumulation_steps from 4 to 8")
        print(f"    - Or keep effective batch size = 1 * 8 = 8")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
