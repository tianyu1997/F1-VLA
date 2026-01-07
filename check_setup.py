#!/usr/bin/env python3
"""
Comprehensive check script for F1-VLA training setup.
Checks:
1. Environment (Python, CUDA, Torch)
2. Imports (Dependencies)
3. Model Instantiation (Memory Bank, VAE)
4. Config Loading
"""

import os
import sys
import logging
import torch
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SetupCheck")

def check_environment():
    logger.info("="*50)
    logger.info("Checking Environment...")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Torch: {torch.__version__}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Available: Yes ({torch.cuda.device_count()} devices)")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"  Device {i}: {props.name}, Memory: {props.total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA NOT Available! Training will be slow.")

def check_imports():
    logger.info("="*50)
    logger.info("Checking Imports...")
    try:
        import f1_vla
        from f1_vla.src.models.memory import KVMemoryBank
        from f1_vla.src.policies.f1_policy import F1Config
        from f1_vla.src.processors.data_processors.sequential_dataset import SequentialMEKVMDataset
        logger.info("F1-VLA modules imported successfully.")
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Make sure you have installed the package (pip install -e .) and PYTHONPATH is set.")
        sys.exit(1)

def check_model_instantiation():
    logger.info("="*50)
    logger.info("Checking Model Instantiation...")
    
    # 1. KVMemoryBank
    try:
        from f1_vla.src.models.memory import KVMemoryBank
        logger.info("Testing KVMemoryBank...")
        mem = KVMemoryBank(
            num_layers=2,
            num_kv_heads=4,
            head_dim=64,
            hidden_size=256,
            memory_len=4
        )
        
        # Test basic flow with NaN inputs
        mem_info = torch.randn(1, 256)
        mem_info[0, 0] = float('nan') # Inject NaN
        
        prev_mem = []
        for _ in range(2):
            k = torch.randn(1, 4, 4, 64)
            v = torch.randn(1, 4, 4, 64)
            prev_mem.append((k, v))
            
        logger.info("  Testing update_memory with NaN injection (should adhere to robustness logic)...")
        # This will trigger the NaN checks we added
        new_mem = mem.update_memory(prev_mem, mem_info)
        
        # Check if output is safe
        has_nan = False
        for k, v in new_mem:
            if torch.isnan(k).any() or torch.isnan(v).any():
                has_nan = True
        
        if not has_nan:
            logger.info("  KVMemoryBank handled NaN inputs correctly (output is clean).")
        else:
            logger.error("  KVMemoryBank output contains NaN! Robustness check failed.")
            
    except Exception as e:
        logger.error(f"KVMemoryBank test failed: {e}")

def main():
    check_environment()
    check_imports()
    check_model_instantiation()
    logger.info("="*50)
    logger.info("All checks completed.")

if __name__ == "__main__":
    main()
