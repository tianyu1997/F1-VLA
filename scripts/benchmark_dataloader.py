#!/usr/bin/env python3
"""
Benchmark Dataloader Performance for F1-VLA.

Usage:
    python scripts/benchmark_dataloader.py --config-file f1_vla/config/memory_from_f1pretrain.yaml --num-workers 4
"""

import os
import sys
import time
import argparse
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from f1_vla.src.models.configuration_f1 import F1Config
from f1_vla.src.utils.utils import clean_overrides
from f1_vla.src.processors.data_processors.sequential_dataset import (
    create_sequential_mekvm_data, SequentialCollateFn, SequentialBatchSampler
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def benchmark(args):
    config = OmegaConf.load(args.config_file)
    logger.info(f"Loaded config from {args.config_file}")

    # Dummy policy config setup
    policy_config = F1Config.from_pretrained(f"{config.policy.ckpt_path}")
    policy_config.chunk_size = config.policy.get('chunk_size', 4)
    policy_config.max_state_dim = 32 # Dummy default
    policy_config.max_action_dim = 32 # Dummy default
    
    # Training args dummy
    class DummyArgs:
        image_transforms_enabled = True
        image_transforms_max_num_transforms = 3
        image_transforms_random_order = True
        image_transforms_type = ["brightness", "contrast"]
        per_device_train_batch_size = args.batch_size
    
    training_args = DummyArgs()

    # Create Dataset
    logger.info("Creating dataset...")
    (
        dataset,
        image_transforms,
        _, _, _
    ) = create_sequential_mekvm_data(
        policy_config=policy_config,
        dataset_config=config.dataset,
        training_args=training_args,
        stage="stage3_finetune_vla",
    )
    
    logger.info(f"Dataset size: {len(dataset)}")
    
    # Create Sampler
    sampler = SequentialBatchSampler(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle_episodes=True,
        drop_last=False
    )
    
    collate_fn = SequentialCollateFn(policy_config.max_state_dim, policy_config.max_action_dim)
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    logger.info(f"Starting benchmark: workers={args.num_workers}, batch={args.batch_size}, pin={args.pin_memory}")
    
    start_time = time.time()
    steps = 0
    data_time_sum = 0
    iter_start = time.time()
    
    # Warmup
    logger.info("Warming up (5 steps)...")
    iterator = iter(loader)
    for _ in range(5):
        try:
            _ = next(iterator)
        except StopIteration:
            break
            
    logger.info("Benchmarking (50 steps)...")
    iter_start = time.time()
    
    try:
        for i, batch in enumerate(tqdm(loader, total=50)):
            if i >= 50: break
            # Simulate GPU transfer
            if torch.cuda.is_available():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        v = v.cuda(non_blocking=True)
            steps += 1
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        
    total_time = time.time() - iter_start
    avg_speed = steps / total_time if total_time > 0 else 0
    
    logger.info(f"Results:")
    logger.info(f"  Total Steps: {steps}")
    logger.info(f"  Total Time: {total_time:.2f}s")
    logger.info(f"  Speed: {avg_speed:.2f} batches/sec")
    logger.info(f"  Throughput: {avg_speed * args.batch_size:.2f} samples/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    args = parser.parse_args()
    
    benchmark(args)
