#!/bin/bash
# Train No-Memory model on loop trajectory data
# Uses 3 GPUs

set -e

cd /mnt/data2/ty/F1-VLA

# Check if data is ready
TRAIN_COUNT=$(ls RoboTwin/data/loop_trajectory/*.pt 2>/dev/null | wc -l)
TEST_COUNT=$(ls RoboTwin/data/loop_trajectory_test/*.pt 2>/dev/null | wc -l)

echo "=========================================="
echo "Training No-Memory Model on Loop Trajectory"
echo "=========================================="
echo "Training data: $TRAIN_COUNT episodes"
echo "Test data: $TEST_COUNT episodes"
echo "GPUs: 3,4,5"
echo "Config: f1_vla/config/no_memory_head_and_wrist.yaml"
echo "=========================================="

# Wait for test data if not ready
if [ $TEST_COUNT -lt 64 ]; then
    echo "Waiting for test data collection to complete..."
    while [ $(ls RoboTwin/data/loop_trajectory_test/*.pt 2>/dev/null | wc -l) -lt 64 ]; do
        sleep 30
    done
    echo "Test data ready!"
fi

# Start training
CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 --master_port=29502 \
    train_hf.py \
    --config f1_vla/config/no_memory_head_and_wrist.yaml \
    2>&1 | tee logs/train_no_memory_loop_$(date +%Y%m%d_%H%M%S).log
