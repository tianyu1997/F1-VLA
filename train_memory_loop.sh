#!/bin/bash
# Train Memory model on loop trajectory data
# Uses 3 GPUs

set -e

cd /mnt/data2/ty/F1-VLA

# Check if data is ready
TRAIN_COUNT=$(ls RoboTwin/data/loop_trajectory/*.pt 2>/dev/null | wc -l)
TEST_COUNT=$(ls RoboTwin/data/loop_trajectory_test/*.pt 2>/dev/null | wc -l)

echo "=========================================="
echo "Training Memory Model on Loop Trajectory"
echo "=========================================="
echo "Training data: $TRAIN_COUNT episodes"
echo "Test data: $TEST_COUNT episodes"
echo "GPUs: 0,1,2"
echo "Config: f1_vla/config/memory_bptt.yaml"
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
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master_port=29501 \
    train_hf.py \
    --config f1_vla/config/memory_bptt.yaml \
    2>&1 | tee logs/train_memory_loop_$(date +%Y%m%d_%H%M%S).log
