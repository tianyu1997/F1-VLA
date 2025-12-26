#!/bin/bash

# F1-VLA Training Script
# Usage: ./train.sh [config_file] [num_gpus] [gpu_ids]

# Default values
CONFIG_FILE="${1:-f1_vla/config/train_config.yaml}"
NUM_GPUS="${2:-2}"
GPU_IDS="${3:-0,1}"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate f1

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_IDS
export TOKENIZERS_PARALLELISM=false

# Create log directory
LOG_DIR="logs"
mkdir -p $LOG_DIR

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

echo "=========================================="
echo "F1-VLA Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $GPU_IDS (num=$NUM_GPUS)"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Run training with torchrun
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_hf.py \
    --config-file $CONFIG_FILE \
    > $LOG_FILE 2>&1 &

# Get PID
PID=$!
echo "Training started with PID: $PID"
echo "PID saved to: ${LOG_DIR}/train_pid.txt"
echo $PID > ${LOG_DIR}/train_pid.txt

# Create latest_log symlink
ln -sf "train_${TIMESTAMP}.log" "${LOG_DIR}/latest_log.log"
echo "Latest log link: ${LOG_DIR}/latest_log.log"

echo ""
echo "To monitor training:"
echo "  tail -f ${LOG_DIR}/latest_log.log"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo ""
