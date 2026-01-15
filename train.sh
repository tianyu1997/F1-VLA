#!/bin/bash

# F1-VLA Training Script
# Supports both manual GPU selection and auto-detection of free GPUs
#
# Usage:
#   ./train.sh [OPTIONS]
#
# Options:
#   -c, --config FILE     Config file (default: f1_vla/config/train_config.yaml)
#   -g, --gpus IDS        GPU IDs, comma-separated (e.g., "0,1,2")
#   -n, --num-gpus N      Number of GPUs (used with manual mode)
#   -a, --auto            Auto-detect free GPUs (memory < 2GB)
#   -m, --max-gpus N      Max GPUs to use in auto mode (default: 4)
#   -r, --resume PATH     Resume from checkpoint (path or name like "checkpoint-episode-15500")
#   -p, --port PORT       Master port for distributed training (default: 29500)
#   -h, --help            Show this help message
#
# Examples:
#   ./train.sh -c config.yaml -g 0,1           # Manual: use GPU 0,1
#   ./train.sh -a -c config.yaml               # Auto: detect free GPUs
#   ./train.sh -a -m 2 -c config.yaml          # Auto: use max 2 free GPUs
#   ./train.sh -a -c config.yaml -r checkpoint-episode-15500  # Resume training

set -e

# ============================================
# Default values
# ============================================
CONFIG_FILE="/mnt/data2/ty/F1-VLA/f1_vla/config/no_memory_head_and_wrist.yaml"
GPU_IDS=""
NUM_GPUS=""
AUTO_MODE=false
MAX_GPUS=4
MASTER_PORT=29500
# Default: Resume from checkpoint-episode-10000 with VAE decoder unfrozen
# RESUME_CKPT="/mnt/data2/ty/F1-VLA/outputs/memory_from_f1pretrain/checkpoint-episode-4794" 
RESUME_CKPT=""
MEMORY_THRESHOLD=1000  # MB

# ============================================
# Parse arguments
# ============================================
show_help() {
    head -25 "$0" | tail -22
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -g|--gpus)
            GPU_IDS="$2"
            shift 2
            ;;
        -n|--num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -a|--auto)
            AUTO_MODE=true
            shift
            ;;
        -m|--max-gpus)
            MAX_GPUS="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CKPT="$2"
            shift 2
            ;;
        -p|--port)
            MASTER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage"
            exit 1
            ;;
    esac
done

# ============================================
# Change to project directory
# ============================================
cd /mnt/data2/ty/F1-VLA

# ============================================
# GPU Selection
# ============================================
if [ "$AUTO_MODE" = true ]; then
    echo "Auto-detecting free GPUs (memory < ${MEMORY_THRESHOLD}MB)..."
    
    FREE_GPUS=""
    GPU_COUNT=0
    
    while IFS=, read -r gpu_id mem_used; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ' | sed 's/MiB//')
        
        if [ "$mem_used" -lt "$MEMORY_THRESHOLD" ] && [ "$GPU_COUNT" -lt "$MAX_GPUS" ]; then
            if [ -z "$FREE_GPUS" ]; then
                FREE_GPUS="$gpu_id"
            else
                FREE_GPUS="$FREE_GPUS,$gpu_id"
            fi
            GPU_COUNT=$((GPU_COUNT + 1))
            echo "  GPU $gpu_id: ${mem_used}MB (FREE)"
        else
            echo "  GPU $gpu_id: ${mem_used}MB (busy or skipped)"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    
    if [ -z "$FREE_GPUS" ]; then
        echo "ERROR: No free GPUs available!"
        exit 1
    fi
    
    GPU_IDS="$FREE_GPUS"
    NUM_GPUS="$GPU_COUNT"
else
    # Manual mode
    if [ -z "$GPU_IDS" ]; then
        GPU_IDS="0,1"
    fi
    if [ -z "$NUM_GPUS" ]; then
        # Count GPUs from GPU_IDS
        NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
    fi
fi

# ============================================
# Setup environment
# ============================================
source ~/.bashrc
# Activate env, fall back to sourcing conda.sh if direct activation fails
conda activate f1 2>/dev/null || { source ~/miniconda3/etc/profile.d/conda.sh && conda activate f1; }

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

# ============================================
# Create log directory and file
# ============================================
LOG_DIR="logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# ============================================
# Print training info
# ============================================
echo ""
echo "=========================================="
echo "F1-VLA Training (torchrun)"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPUs: $GPU_IDS ($NUM_GPUS GPUs)"
echo "Mode: $([ "$AUTO_MODE" = true ] && echo "Auto-detect" || echo "Manual")"
if [ -n "$RESUME_CKPT" ]; then
echo "Resume: $RESUME_CKPT"
fi
echo "Master port: $MASTER_PORT"
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

# ============================================
# Build extra arguments
# ============================================
EXTRA_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
    # Check if checkpoint exists
    if [ ! -d "$RESUME_CKPT" ]; then
        echo "WARNING: Checkpoint not found: $RESUME_CKPT"
        # echo "Available checkpoints in outputs/memory_wm_clean_only/:"
        # ls -d outputs/memory_wm_clean_only/checkpoint-* 2>/dev/null || echo "  (none)"
        echo ""
        if [ -t 0 ]; then
            read -p "Continue without resuming? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        else
             echo "Non-interactive mode detected. Aborting because resume checkpoint was not found."
             exit 1
        fi
        RESUME_CKPT=""
    else
        # Checkpoint exists, use it
        EXTRA_ARGS="exp.training_args.resume_from_checkpoint=$RESUME_CKPT"
        echo "Checkpoint verified: $RESUME_CKPT"
    fi
fi
# ============================================
# Run training with torchrun
# ============================================
nohup torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_hf.py \
    --config "$CONFIG_FILE" \
    $EXTRA_ARGS \
    > "$LOG_FILE" 2>&1 &

# Save PID
PID=$!
echo $PID > "${LOG_DIR}/train_pid.txt"

# Create latest_log symlink
ln -sf "train_${TIMESTAMP}.log" "${LOG_DIR}/latest_log.log"

echo "Training started with PID: $PID"
echo ""
echo "Monitor:  tail -f logs/latest_log.log"
echo "Stop:     kill $PID"
echo ""
