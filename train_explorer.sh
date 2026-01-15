#!/bin/bash

# Explorer Actor RL Training Script
# 参照 train.sh 风格，支持自动选择空闲GPU和后台训练
#
# Usage:
#   ./train_explorer.sh [OPTIONS]
#
# Options:
#   -c, --config FILE     Config file (default: f1_vla/config/explorer_train_config.yaml)
#   -g, --gpus IDS        GPU IDs, comma-separated (e.g., "0,1")
#   -a, --auto            Auto-detect free GPUs (memory < 2GB)
#   -m, --max-gpus N      Max GPUs to use in auto mode (default: 1)
#   -p, --phase N         Training phase (1, 2, or both)
#   -r, --resume PATH     Resume from checkpoint
#   -h, --help            Show this help message
#
# Examples:
#   ./train_explorer.sh -a                          # Auto GPU, default config
#   ./train_explorer.sh -g 5                        # Use GPU 5
#   ./train_explorer.sh -a -p 1                     # Phase 1 only
#   ./train_explorer.sh -a -r checkpoint.pth        # Resume training

set -e

# ============================================
# Default values
# ============================================
CONFIG_FILE="f1_vla/config/explorer_train_config.yaml"
GPU_IDS=""
AUTO_MODE=false
MAX_GPUS=1
PHASE=""
RESUME_CKPT=""
MEMORY_THRESHOLD=2000  # MB

# ============================================
# Parse arguments
# ============================================
show_help() {
    head -22 "$0" | tail -19
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
        -a|--auto)
            AUTO_MODE=true
            shift
            ;;
        -m|--max-gpus)
            MAX_GPUS="$2"
            shift 2
            ;;
        -p|--phase)
            PHASE="$2"
            shift 2
            ;;
        -r|--resume)
            RESUME_CKPT="$2"
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
            echo "  GPU $gpu_id: ${mem_used}MB (FREE - selected)"
        else
            if [ "$mem_used" -ge "$MEMORY_THRESHOLD" ]; then
                echo "  GPU $gpu_id: ${mem_used}MB (busy)"
            else
                echo "  GPU $gpu_id: ${mem_used}MB (skipped - max reached)"
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    
    if [ -z "$FREE_GPUS" ]; then
        echo "ERROR: No free GPUs available!"
        exit 1
    fi
    
    GPU_IDS="$FREE_GPUS"
else
    # Manual mode
    if [ -z "$GPU_IDS" ]; then
        GPU_IDS="0"
    fi
fi

# ============================================
# Setup environment
# ============================================
# Robust conda activation (works with different conda installations)
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# Try multiple conda activation methods
if ! command -v conda &> /dev/null; then
    # Try common conda paths
    for CONDA_PATH in ~/miniconda3 ~/anaconda3 /opt/conda ~/mambaforge; do
        if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
            source "$CONDA_PATH/etc/profile.d/conda.sh"
            break
        fi
    done
fi

# Activate the environment
conda activate f1 2>/dev/null || {
    echo "Warning: Failed to activate conda environment 'f1'. Proceeding with current environment."
}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

# ============================================
# Create log directory and file
# ============================================
LOG_DIR="logs/explorer"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_explorer_${TIMESTAMP}.log"

# ============================================
# Print training info
# ============================================
echo ""
echo "=========================================="
echo "Explorer Actor RL Training"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "GPU: $GPU_IDS"
echo "Mode: $([ "$AUTO_MODE" = true ] && echo "Auto-detect" || echo "Manual")"
if [ -n "$PHASE" ]; then
echo "Phase: $PHASE"
else
echo "Phase: Both (1 & 2)"
fi
if [ -n "$RESUME_CKPT" ]; then
echo "Resume: $RESUME_CKPT"
fi
echo "Log file: $LOG_FILE"
echo "=========================================="
echo ""

# ============================================
# Build command arguments
# ============================================
CMD_ARGS="--config $CONFIG_FILE"

if [ -n "$PHASE" ]; then
    CMD_ARGS="$CMD_ARGS --phase $PHASE"
fi

if [ -n "$RESUME_CKPT" ]; then
    CMD_ARGS="$CMD_ARGS --resume $RESUME_CKPT"
fi

# ============================================
# Run training in background
# ============================================
nohup python -u f1_vla/src/scripts/train_explorer.py $CMD_ARGS > "$LOG_FILE" 2>&1 &

# Save PID
PID=$!
echo $PID > "${LOG_DIR}/train_explorer_pid.txt"

# Create latest_log symlink
ln -sf "train_explorer_${TIMESTAMP}.log" "${LOG_DIR}/latest_log.log"

echo "Training started with PID: $PID"
echo ""
echo "Monitor:  tail -f logs/explorer/latest_log.log"
echo "Stop:     kill $PID"
echo ""
