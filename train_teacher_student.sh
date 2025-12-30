#!/bin/bash

# Teacher-Student Distillation Training Script
# Usage:
#   ./train_teacher_student.sh -c <config> [-a] [-g <gpus>] [-m <max_gpus>] [-p <port>]
#
# Options:
#   -c <config>    Config file (required)
#   -a             Auto-select available GPUs
#   -g <gpus>      Manual GPU selection (e.g., "0,1,2,3")
#   -m <max_gpus>  Maximum number of GPUs to use with -a (default: 4)
#   -p <port>      Master port (default: 29500)
#
# Examples:
#   # Teacher-Student with memory distillation:
#   ./train_teacher_student.sh -a -c f1_vla/config/teacher_student_config.yaml
#
#   # Control group (student only):
#   ./train_teacher_student.sh -a -c f1_vla/config/student_only_config.yaml
#
#   # Specific GPUs:
#   ./train_teacher_student.sh -g 0,1 -c f1_vla/config/teacher_student_config.yaml

set -e

# Default values
CONFIG_FILE=""
AUTO_GPU=false
MANUAL_GPUS=""
MAX_GPUS=4
MASTER_PORT=29501
MEMORY_THRESHOLD=2000

# Change to project directory
cd /mnt/data2/ty/F1-VLA

# Parse arguments
while getopts "c:ag:m:p:h" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG" ;;
        a) AUTO_GPU=true ;;
        g) MANUAL_GPUS="$OPTARG" ;;
        m) MAX_GPUS="$OPTARG" ;;
        p) MASTER_PORT="$OPTARG" ;;
        h) 
            echo "Usage: $0 -c <config> [-a] [-g <gpus>] [-m <max_gpus>] [-p <port>]"
            echo ""
            echo "Options:"
            echo "  -c <config>    Config file (required)"
            echo "  -a             Auto-select available GPUs"
            echo "  -g <gpus>      Manual GPU selection (e.g., '0,1,2,3')"
            echo "  -m <max_gpus>  Maximum number of GPUs with -a (default: 4)"
            echo "  -p <port>      Master port (default: 29501)"
            echo ""
            echo "Examples:"
            echo "  # Teacher-Student training:"
            echo "  $0 -a -c f1_vla/config/teacher_student_config.yaml"
            echo ""
            echo "  # Control group (student only):"
            echo "  $0 -a -c f1_vla/config/student_only_config.yaml"
            exit 0
            ;;
        \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Error: Config file is required (-c)"
    echo "Use -h for help"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# ============================================
# Setup environment (same as train.sh)
# ============================================
source ~/.bashrc
conda activate f1 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh && conda activate f1

export TOKENIZERS_PARALLELISM=false

# ============================================
# GPU Selection
# ============================================
if [ "$AUTO_GPU" = true ]; then
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
    
    GPUS="$FREE_GPUS"
    NUM_GPUS="$GPU_COUNT"
elif [ -n "$MANUAL_GPUS" ]; then
    GPUS="$MANUAL_GPUS"
    NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
    echo "Using specified GPUs: $GPUS"
else
    GPUS="0"
    NUM_GPUS=1
    echo "Using default GPU: 0"
fi

# Create logs directory
mkdir -p logs

# Create log file name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
LOG_FILE="logs/teacher_student_${CONFIG_NAME}_${TIMESTAMP}.log"

echo "=============================================="
echo "Teacher-Student Distillation Training"
echo "=============================================="
echo "Config:     $CONFIG_FILE"
echo "GPUs:       $GPUS ($NUM_GPUS GPUs)"
echo "Port:       $MASTER_PORT"
echo "Log file:   $LOG_FILE"
echo "=============================================="

# Export GPU configuration
export CUDA_VISIBLE_DEVICES=$GPUS

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Starting distributed training on $NUM_GPUS GPUs..."
    nohup torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        train_teacher_student.py \
        --config-file "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1 &
else
    echo "Starting single-GPU training..."
    nohup python train_teacher_student.py \
        --config-file "$CONFIG_FILE" \
        > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo $PID > logs/teacher_student_pid.txt

echo ""
echo "Training started in background"
echo "PID: $PID"
echo "Log: $LOG_FILE"
echo ""
echo "Monitor with: tail -f $LOG_FILE"
echo "Stop with:    kill $PID"

# Create latest log symlink
ln -sf "$LOG_FILE" logs/latest_teacher_student_log
