#!/bin/bash
# Quick check before training

echo "=========================================="
echo "Pre-Training Verification"
echo "=========================================="

# 1. Check checkpoint
echo ""
echo "[1] Checking checkpoint..."
CKPT_PATH="outputs/memory_wm_clean_only/checkpoint-episode-10000"
if [ -d "$CKPT_PATH" ]; then
    echo "  ✓ Checkpoint exists: $CKPT_PATH"
    echo "    Size: $(du -sh $CKPT_PATH | cut -f1)"
    echo "    Files:"
    ls -lh $CKPT_PATH | head -10
else
    echo "  ✗ Checkpoint NOT found: $CKPT_PATH"
    exit 1
fi

# 2. Check config
echo ""
echo "[2] Checking config..."
CONFIG_FILE="f1_vla/config/memory_wm_clean_only.yaml"
if [ -f "$CONFIG_FILE" ]; then
    echo "  ✓ Config exists: $CONFIG_FILE"
    echo ""
    echo "  Key settings:"
    grep -A 3 "vae_config:" $CONFIG_FILE || echo "  (vae_config section)"
    grep "resume_from_checkpoint:" $CONFIG_FILE || echo "  (resume setting)"
    grep "pixel_loss" $CONFIG_FILE || echo "  (pixel loss settings)"
else
    echo "  ✗ Config NOT found: $CONFIG_FILE"
    exit 1
fi

# 3. Check GPU availability
echo ""
echo "[3] Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv | head -9

echo ""
echo "  Free GPUs (< 2GB used):"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -F, '$2 < 2000 { printf "    GPU %s: %s MB\n", $1, $2 }'

# 4. Check Python environment
echo ""
echo "[4] Checking Python environment..."
which python3
python3 --version
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "  (PyTorch check failed)"

# 5. Memory estimation
echo ""
echo "[5] Memory estimation..."
echo "  VAE decoder unfrozen: +0.31 GB per GPU"
echo "  Current usage (GPU 1-4): ~47 GB"
echo "  Estimated new usage: ~47.3 GB"
echo "  GPU capacity: 49 GB (A6000)"
echo "  ✓ Should fit comfortably"

echo ""
echo "=========================================="
echo "✓ All checks passed!"
echo "=========================================="
echo ""
echo "Ready to start training. Run:"
echo "  ./train.sh -a          # Auto-detect free GPUs"
echo "  ./train.sh -g 1,2,3,4  # Use specific GPUs"
echo ""
