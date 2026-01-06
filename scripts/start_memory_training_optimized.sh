#!/bin/bash
# 优化后的训练启动脚本 - 防止OOM
# 
# 主要改进:
# 1. 设置PYTORCH_CUDA_ALLOC_CONF环境变量
# 2. 使用优化后的配置文件
# 3. 限制GPU数量避免显存碎片

set -e

echo "=========================================="
echo "启动优化版本训练 (防OOM)"
echo "=========================================="

# 设置CUDA内存分配策略 - 避免碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# 检查配置文件
CONFIG_FILE="${1:-f1_vla/config/memory_from_f1pretrain_v2.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo "CUDA分配策略: $PYTORCH_CUDA_ALLOC_CONF"
echo ""
echo "优化措施:"
echo "  - memory_len: 32 → 16"
echo "  - gradient_accumulation_steps: 8 → 4"
echo "  - save_episodes: 500 → 200"
echo "  - 启用 expandable_segments"
echo "=========================================="
echo ""

# 使用train.sh自动选择GPU（推荐使用2-3个GPU，避免4个GPU时的显存碎片）
read -p "自动选择GPU数量 [2-3推荐]: " MAX_GPUS
MAX_GPUS=${MAX_GPUS:-2}

./train.sh -a -m "$MAX_GPUS" -c "$CONFIG_FILE"

echo ""
echo "训练已启动！"
echo "监控命令: tail -f logs/latest_log.log"
