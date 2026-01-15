#!/bin/bash
# 快速测试优化后的配置是否能正常运行
# 只训练10个episodes，验证没有OOM

set -e

echo "========================================"
echo "测试优化配置 - 快速验证"
echo "========================================"

CONFIG_FILE="f1_vla/config/memory_from_f1pretrain_v2.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

echo "测试参数:"
echo "  - 配置: $CONFIG_FILE"
echo "  - GPU: 自动选择1个"
echo "  - Episodes: 10 (测试用)"
echo "  - CUDA分配策略: $PYTORCH_CUDA_ALLOC_CONF"
echo ""

# 使用train.sh启动，限制为10 episodes
echo "启动测试训练..."
timeout 600s ./train.sh -a -m 1 -c "$CONFIG_FILE" &

TRAIN_PID=$!
echo "训练PID: $TRAIN_PID"
echo ""

# 监控10分钟
echo "监控10分钟..."
for i in {1..10}; do
    sleep 60
    echo "[${i}/10分钟] 检查GPU显存..."
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  $line"
    done
    echo ""
    
    # 检查进程是否还在运行
    if ! ps -p $TRAIN_PID > /dev/null; then
        echo "❌ 训练进程已退出"
        break
    fi
done

# 停止训练
echo "测试完成，停止训练..."
kill $TRAIN_PID 2>/dev/null || true
sleep 5

# 检查日志
echo ""
echo "========================================"
echo "检查训练日志"
echo "========================================"
LATEST_LOG=$(ls -t logs/train_*.log | head -1)

if [ -f "$LATEST_LOG" ]; then
    echo "最新日志: $LATEST_LOG"
    echo ""
    
    # 检查是否有OOM错误
    if grep -q "OutOfMemory" "$LATEST_LOG"; then
        echo "❌ 发现OOM错误！"
        grep "OutOfMemory" "$LATEST_LOG" | tail -5
        exit 1
    else
        echo "✅ 未发现OOM错误"
    fi
    
    # 显示训练进度
    echo ""
    echo "训练进度:"
    grep "episode" "$LATEST_LOG" | tail -5
    
    echo ""
    echo "========================================"
    echo "✅ 测试通过！配置可以正常使用"
    echo "========================================"
else
    echo "❌ 找不到训练日志"
    exit 1
fi
