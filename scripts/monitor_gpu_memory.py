#!/usr/bin/env python3
"""
Memory清理监控脚本
定期清理CUDA缓存，防止OOM
"""

import os
import time
import subprocess
import torch

def get_gpu_memory_usage():
    """获取GPU显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, used, total = line.split(',')
            gpu_info.append({
                'id': int(gpu_id.strip()),
                'used': float(used.strip()),
                'total': float(total.strip()),
                'usage_pct': float(used.strip()) / float(total.strip()) * 100
            })
        return gpu_info
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def monitor_and_clean(threshold_pct=90, check_interval=60):
    """
    监控GPU显存并在必要时清理
    
    Args:
        threshold_pct: 显存使用率阈值（百分比）
        check_interval: 检查间隔（秒）
    """
    print(f"开始监控GPU显存...")
    print(f"清理阈值: {threshold_pct}%")
    print(f"检查间隔: {check_interval}秒")
    print("-" * 60)
    
    while True:
        gpu_info = get_gpu_memory_usage()
        
        for gpu in gpu_info:
            status = "⚠️ HIGH" if gpu['usage_pct'] > threshold_pct else "✓ OK"
            print(f"[{time.strftime('%H:%M:%S')}] GPU{gpu['id']}: "
                  f"{gpu['used']:.0f}/{gpu['total']:.0f} MB "
                  f"({gpu['usage_pct']:.1f}%) {status}")
            
            # 如果超过阈值，记录警告（实际清理由PyTorch自动管理）
            if gpu['usage_pct'] > threshold_pct:
                print(f"  ⚠️  GPU{gpu['id']} 显存使用率过高！")
                print(f"  建议: 检查训练进程是否正常，考虑降低batch size")
        
        print()
        time.sleep(check_interval)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GPU显存监控')
    parser.add_argument('--threshold', type=float, default=90.0,
                        help='显存使用率阈值（百分比），默认90')
    parser.add_argument('--interval', type=int, default=60,
                        help='检查间隔（秒），默认60')
    
    args = parser.parse_args()
    
    try:
        monitor_and_clean(args.threshold, args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止")
