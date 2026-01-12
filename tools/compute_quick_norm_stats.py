#!/usr/bin/env python3
"""
快速计算数据集归一化统计的工具

使用方法:
    python tools/compute_quick_norm_stats.py --data_dirs data/clean_teacher_offline/part_gpu0

输出:
    norm_stats.yaml - 包含state和action的mean/std
"""

import argparse
import glob
import os
import numpy as np
import torch
from tqdm import tqdm
import yaml


def compute_norm_stats(data_dirs, max_episodes=100, sample_rate=10):
    """
    计算state和action的归一化统计
    
    Args:
        data_dirs: 数据目录列表
        max_episodes: 最多处理多少个episode
        sample_rate: 采样率（每N帧取1帧）
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    # 收集所有episode文件
    all_episode_files = []
    for data_dir in data_dirs:
        ep_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
        all_episode_files.extend(ep_files[:max_episodes])
    
    print(f"Found {len(all_episode_files)} episode files")
    
    # 收集state和action
    all_states = []
    all_actions = []
    
    for ep_file in tqdm(all_episode_files, desc="Processing episodes"):
        try:
            episode = torch.load(ep_file, map_location='cpu', weights_only=False)
            
            # 采样帧
            for i in range(0, len(episode), sample_rate):
                frame = episode[i]
                
                # State
                if 'obs' in frame and 'state' in frame['obs']:
                    state = frame['obs']['state']
                    if isinstance(state, np.ndarray):
                        all_states.append(state)
                
                # Action
                if 'action' in frame:
                    action = frame['action']
                    if isinstance(action, (list, np.ndarray)):
                        all_actions.append(np.array(action))
            
            del episode  # 释放内存
        except Exception as e:
            print(f"Error processing {ep_file}: {e}")
            continue
    
    print(f"Collected {len(all_states)} state samples, {len(all_actions)} action samples")
    
    # 计算统计
    if len(all_states) == 0 or len(all_actions) == 0:
        raise ValueError("No data collected! Check data format.")
    
    states_array = np.stack(all_states, axis=0)  # (N, state_dim)
    actions_array = np.stack(all_actions, axis=0)  # (N, action_dim)
    
    state_mean = states_array.mean(axis=0).tolist()
    state_std = states_array.std(axis=0).tolist()
    action_mean = actions_array.mean(axis=0).tolist()
    action_std = actions_array.std(axis=0).tolist()
    
    # 替换接近0的std为1.0（常量特征）
    state_std = [s if s > 1e-6 else 1.0 for s in state_std]
    action_std = [s if s > 1e-6 else 1.0 for s in action_std]
    
    norm_stats = {
        'norm_stats': {
            'state': {
                'mean': state_mean,
                'std': state_std,
            },
            'action': {
                'mean': action_mean,
                'std': action_std,
            }
        }
    }
    
    return norm_stats


def main():
    parser = argparse.ArgumentParser(description='计算数据集归一化统计')
    parser.add_argument('--data_dirs', nargs='+', required=True, 
                       help='数据目录列表')
    parser.add_argument('--output', type=str, default='f1_vla/config/norm_stats.yaml',
                       help='输出文件路径')
    parser.add_argument('--max_episodes', type=int, default=100,
                       help='最多处理多少个episode (加快计算)')
    parser.add_argument('--sample_rate', type=int, default=10,
                       help='采样率 (每N帧取1帧)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("计算数据集归一化统计")
    print("=" * 60)
    print(f"数据目录: {args.data_dirs}")
    print(f"最多处理: {args.max_episodes} episodes")
    print(f"采样率: 1/{args.sample_rate}")
    print(f"输出文件: {args.output}")
    print("=" * 60)
    
    # 计算统计
    norm_stats = compute_norm_stats(
        args.data_dirs, 
        max_episodes=args.max_episodes,
        sample_rate=args.sample_rate
    )
    
    # 保存到yaml
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(norm_stats, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ 统计已保存到: {args.output}")
    print("\n统计摘要:")
    print(f"State维度: {len(norm_stats['norm_stats']['state']['mean'])}")
    print(f"State mean range: [{min(norm_stats['norm_stats']['state']['mean']):.4f}, "
          f"{max(norm_stats['norm_stats']['state']['mean']):.4f}]")
    print(f"State std range: [{min(norm_stats['norm_stats']['state']['std']):.4f}, "
          f"{max(norm_stats['norm_stats']['state']['std']):.4f}]")
    print(f"Action维度: {len(norm_stats['norm_stats']['action']['mean'])}")
    print(f"Action mean range: [{min(norm_stats['norm_stats']['action']['mean']):.4f}, "
          f"{max(norm_stats['norm_stats']['action']['mean']):.4f}]")
    print(f"Action std range: [{min(norm_stats['norm_stats']['action']['std']):.4f}, "
          f"{max(norm_stats['norm_stats']['action']['std']):.4f}]")
    
    print("\n使用方法:")
    print("1. 在配置文件顶部添加:")
    print(f"   !include {args.output}")
    print("2. 或在dataset配置中添加:")
    print("   norm_stats: !include " + args.output)


if __name__ == '__main__':
    main()
