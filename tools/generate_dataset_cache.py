import os
import glob
import torch
import json
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_file(file_path):
    try:
        data = torch.load(file_path, weights_only=False)
        return len(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def process_directory(data_dir):
    print(f"Processing {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
    if not files:
        print(f"No episode files found in {data_dir}")
        return

    # Extract episode indices to ensure correct ordering in cache list
    # The cache is a list where index i corresponds to episode_i.pt
    # We must handle potential missing episodes if any, though usually sequential.
    
    max_idx = -1
    file_map = {}
    
    for f in files:
        basename = os.path.basename(f)
        # episode_000123.pt
        try:
            idx = int(basename.split('_')[1].split('.')[0])
            file_map[idx] = f
            max_idx = max(max_idx, idx)
        except:
            print(f"Skipping malformed filename: {basename}")
            continue
            
    if max_idx == -1:
        print("No valid episodes found.")
        return

    # Prepare list for cache
    cache_lengths = [0] * (max_idx + 1)
    
    # Identify files to process
    tasks = []
    for i in range(max_idx + 1):
        if i in file_map:
            tasks.append((i, file_map[i]))
    
    print(f"Loading {len(tasks)} episodes from {data_dir}...")
    
    # Use multiprocessing
    with Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_file, [t[1] for t in tasks]), total=len(tasks)))
    
    # Fill cache
    for (idx, _), length in zip(tasks, results):
        if length is not None:
            cache_lengths[idx] = length
            
    # Save cache
    cache_path = os.path.join(data_dir, ".mekvm_index_cache.json")
    with open(cache_path, 'w') as f:
        json.dump({'episode_lengths': cache_lengths}, f)
    
    print(f"Saved cache to {cache_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/data2/ty/F1-VLA/data/clean_teacher_offline")
    args = parser.parse_args()
    
    # Find Partitions
    subdirs = glob.glob(os.path.join(args.data_root, "part_gpu*"))
    print(f"Found partitions: {subdirs}")
    
    for d in subdirs:
        process_directory(d)
