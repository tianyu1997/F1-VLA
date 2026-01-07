import torch
import os
import glob
import numpy as np
from tqdm import tqdm

def compute_stats(data_dir):
    print(f"Scanning {data_dir}...")
    episode_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))
    if not episode_files:
        print("No episode files found.")
        return

    print(f"Found {len(episode_files)} episodes.")
    
    # Initialize accumulators
    state_sum = None
    state_sq_sum = None
    action_sum = None
    action_sq_sum = None
    state_count = 0
    action_count = 0
    
    # Process a subset if too many
    max_episodes = 200 # Sample 200 episodes to estimate
    step = max(1, len(episode_files) // max_episodes)
    sampled_files = episode_files[::step][:max_episodes]
    
    print(f"Processing {len(sampled_files)} episodes for estimation...")

    for fpath in tqdm(sampled_files):
        try:
            episode = torch.load(fpath, weights_only=False)
            # episode is a list of steps, each step is dict with 'obs', 'action'
            
            states = []
            actions = []
            
            for step in episode:
                if 'state' in step['obs']:
                    states.append(step['obs']['state'])
                if 'action' in step:
                    actions.append(step['action'])
            
            if states:
                states = np.array(states) # (T, state_dim)
                if state_sum is None:
                    state_sum = np.zeros(states.shape[1], dtype=np.float64)
                    state_sq_sum = np.zeros(states.shape[1], dtype=np.float64)
                    
                state_sum += states.sum(axis=0)
                state_sq_sum += (states ** 2).sum(axis=0)
                state_count += states.shape[0]
            
            if actions:
                actions = np.array(actions) # (T, action_dim)
                if action_sum is None:
                    action_sum = np.zeros(actions.shape[1], dtype=np.float64)
                    action_sq_sum = np.zeros(actions.shape[1], dtype=np.float64)
                
                action_sum += actions.sum(axis=0)
                action_sq_sum += (actions ** 2).sum(axis=0)
                action_count += actions.shape[0]
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

    print("\n--- Statistics ---")
    
    if state_count > 0:
        state_mean = state_sum / state_count
        state_var = (state_sq_sum / state_count) - (state_mean ** 2)
        state_std = np.sqrt(np.maximum(state_var, 1e-8))
        
        print(f"State Dim: {state_mean.shape[0]}")
        print("State Mean:")
        print(state_mean.tolist())
        print("State Std:")
        print(state_std.tolist())
        print(f"State Max element: {np.abs(state_mean).max() + 3*state_std.max()}")
    else:
        print("No state data found.")

    if action_count > 0:
        action_mean = action_sum / action_count
        action_var = (action_sq_sum / action_count) - (action_mean ** 2)
        action_std = np.sqrt(np.maximum(action_var, 1e-8))
        
        print(f"Action Dim: {action_mean.shape[0]}")
        print("Action Mean:")
        print(action_mean.tolist())
        print("Action Std:")
        print(action_std.tolist())
        print(f"Action Max element: {np.abs(action_mean).max() + 3*action_std.max()}")
    else:
        print("No action data found.")

if __name__ == "__main__":
    compute_stats("/mnt/data2/ty/F1-VLA/ME_KVM_VLA/data/clean")
