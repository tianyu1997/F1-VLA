
import torch
import glob
import os

data_dir = "/mnt/data2/ty/F1-VLA/data/clean_teacher_offline/part_gpu0"
ep_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.pt")))

if not ep_files:
    print(f"No files in {data_dir}")
    exit(1)

print(f"Loading {ep_files[0]}")
episode = torch.load(ep_files[0])
print(f"Episode length: {len(episode)}")
print(f"Type of episode: {type(episode)}")

if len(episode) > 0:
    frame = episode[0]
    print(f"Frame keys: {frame.keys()}")
    if 'obs' in frame:
        obs = frame['obs']
        print(f"Obs keys: {obs.keys()}")
        for k, v in obs.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            elif isinstance(v, list):
                print(f"  {k}: list len {len(v)}")
            else:
                print(f"  {k}: {type(v)}")

        # Check image values
        if 'head_rgb' in obs:
            img = obs['head_rgb']
            print(f"  head_rgb min/max: {img.min()}, {img.max()}")
