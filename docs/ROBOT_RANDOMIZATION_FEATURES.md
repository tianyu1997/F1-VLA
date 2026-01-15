# Robot Randomization Features for RL Data Collection

## Overview

Added two new randomization features to improve diversity of RL training data:

1. **Random Embodiment Selection**: Randomly select robot embodiment at each episode reset
2. **Configurable Robot Init Noise**: Adjustable joint position noise magnitude

## Configuration Parameters

### In `rl_config.yaml`:

```yaml
# Robot embodiment settings
embodiment: ["franka-panda", "ur5", "kuka-iiwa"]  # List of available embodiments
randomize_embodiment: false  # Enable/disable random embodiment selection

# Robot initialization settings
randomize_robot_init: true  # Enable/disable robot init randomization
robot_init_noise_range: 0.1  # Joint position noise in radians (default: 0.1 ≈ 5.7°)
```

## Implementation Details

### 1. Random Embodiment Selection

**Modified Files:**
- `RoboTwin/rl/training/collect_data_teacher.py`
- `RoboTwin/envs/tasks/random_exploration.py`

**How it works:**
- When `randomize_embodiment: true` and multiple embodiments are provided
- At each `reset_for_new_episode()`, randomly selects one embodiment from the list
- Logs the selected embodiment: `[Random Embodiment] Selected: franka-panda from [...]`

**Example config:**
```yaml
embodiment: ["franka-panda", "ur5-v2", "kinova-gen3"]
randomize_embodiment: true  # Each episode uses random robot
```

### 2. Configurable Robot Init Noise

**Modified Files:**
- `RoboTwin/rl/training/collect_data_teacher.py`
- `RoboTwin/envs/tasks/random_exploration.py`

**How it works:**
- Parameter `robot_init_noise_range` controls joint position noise magnitude
- Applied to first 3 joints as: `qpos += uniform(-noise_range, +noise_range)`
- Default: 0.1 radians (~5.7 degrees)

**Noise range recommendations:**
- **Small (0.05 rad)**: Conservative, ~2.9° per joint - safe for most robots
- **Medium (0.1 rad)**: Default, ~5.7° per joint - good diversity
- **Large (0.2 rad)**: Aggressive, ~11.5° per joint - may cause collisions

**Example config:**
```yaml
randomize_robot_init: true
robot_init_noise_range: 0.15  # Larger noise for more diversity
```

## Data Flow

```
collect_data_teacher.py
  ├─> Reads config parameters:
  │     - embodiment: [list]
  │     - randomize_embodiment: bool
  │     - robot_init_noise_range: float
  │
  └─> Passes to TeacherEnv:
        └─> task_config contains:
              - embodiment: [list]
              - randomize_embodiment: bool
              - robot_init_noise_range: float
              
TeacherEnv.__init__()
  └─> Passes config to task.setup_demo()
  
random_exploration.setup_demo()
  ├─> If randomize_embodiment:
  │     └─> kwargs["embodiment"] = random.choice(embodiment_list)
  │
  └─> _randomize_robot_initial_state()
        └─> Uses robot_init_noise_range for perturbation
```

## Usage Examples

### Example 1: Single Embodiment with Variable Init
```yaml
embodiment: ["franka-panda"]
randomize_embodiment: false
randomize_robot_init: true
robot_init_noise_range: 0.1
```

### Example 2: Multiple Embodiments with Random Selection
```yaml
embodiment: ["franka-panda", "ur5-v2", "kinova-gen3"]
randomize_embodiment: true
randomize_robot_init: true
robot_init_noise_range: 0.15
```

### Example 3: Fixed Setup (No Randomization)
```yaml
embodiment: ["franka-panda"]
randomize_embodiment: false
randomize_robot_init: false
robot_init_noise_range: 0.0  # Ignored when randomize_robot_init=false
```

## Benefits

1. **Embodiment Diversity**: Training data covers multiple robot types → better generalization
2. **Initial State Diversity**: Varying start positions → more robust policies
3. **Configurable**: Easy to tune randomization strength via config file
4. **Backward Compatible**: Default behavior unchanged (single embodiment, 0.1 rad noise)

## Testing

To test the new features:

```bash
# Edit rl_config.yaml to enable features
vi rl_config.yaml

# Run data collection
cd RoboTwin/rl
python -m training.collect_data_teacher

# Check logs for:
# - "[Random Embodiment] Selected: ..." messages
# - Varying robot positions in saved trajectories
```

## Notes

- Embodiment list must contain valid embodiment names from `task_config/_embodiment_config.yml`
- Random embodiment selection happens at `reset_for_new_episode()`, not mid-episode
- Robot init noise only affects first 3 joints to avoid collision issues
- Set `robot_init_noise_range: 0.0` to disable noise (but keep `randomize_robot_init: true`)
