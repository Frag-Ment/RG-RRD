# Hyperparameter Tuning Guide for MRCMPSP

This guide explains how to use the automatic hyperparameter tuning system to optimize your MRCMPSP agent's performance on MISTA benchmarks.

## Overview

The hyperparameter tuning system automatically:
- Tests multiple hyperparameter configurations
- Trains a model for each configuration
- Evaluates on validation set
- Identifies the best performing configuration
- Saves results for analysis

## Quick Start

### 1. Quick Search (Recommended for First Time)

Test 10 configurations in ~2-3 hours:

```bash
cd RL-RCPSP
python tune_hyperparameters.py --search_type quick
```

This will:
- Test 10 random configurations from a reduced search space
- Train each for 50K timesteps (~10 minutes per config)
- Save best configuration to `hp_tuning_results/best_config.json`

### 2. Random Search (Recommended for Comprehensive Tuning)

Test 20 configurations exploring full hyperparameter space:

```bash
python tune_hyperparameters.py --search_type random --n_configs 20 --timesteps 100000
```

This will:
- Test 20 random configurations from full search space
- Train each for 100K timesteps (~20 minutes per config)
- Total time: ~6-8 hours

### 3. Grid Search (For Small Search Spaces)

Test all combinations (use with caution - can be very slow):

```bash
python tune_hyperparameters.py --search_type grid --timesteps 50000
```

**Warning**: Grid search on full space = thousands of configurations. Only use on reduced search space.

---

## Hyperparameters Being Tuned

### Architecture Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `gnn_layers` | [2, 3, 4] | Number of GNN layers for task graph processing |
| `gnn_hidden_dim` | [32, 64, 128] | Hidden dimension size in GNN |
| `mlp_hidden_dim` | [16, 32, 64] | Hidden dimension for task-mode MLP |
| `project_hidden_dim` | [32, 64, 128] | Hidden dimension for project selector |

### Training Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `learning_rate` | [1e-5, 3e-5, 1e-4] | Adam optimizer learning rate |
| `clip_coef` | [0.1, 0.15, 0.2] | PPO clipping coefficient |
| `ent_coef` | [0.005, 0.01, 0.02] | Entropy regularization coefficient |
| `vf_coef` | [0.3, 0.5, 0.7] | Value function loss coefficient |

### PPO Parameters

| Parameter | Range | Description |
|-----------|-------|-------------|
| `num_envs` | [2, 4, 8] | Number of parallel environments |
| `num_steps` | [64, 128, 256] | Steps per environment per update |
| `minibatch_size` | [32, 64, 128] | Minibatch size for optimization |
| `update_epochs` | [3, 4, 5] | Optimization epochs per update |

---

## Understanding the Output

### During Training

For each configuration, you'll see:

```
======================================================================
Training with config:
  gnn_layers: 3
  gnn_hidden_dim: 64
  learning_rate: 3e-05
  clip_coef: 0.1
  ent_coef: 0.01
  ...
======================================================================
Training for 488 updates...
Training: 100%|████████████████████| 488/488 [08:32<00:00,  1.05s/it]

Evaluating on validation set...

✓ Results:
  Val makespan: 45.32
  Success rate: 100.0%
```

### Final Results

After all configurations are tested:

```
======================================================================
Hyperparameter Search Complete!
======================================================================

Top 5 Configurations:
======================================================================

Rank 1:
  Val Makespan: 42.18 ± 3.24
  Config:
    gnn_layers: 3
    gnn_hidden_dim: 128
    learning_rate: 1e-04
    clip_coef: 0.15
    ...

Rank 2:
  Val Makespan: 44.56 ± 4.12
  Config:
    gnn_layers: 4
    gnn_hidden_dim: 64
    ...
```

---

## Output Files

All results are saved to `hp_tuning_results/`:

### `best_config.json`
The best hyperparameter configuration:

```json
{
  "gnn_layers": 3,
  "gnn_hidden_dim": 128,
  "mlp_hidden_dim": 32,
  "project_hidden_dim": 64,
  "learning_rate": 0.0001,
  "clip_coef": 0.15,
  "ent_coef": 0.01,
  "vf_coef": 0.5,
  "num_envs": 4,
  "num_steps": 128,
  "minibatch_size": 64,
  "update_epochs": 4
}
```

### `hp_search_{type}_{timestamp}.json`
Complete results for all configurations:

```json
{
  "search_type": "random",
  "n_configs": 20,
  "timesteps_per_config": 100000,
  "best_config": {...},
  "best_makespan": 42.18,
  "all_results": [
    {
      "config": {...},
      "val_makespan_mean": 42.18,
      "val_makespan_std": 3.24,
      "train_loss_mean": 0.234,
      "train_reward_mean": 0.456,
      "val_success_rate": 100.0,
      "model_path": "hp_tuning_models/model_a1b2c3d4.pt",
      "num_parameters": 45678
    },
    ...
  ]
}
```

### `results.json`
Intermediate results (updated during search):
- Useful if search is interrupted
- Contains results for completed configurations

### Trained Models

All trained models are saved to `hp_tuning_models/`:
- `model_{hash}.pt` - Model weights for each configuration
- Hash is unique identifier based on configuration

---

## Using the Best Configuration

### Option 1: Manual Update

Copy best hyperparameters to `train_mista.py`:

```python
class MRCMPSPArgs:
    # Architecture (from best_config.json)
    gnn_layers = 3
    gnn_hidden_dim = 128
    mlp_hidden_dim = 32
    project_hidden_dim = 64

    # Training
    learning_rate = 1e-4
    clip_coef = 0.15
    ent_coef = 0.01
    vf_coef = 0.5

    # PPO
    num_envs = 4
    num_steps = 128
    minibatch_size = 64
    update_epochs = 4
```

### Option 2: Load Best Model

Continue training from best checkpoint:

```python
import torch
from model.hierarchical_agent import HierarchicalAgent

# Load best model
agent = HierarchicalAgent(...)
agent.load_state_dict(torch.load('hp_tuning_models/model_a1b2c3d4.pt'))

# Continue training or evaluate
```

### Option 3: Train from Scratch with Best Config

Use the identified hyperparameters but train for longer:

```bash
# After updating train_mista.py with best hyperparameters
python train_mista.py  # Will train for full 500K timesteps
```

---

## Search Strategies

### Quick Search (Default)

**When to use**: First time tuning, quick iteration

**Characteristics**:
- Reduced search space (2 values per parameter)
- 10 random configurations
- 50K timesteps per config
- Time: ~2-3 hours

**Pros**:
- Fast results
- Good for identifying important hyperparameters
- Reasonable performance improvement

**Cons**:
- May not find optimal configuration
- Limited exploration

### Random Search

**When to use**: Comprehensive tuning for publication/production

**Characteristics**:
- Full search space
- Customizable number of configurations (20-50 recommended)
- 100K-200K timesteps per config
- Time: 6-24 hours

**Pros**:
- Better coverage than grid search with fewer configurations
- Can find good configurations efficiently
- Scalable to large search spaces

**Cons**:
- Requires more time than quick search
- Results can vary between runs

### Grid Search

**When to use**: Very small search spaces, reproducible results needed

**Characteristics**:
- Tests ALL combinations
- Number of configs = product of all value counts
- Example: 2×2×2×2×2 = 32 configs (manageable)
- Full space: 3×3×3×3×3×3×3×3×3×3 = 59,049 configs (impractical!)

**Pros**:
- Exhaustive - guaranteed to find best in search space
- Reproducible
- No randomness

**Cons**:
- Exponential growth with parameters
- Very slow for large search spaces
- Wasteful - tests many poor configurations

---

## Tips for Best Results

### 1. Start Small

```bash
# First: Quick search to get baseline
python tune_hyperparameters.py --search_type quick --timesteps 50000

# Then: Random search with more configs
python tune_hyperparameters.py --search_type random --n_configs 20 --timesteps 100000

# Finally: Train best config for full duration
# (Update train_mista.py and run for 500K-1M timesteps)
```

### 2. Monitor GPU Memory

If you get CUDA out of memory errors:

```python
# In tune_hyperparameters.py, modify search space to exclude large configs:
'gnn_hidden_dim': [32, 64],  # Remove 128
'num_envs': [2, 4],          # Remove 8
```

### 3. Adjust Training Duration

- **Initial search**: 50K timesteps (fast, rough estimates)
- **Refined search**: 100K timesteps (better estimates)
- **Final training**: 500K-1M timesteps (best performance)

### 4. Validate on Multiple Sets

After finding best config, validate on all sets:

```bash
# Edit evaluate_mista.py to test on Set A, B, and X
python evaluate_mista.py
```

### 5. Analyze Trends

Look for patterns in results:
- Do larger models always win?
- Is learning rate the most important parameter?
- Which parameters have little effect?

This helps refine future searches.

---

## Advanced Usage

### Custom Search Space

Edit `tune_hyperparameters.py` to define custom ranges:

```python
def get_custom_search_space():
    return {
        'gnn_layers': [3, 4, 5],              # Focus on deeper models
        'gnn_hidden_dim': [64, 128, 256],     # Larger models
        'learning_rate': [1e-5, 3e-5, 5e-5],  # Finer LR grid
        'num_envs': [4],                       # Fix this parameter
        'num_steps': [128],                    # Fix this parameter
        # ... other parameters
    }
```

Then modify the search function to use it:

```python
search_space = get_custom_search_space()
configs = generate_random_search_configs(search_space, n_configs=30)
```

### Curriculum Learning with Hyperparameters

Test if curriculum learning helps:

```python
# Train first on j10, then on Set A
train_instances_phase1 = loader.load_single_project_set(10)[:50]
train_instances_phase2 = loader.load_set('A')

# Phase 1: Small instances (25K timesteps)
result1 = train_with_config(config, train_instances_phase1, val_instances, timesteps=25000)

# Phase 2: Full instances (50K timesteps)
result2 = train_with_config(config, train_instances_phase2, val_instances, timesteps=50000)
```

### Parallel Search (Manual)

Run multiple searches in parallel on different GPUs:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python tune_hyperparameters.py --search_type random --n_configs 10

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python tune_hyperparameters.py --search_type random --n_configs 10

# Combine results manually
```

---

## Interpreting Results

### What Makes a Good Configuration?

**Primary metric**: Validation makespan (lower is better)

**Secondary metrics**:
- **Success rate**: Should be 100% (model completes all instances)
- **Standard deviation**: Lower is better (more consistent)
- **Training loss**: Should decrease but doesn't directly correlate with performance
- **Model size**: Fewer parameters = faster inference (if performance is similar)

### Typical Improvements

- **Quick search**: 5-10% improvement over default
- **Random search**: 10-20% improvement over default
- **Best possible**: 15-30% improvement (depends on data)

### When to Stop Searching

Stop when:
1. ✓ Top 3 configurations have similar performance (< 2% difference)
2. ✓ You've tested 20+ configurations with random search
3. ✓ Results plateaued (new configs not improving)
4. ✓ Time/compute budget exhausted

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1**: Reduce search space
```python
'gnn_hidden_dim': [32, 64],     # Remove 128
'num_envs': [2, 4],              # Remove 8
'minibatch_size': [32, 64],     # Remove 128
```

**Solution 2**: Use CPU
```bash
CUDA_VISIBLE_DEVICES="" python tune_hyperparameters.py --timesteps 20000
```

### Issue: "Training failed" for some configs

**This is normal!** Some hyperparameter combinations don't work well:
- Very large learning rates → unstable training
- Too many environments with small minibatch → poor updates
- Mismatched batch/minibatch sizes → errors

The tuning script will catch these and continue with other configs.

### Issue: All configurations fail

**Check**:
1. MISTA data is accessible: `ls Mista/A/`
2. Dependencies installed: `python check_setup.py`
3. Basic training works: `python train_mista.py` (let it run for 1 update)

### Issue: Very slow progress

**Expected speeds** (on GPU):
- Configuration with num_envs=2: ~50-60 updates/hour
- Configuration with num_envs=4: ~80-100 updates/hour
- Configuration with num_envs=8: ~120-150 updates/hour

**If much slower**:
- Running on CPU? (10x slower is expected)
- Very large GNN? (gnn_hidden_dim=256 can be slow)
- Disk I/O bottleneck? (reduce logging)

### Issue: Results inconsistent between runs

**This is expected** due to:
- Random initialization
- Random episode sampling during training
- Random action exploration (entropy)

**To reduce variance**:
- Increase `timesteps_per_config` to 200K
- Evaluate on more validation instances (edit code to use all 10)
- Run multiple seeds and average results

---

## Example Workflow

### Complete Tuning Pipeline (3 Days)

**Day 1 - Initial Exploration**
```bash
# Quick search to get baseline (3 hours)
python tune_hyperparameters.py --search_type quick --timesteps 50000

# Check results
cat hp_tuning_results/best_config.json

# Note: Let's say best makespan is 45.2
```

**Day 2 - Comprehensive Search**
```bash
# Random search with more configs (8 hours)
python tune_hyperparameters.py --search_type random --n_configs 20 --timesteps 100000

# Check improvement
cat hp_tuning_results/best_config.json

# Note: Best makespan improved to 42.8
```

**Day 3 - Final Training**
```bash
# Update train_mista.py with best hyperparameters
# Train for full duration (4 hours)
python train_mista.py

# Evaluate
python evaluate_mista.py

# Final result: Makespan 41.5, 15% improvement over EDD baseline
```

---

## FAQ

**Q: How long does hyperparameter tuning take?**

A: Depends on search type:
- Quick: 2-3 hours (10 configs × 10 min)
- Random (20 configs): 6-8 hours (20 configs × 20 min)
- Grid (small): 4-6 hours
- Grid (full): Days to weeks (not recommended)

**Q: Can I resume interrupted search?**

A: Yes! Results are saved incrementally to `hp_tuning_results/results.json`. You can:
1. Check which configs completed
2. Manually re-run with remaining configs
3. Or just start a new search (results are cumulative)

**Q: How do I use the best model directly?**

A: Load it for evaluation:
```python
agent = HierarchicalAgent(...)
agent.load_state_dict(torch.load('hp_tuning_models/model_XXXXXXXX.pt'))
# Now use agent for inference
```

**Q: Should I tune on Set A or Set B?**

A: Train on Set A, validate on Set B (this is the default). This tests generalization.

**Q: Can I tune other hyperparameters?**

A: Yes! Edit `HyperparameterConfig` class in `tune_hyperparameters.py`. You can add:
- Gamma (discount factor)
- GAE lambda
- Max grad norm
- Model architecture choices (attention heads, etc.)

**Q: How many random configs should I test?**

A: General rule:
- 10-15 configs: Quick insights
- 20-30 configs: Good coverage
- 50+ configs: Diminishing returns (unless huge search space)

**Q: Is random search better than grid search?**

A: For most cases, **yes**:
- Random search explores space more efficiently
- Grid search wastes time on bad combinations
- Research shows random search finds good configs faster

Only use grid search when:
- Search space is very small (< 50 configs)
- You need exhaustive coverage for publication
- Reproducibility is critical

---

## Summary

**For most users, follow this simple workflow**:

```bash
# 1. Quick test (optional, but recommended for first time)
python tune_hyperparameters.py --search_type quick

# 2. Comprehensive search
python tune_hyperparameters.py --search_type random --n_configs 20 --timesteps 100000

# 3. Copy best config to train_mista.py

# 4. Final training
python train_mista.py

# 5. Evaluate
python evaluate_mista.py
```

**Expected outcome**: 10-20% improvement over default configuration, leading to better results than baseline heuristics.

**Good luck with your hyperparameter tuning!** 🚀
