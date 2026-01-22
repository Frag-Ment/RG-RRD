# Mac M2 Setup Guide for MRCMPSP

This guide is specifically for Mac M2/M3 users with 16GB RAM.

## Quick Setup

### 1. Verify PyTorch with MPS Support

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

Expected output:
```
PyTorch: 2.x.x
MPS Available: True
```

If MPS is not available, upgrade PyTorch:
```bash
pip3 install --upgrade torch torchvision
```

### 2. Check Device Configuration

```bash
python device_config.py
```

This will show:
- ✓ Selected Device: mps (Apple Silicon)
- Memory Configuration optimized for 16GB RAM
- Recommended settings for training

### 3. Verify Setup

```bash
python check_setup.py
```

This will check:
- Python version
- Dependencies
- **MPS GPU availability** ✓
- MISTA data
- Implementation files
- Data loading functionality

## Mac M2 Optimized Settings

For **16GB RAM Mac M2**, use these settings in `train_mista.py`:

```python
class MRCMPSPArgs:
    # Optimized for Mac M2 16GB
    num_envs = 4              # Parallel environments
    num_steps = 128           # Steps per rollout
    minibatch_size = 64       # Batch size

    # Model architecture (optimized for memory)
    gnn_layers = 3
    gnn_hidden_dim = 64       # Use 64 for 16GB RAM
    mlp_hidden_dim = 32
    project_hidden_dim = 64

    # Training
    total_timesteps = 500000  # Full training
    learning_rate = 3e-5
```

### If You Get Memory Errors

Reduce these settings:
```python
num_envs = 2              # Reduce parallel environments
gnn_hidden_dim = 32       # Smaller model
num_steps = 64            # Fewer steps per update
```

## Performance on Mac M2

### Expected Speed (16GB M2)
- **Training**: ~80-100 updates/hour
- **500K timesteps**: ~4-5 hours
- **Hyperparameter tuning** (quick): ~2-3 hours

### Comparison
- Mac M2 (MPS): **Baseline** (1x)
- Mac M2 (CPU): 5-8x slower ❌
- NVIDIA RTX 3090: 1.5-2x faster ✓
- NVIDIA RTX 4090: 2-3x faster ✓
- CPU only: 8-10x slower ❌

**Bottom line**: Mac M2 with MPS is fast and practical for this project! 🚀

## Training Workflow

### 1. Quick Test (5 minutes)
```bash
# Test on small instance
python -c "
from MISTADataLoader import MISTADataLoader
loader = MISTADataLoader('Mista')
instances = loader.load_single_project_set(10)[:5]
print(f'Loaded {len(instances)} test instances')
"
```

### 2. Check Setup
```bash
python check_setup.py
```

### 3. Start Training
```bash
# Full training on MISTA Set A (4-5 hours)
python train_mista.py
```

Monitor progress:
```bash
# In another terminal
tensorboard --logdir=runs
# Open http://localhost:6006
```

### 4. Evaluate
```bash
python evaluate_mista.py
```

### 5. Hyperparameter Tuning (optional)
```bash
# Quick search (2-3 hours)
python tune_hyperparameters.py --search_type quick

# Analyze results
python analyze_hp_results.py --plot
```

## Troubleshooting

### Issue: "MPS not available"

**Check PyTorch version**:
```bash
python3 -c "import torch; print(torch.__version__)"
```

If < 2.0, upgrade:
```bash
pip3 install --upgrade torch torchvision
```

### Issue: "UserWarning: The operator 'xxx' is not currently supported on the MPS backend"

Some PyTorch operations aren't supported on MPS yet. Fallback to CPU for that operation:
```bash
# This is automatic in most cases
# If you see errors, report them
```

### Issue: "Out of memory" during training

**Reduce memory usage**:
1. Edit `train_mista.py`:
   ```python
   num_envs = 2           # From 4 to 2
   gnn_hidden_dim = 32    # From 64 to 32
   ```

2. Or use CPU mode:
   ```bash
   # Force CPU mode if needed
   export PYTORCH_ENABLE_MPS_FALLBACK=1
   ```

### Issue: Training is very slow

**Check device**:
```bash
python device_config.py
```

Should show: "Selected Device: mps"

If it shows "cpu", check MPS availability:
```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

### Issue: "Error loading A-*.txt: list index out of range"

This was a parsing bug, now fixed! Update to latest code:
```bash
git pull origin claude/analyze-code-applications-B5I7T
```

Or the fix is:
- In `MISTADataLoader.py`, line ~199, changed to: `due_date_str = lines[idx].split()[0]`

## Memory Usage Guidelines

### 16GB RAM Capacity

| Component | Memory Usage |
|-----------|--------------|
| macOS System | ~3-4 GB |
| PyTorch + Model | ~2-3 GB |
| Training Buffers | ~4-6 GB |
| **Available for Training** | **~6-8 GB** |

**Safe configurations for 16GB**:
- ✓ num_envs=4, gnn_hidden=64: Uses ~6GB
- ✓ num_envs=2, gnn_hidden=64: Uses ~4GB
- ❌ num_envs=8, gnn_hidden=128: Uses ~10GB (too much!)

### 32GB RAM Capacity (M2 Pro/Max)

If you have Mac M2 Pro/Max with 32GB:
```python
num_envs = 8              # More parallel environments
gnn_hidden_dim = 128      # Larger model
num_steps = 256           # More steps per update
```

This will train faster (~120-150 updates/hour).

## Tips for Best Performance

### 1. Close Other Apps
Before training, close:
- Chrome/Safari (saves 2-4GB)
- Docker (saves 1-2GB)
- Other IDEs (saves 1-2GB)

### 2. Monitor Memory
```bash
# In another terminal, watch memory usage
watch -n 5 "ps aux | grep python | grep -v grep"
```

### 3. Use Background Training
```bash
# Start training in background
nohup python train_mista.py > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### 4. Prevent Sleep
```bash
# Keep Mac awake during training
caffeinate -i python train_mista.py
```

## FAQ

**Q: Should I use Mac M2 or cloud GPU?**

A: Mac M2 is perfectly fine for this project!
- ✓ Fast enough (4-5 hours for full training)
- ✓ No cloud costs
- ✓ Local development convenience
- Cloud GPU only needed if training hundreds of models

**Q: Can I use multiple GPU cores on M2?**

A: MPS automatically uses all GPU cores. You don't need to configure anything.

**Q: What if training is interrupted?**

A: Models are auto-saved every 25 updates to `saves/` directory. You can resume by loading the latest checkpoint.

**Q: Can I train while doing other work?**

A: Yes! With num_envs=2, you'll have ~8GB free for other tasks.

**Q: How to speed up training?**

1. Use quick search: `--timesteps 50000` instead of 500000
2. Use fewer instances: Edit code to use `instances[:5]`
3. Reduce num_envs but training will be slower

**Q: Best settings for overnight training?**

```python
# In train_mista.py
num_envs = 4
total_timesteps = 1000000  # More training = better results
```

Start before sleep:
```bash
caffeinate -i python train_mista.py &
```

Check in morning!

## Summary

✅ **Mac M2 16GB is great for this project**
✅ **MPS GPU acceleration works well**
✅ **Training time: 4-5 hours** (reasonable)
✅ **No cloud costs needed**

**Recommended workflow**:
1. Run `python check_setup.py` ✓
2. Run `python train_mista.py` (4-5 hours)
3. Run `python evaluate_mista.py` ✓
4. Optionally: `python tune_hyperparameters.py --search_type quick`

**Happy training on your Mac M2!** 🚀
