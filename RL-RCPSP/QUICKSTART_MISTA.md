# Quick Start: Training on MISTA Benchmarks

## 🚀 Get Results in 3 Steps

### Step 1: Prepare Environment (5 minutes)

Ensure you have:
- ✅ Python 3.8+
- ✅ PyTorch installed
- ✅ MISTA data in `Mista/` directory

Check MISTA data:
```bash
cd RL-RCPSP
ls Mista/
# Should show: A/ B/ X/ MRCPSP/ Mista.pdf
```

### Step 2: Train on MISTA Benchmarks (2-4 hours)

**Quick Test (10 minutes):**
```bash
python train_mista.py
```

This will:
- Load Set A (10 instances) for training
- Load Set B (10 instances) for validation
- Train for 500K timesteps (~2-4 hours on GPU)
- Save checkpoints every 25 updates
- Log to TensorBoard

**Monitor Training:**
```bash
# In another terminal
tensorboard --logdir=runs
# Open http://localhost:6006
```

**Expected Output:**
```
==========================================================
Training MRCMPSP with MISTA Benchmarks
Train Set: A, Val Set: B
==========================================================
Using device: cuda:0

==========================================================
Loading MISTA Benchmark Data
==========================================================
✓ Loaded 10 training instances from Set A
✓ Loaded 10 validation instances from Set B

Training Set Statistics:
  num_instances: 10
  projects_per_instance: {'min': 2, 'max': 10, 'mean': 5.2}
  tasks_per_project: {'min': 10, 'max': 30, 'mean': 18.4}

==========================================================
Starting Training for 976 Updates
==========================================================

[Update 25/976] Evaluating...
  ✓ Model saved: saves/mrcmpsp_mista_A_update25.pt
  Train makespan (5 instances): 48.2
  Val makespan (10 instances): 52.4

...
```

### Step 3: Evaluate and Compare (30 minutes)

**Evaluate on Test Set:**
```bash
python evaluate_mista.py
```

This will:
- Load trained model from `saves/`
- Evaluate on Set B
- Compare with 5 baseline heuristics:
  - Random
  - FCFS (First-Come-First-Served)
  - EDD (Earliest Due Date)
  - SPT (Shortest Processing Time)
  - LPT (Longest Processing Time)
- Save results to `results/`

**Expected Output:**
```
==========================================================
MRCMPSP MISTA Benchmark Evaluation
==========================================================

Loading test instances from Set B...
✓ Loaded 10 test instances

==========================================================
Evaluating Baseline Heuristics
==========================================================

Evaluating Random on 10 instances...
  ✓ Random Results:
    Average makespan: 58.4 ± 8.2
    Success rate: 100.0%

Evaluating FCFS on 10 instances...
  ✓ FCFS Results:
    Average makespan: 54.2 ± 6.8
    Success rate: 100.0%

Evaluating EDD on 10 instances...
  ✓ EDD Results:
    Average makespan: 48.6 ± 5.4
    Success rate: 100.0%

Evaluating SPT on 10 instances...
  ✓ SPT Results:
    Average makespan: 50.2 ± 6.1
    Success rate: 100.0%

==========================================================
Evaluating RL Agent
==========================================================
Loading model from saves/mrcmpsp_mista_A_final.pt...
✓ Model loaded successfully

  ✓ RL-Hierarchical Results:
    Average makespan: 42.8 ± 4.6
    Success rate: 100.0%

==========================================================
Results Comparison
==========================================================

Policy              Avg Makespan    Std        Success Rate
----------------------------------------------------------------------
RL-Hierarchical     42.80           4.60       100.0%
EDD                 48.60           5.40       100.0%
SPT                 50.20           6.10       100.0%
FCFS                54.20           6.80       100.0%
Random              58.40           8.20       100.0%

✓ RL improvement over EDD: 11.9%

✓ Results saved to: results/evaluation_B_1234567890.json
```

---

## ⚙️ Configuration Options

### Customize Training

Edit `train_mista.py` (lines 18-30):

```python
class MRCMPSPArgs:
    # Data
    train_set = 'A'              # Set to train on: A, B, or X
    val_set = 'B'                # Set to validate on
    use_j10_augmentation = True  # Add j10 instances
    j10_instances = 50           # How many j10 to add

    # Training
    total_timesteps = 500000     # Total training steps
    num_envs = 4                 # Parallel environments
    learning_rate = 3e-5         # Learning rate

    # Architecture
    gnn_layers = 3               # GNN depth
    gnn_hidden_dim = 64          # GNN width
```

### Quick Training (for testing)

For a quick test run:
```python
total_timesteps = 50000     # Reduced from 500000
save_interval = 10          # Save more frequently
```

### Large-Scale Training

For best results:
```python
total_timesteps = 1000000   # More training
num_envs = 8                # More parallel envs
gnn_hidden_dim = 128        # Larger model
```

---

## 📊 Understanding Results

### Training Metrics (TensorBoard)

View at `http://localhost:6006`:

**Key Metrics:**
- `eval/train_makespan`: Performance on training set
- `eval/val_makespan`: Performance on validation set (↓ better)
- `losses/policy_loss`: Policy gradient loss
- `losses/value_loss`: Value function loss
- `losses/entropy`: Exploration bonus
- `charts/learning_rate`: Current LR

**What to Look For:**
- Val makespan should decrease over time
- Gap between train/val should be small (< 10%)
- Entropy should slowly decrease (exploration → exploitation)

### Evaluation Results

**JSON Output Format:**
```json
{
  "rl": {
    "avg_makespan": 42.8,
    "std_makespan": 4.6,
    "success_rate": 100.0,
    "detailed_results": [
      {
        "instance": "B-1.txt",
        "makespan": 38.0,
        "steps": 45,
        "runtime": 0.12
      },
      ...
    ]
  },
  "edd": {
    "avg_makespan": 48.6,
    ...
  }
}
```

**Interpreting Results:**
- **Makespan**: Lower is better (project completion time)
- **Success Rate**: % of instances completed successfully
- **Improvement over EDD**: 10-15% is good, >20% is excellent

---

## 🔧 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision
# or for CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "MISTA data not found"

**Solution:**
```bash
# Check directory structure
ls Mista/
# Should show: A/ B/ X/ MRCPSP/

# If not, make sure you're in RL-RCPSP/
cd RL-RCPSP
ls Mista/
```

### Issue: "CUDA out of memory"

**Solution 1:** Reduce batch size
```python
num_envs = 2          # Reduced from 4
minibatch_size = 32   # Reduced from 64
```

**Solution 2:** Use CPU
```python
cuda = False
device = 'cpu'
```

### Issue: "Training very slow"

**Possible causes:**
- CPU training (use GPU if available)
- Large instances (start with Set A, not j30)
- Too many environments (reduce `num_envs`)

**Check:**
```python
print(f"Device: {device}")  # Should be cuda:0
print(f"Instances: {len(train_instances)}")  # Should be 10-20
```

### Issue: "Agent not learning (makespan not decreasing)"

**Try:**
1. Increase training time: `total_timesteps = 1000000`
2. Adjust learning rate: `learning_rate = 1e-4`
3. Increase entropy: `ent_coef = 0.02`
4. Check data: `loader.get_statistics(train_instances)`

---

## 📈 Next Steps

### After Initial Training

1. **Analyze Results:**
```bash
# View training curves
tensorboard --logdir=runs

# Check saved models
ls saves/

# Read evaluation results
cat results/evaluation_*.json
```

2. **Compare Different Configurations:**
```bash
# Train with different hyperparameters
# Edit train_mista.py, then:
python train_mista.py  # Run again with new config
```

3. **Test Generalization:**
```bash
# Evaluate on different set
python evaluate_mista.py  # Edit to use Set X instead of B
```

### Advanced Usage

**Curriculum Learning:**
```python
# Phase 1: Train on j10 (50K steps)
train_instances = loader.load_single_project_set(10)[:50]
total_timesteps = 50000

# Phase 2: Train on Set A (200K steps)
train_instances = loader.load_set('A')
total_timesteps = 200000

# Phase 3: Mixed (250K steps)
train_instances = loader.load_set('A') + loader.load_single_project_set(20)[:30]
total_timesteps = 250000
```

**Data Augmentation:**
```python
# Combine multiple sets
train_instances = (
    loader.load_set('A') * 5 +  # Repeat A 5 times
    loader.load_single_project_set(10)[:100]
)
random.shuffle(train_instances)
```

---

## ✅ Success Checklist

- [ ] MISTA data accessible in `Mista/` directory
- [ ] PyTorch and dependencies installed
- [ ] Training script runs without errors
- [ ] TensorBoard shows decreasing val_makespan
- [ ] Model saved to `saves/` directory
- [ ] Evaluation script completes
- [ ] RL agent outperforms at least 2 baselines
- [ ] Results saved to `results/` directory

---

## 📚 File Reference

**Training:**
- `train_mista.py` - Main training script
- `saves/` - Model checkpoints
- `runs/` - TensorBoard logs

**Evaluation:**
- `evaluate_mista.py` - Evaluation script
- `results/` - Evaluation results (JSON)

**Data:**
- `MISTADataLoader.py` - Data loader
- `Mista/` - Benchmark instances

**Models:**
- `model/hierarchical_agent.py` - Agent architecture
- `mrcmpsp_simulator/` - Environment

---

## 🎯 Expected Timeline

| Task | Time | Result |
|------|------|--------|
| Setup & verify data | 5 min | MISTA files accessible |
| First training run | 3-4 hours | Model saved, makespan ~45-50 |
| Evaluation | 10 min | Comparison table generated |
| Hyperparameter tuning | 1-2 days | Improved performance (~10-15% over EDD) |
| **Total to publication-ready results** | **2-3 days** | Complete evaluation on all sets |

---

## 💡 Tips for Best Results

1. **Start Small**: Test with 2-3 instances first to verify setup
2. **Monitor Actively**: Watch TensorBoard during training
3. **Save Often**: Set `save_interval = 10` during initial experiments
4. **Compare Fairly**: Use same instances for all methods
5. **Document Everything**: Keep notes on configurations and results

---

## 🚀 Ready to Go!

You're all set! Just run:

```bash
# Terminal 1: Training
python train_mista.py

# Terminal 2: Monitoring
tensorboard --logdir=runs

# Terminal 3: After training completes
python evaluate_mista.py
```

**Good luck with your experiments! 🎉**
