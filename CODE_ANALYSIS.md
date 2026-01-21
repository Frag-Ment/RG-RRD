# RL-GNN for RCPSP: Code Analysis

## Overview

This codebase implements a **Reinforcement Learning (RL) + Graph Neural Network (GNN)** approach to solve the Resource-Constrained Project Scheduling Problem (RCPSP) and its variant RCPSP with Resource Disruptions (RCPSP-RD). It uses **Proximal Policy Optimization (PPO)** to train an agent that learns to schedule tasks optimally.

**Paper**: https://doi.org/10.1016/j.rcim.2023.102628

---

## How It Works

### 1. **Problem Representation**

**RCPSP Components**:
- **Tasks/Nodes**: Activities with duration and resource requirements (4 resource types)
- **Dependencies**: Directed acyclic graph (DAG) showing precedence constraints
- **Resources**: Limited renewable resources shared across tasks
- **Objective**: Minimize project makespan (total completion time)

**Data Structure** (`jobdag.py`, `node.py`):
```
- Adjacency Matrix: Task dependencies (n x n)
- Feature Matrix: [duration, res1, res2, res3, res4, status_flags]
- Resource Vector: [available_res1, res2, res3, res4]
```

### 2. **Architecture Components**

#### A. **Environment** (`normal_env.py`, `skip_res_env.py`)
Simulates project execution:
- **State**: (adjacency_matrix, feature_matrix, resource_availability, runnable_tasks)
- **Action**: Select which task to schedule next from runnable tasks
- **Reward**: Negative time elapsed (encourages faster completion)
- **Done**: When all tasks completed

Key method:
```python
def step(action):
    1. Wait until resources available & predecessors complete
    2. Assign task to execution
    3. Update resource availability
    4. Unlock child tasks when predecessors finish
    5. Return (next_state, -time_elapsed, done)
```

#### B. **GNN Feature Extractor** (`graphcnn.py`)
Extracts graph features from task network:
- **Input**: Task features (7D) + Adjacency matrix
- **Layers**: 3 GNN layers with MLP transformations
- **Pooling**: Graph-level pooling for global state
- **Output**:
  - `h_nodes`: Node embeddings (64D per task)
  - `h_pooled`: Global graph embedding (64D)

**Message Passing**:
```python
h^(l+1) = LeakyReLU(MLP(Adjacency × h^(l)))
```
Each layer aggregates neighbor information to capture task relationships.

#### C. **Actor-Critic Agent** (`actor_critic.py`)
Decides which task to schedule:

**Critic (Value Network)**:
- Input: `[global_graph_embedding, resource_state]`
- Output: State value estimate
- Purpose: Evaluate how good current state is

**Actor (Policy Network)**:
- Input: `[task_embedding, global_graph_embedding, resource_state]` for each runnable task
- Output: Probability distribution over runnable tasks
- Purpose: Select best task to schedule

**Action Selection**:
```python
1. Extract features for all runnable tasks
2. Concatenate: [task_features, global_features, resources]
3. Pass through MLP → get scores
4. Softmax → probability distribution
5. Sample action or take argmax (greedy)
```

#### D. **PPO Training** (`ppo.py`)
Standard PPO algorithm:
1. **Rollout**: Collect trajectories from parallel environments
2. **Compute Advantages**: GAE (Generalized Advantage Estimation)
3. **Update Policy**: Clipped surrogate objective
4. **Update Value**: MSE loss on value predictions

**Hyperparameters**:
- 8 parallel environments
- 32 steps per rollout
- Learning rate: 1e-5
- Clip coefficient: 0.1
- Entropy bonus: 0.01

---

## Key Mechanisms

### 1. **Sequential Decision Making**
Agent makes decisions at discrete time points:
- When task completes → resources freed → decision point
- Agent selects next task from available tasks
- Simulator advances time until task can start

### 2. **Masking Invalid Actions**
Only tasks with:
- All predecessors completed/assigned
- Sufficient resources available
are selectable (via `runable_nodes_idx`)

### 3. **Resource Disruptions (RCPSP-RD)**
Variant handles dynamic resource changes:
- `resource_variant`: Time-varying resource availability
- Executor checks resource status at each time step
- Agent must adapt to changing constraints

### 4. **Reward Shaping**
```python
reward = -(time_elapsed) / 10  # Normalized negative time
```
Encourages shorter schedules by penalizing delays.

---

## Adapting to Other Problems

### 1. **RCMPSP (Resource-Constrained Multi-Project Scheduling)**

**Modifications Needed**:

**A. Environment Changes**:
```python
class RCMPSP_Environment:
    def __init__(self):
        self.projects = []  # Multiple project DAGs
        self.shared_resources = []

    def reset(self, project_list, shared_resources):
        # Load multiple projects
        self.jobdags = [Jobdag(proj) for proj in project_list]

    def step(self, action):
        # action = (project_id, task_id)
        project, task = action
        # Same logic but track which project
```

**B. State Representation**:
- **Multi-graph input**: List of adjacency matrices
- **Project features**: Add project ID embedding
- **Global resource pool**: Shared across all projects

**C. GNN Modifications**:
```python
# Option 1: Separate graphs with shared resource encoding
h_projects = [gnn(proj) for proj in projects]
h_global = aggregate(h_projects, shared_resources)

# Option 2: Super-graph with project-level nodes
# Add "project nodes" that connect to all tasks in that project
```

**D. Action Space**:
- Expand to (project_id, task_id) tuples
- Mask per project: only show runnable tasks per project
- Resource conflicts handled globally

---

### 2. **RCMPSP-RD (with Release Dates/Deadlines)**

**Additional Features**:

**A. Node Features (add to feature matrix)**:
```python
feature_matrix = [
    duration, res1, res2, res3, res4,
    release_date,    # Earliest start time
    deadline,        # Latest finish time
    slack_time       # deadline - earliest_finish
]
```

**B. Environment Constraints**:
```python
def step(self, action):
    # Wait until BOTH:
    # 1. Resources available
    # 2. Current time >= release_date
    while (not resources_ok or
           self.walltime < task.release_date):
        self.advance_time()

    # Check deadline violation
    if finish_time > task.deadline:
        reward -= penalty
```

**C. Time-Based Features**:
Add current time to state:
```python
state = (adj, features, resources, current_time, runnable_tasks)
```

**D. Reward Function**:
```python
reward = -time_elapsed
if deadline_violated:
    reward -= 1000  # Heavy penalty
if early_completion:
    reward += bonus  # Encourage early finish
```

---

### 3. **Mixed Model Assembly Line Balancing**

**Problem Differences**:
- **Cyclical**: Repeating production cycles
- **Workstations**: Fixed positions, not just resource pools
- **Cycle time**: Target time per production cycle
- **Task-station assignment**: Tasks assigned to stations

**Adaptation Strategy**:

**A. Problem Formulation**:
- **Nodes**: Assembly tasks
- **Edges**: Precedence constraints
- **Resources**: Workstations (discrete positions)
- **Objective**: Minimize cycle time OR minimize stations

**B. State Representation**:
```python
state = {
    'task_graph': adjacency_matrix,
    'task_features': [duration, precedence_level, zone],
    'station_status': [occupied_time, capacity, location],
    'cycle_time_target': target_time,
    'assigned_tasks': station_assignments
}
```

**C. Action Space**:
```python
# Action: Assign task to station
action = (task_id, station_id)

# Constraints:
# 1. Precedence satisfied
# 2. Station capacity not exceeded
# 3. Cycle time feasible
```

**D. Environment Modifications**:
```python
class AssemblyLineEnv:
    def __init__(self, num_stations, cycle_time):
        self.stations = [Station(i) for i in range(num_stations)]
        self.cycle_time_target = cycle_time

    def step(self, action):
        task, station = action
        # Check if assignment valid
        if not self.check_precedence(task):
            return state, -penalty, False

        # Assign task
        self.stations[station].add_task(task)

        # Check cycle time constraint
        max_time = max(s.total_time for s in self.stations)
        if max_time > self.cycle_time_target:
            reward = -max_time
        else:
            reward = 1  # Valid assignment

        done = all_tasks_assigned()
        return state, reward, done
```

**E. GNN Architecture**:
- **Node features**: Task properties + spatial info
- **Edge features**: Precedence type (direct/indirect)
- **Station embedding**: Separate encoder for workstation states
```python
task_features = GNN(task_graph)
station_features = MLP(station_states)
combined = concat(task_features, station_features)
action_scores = Actor(combined)
```

**F. Reward Function**:
```python
# Option 1: Minimize cycle time
reward = -max(station_times)

# Option 2: Minimize stations used
reward = -num_stations + balance_bonus

# Option 3: Multi-objective
reward = -w1*cycle_time - w2*num_stations + w3*balance_score
```

**G. Training Differences**:
- **Episode length**: Fixed (all tasks assigned)
- **Constraints**: Hard (invalid actions heavily penalized)
- **Symmetry**: Stations are interchangeable (need symmetry breaking)

---

## Key Code Modifications Template

### For Any New Problem:

1. **Define Problem Structure**:
```python
# In new_problem/problem.py
class NewProblem:
    def __init__(self, problem_data):
        self.graph = build_graph(problem_data)
        self.constraints = extract_constraints(problem_data)
        self.resources = initialize_resources(problem_data)
```

2. **Create Environment**:
```python
# In new_problem/environment.py
class NewEnvironment:
    def reset(self, problem_instance):
        # Initialize state
        return initial_state

    def step(self, action):
        # Validate action
        # Update state
        # Compute reward
        return next_state, reward, done
```

3. **Adapt Feature Extraction**:
```python
# Modify actor_critic.py
def get_action_and_value(self, state):
    # Extract problem-specific features
    graph_features = self.feature_extract(state.graph)
    context_features = self.encode_context(state.context)

    # Combine for decision
    combined = concat(graph_features, context_features)
    action_probs = self.actor(combined)
    value = self.critic(combined)
```

4. **Adjust Hyperparameters**:
```python
# In Params.py
parser.add_argument("--problem_specific_param", ...)
```

---

## Practical Tips

1. **Start Simple**: Begin with basic RCPSP, then add constraints
2. **Validate Environment**: Test with heuristic policies first
3. **Feature Engineering**: Problem-specific features crucial for GNN
4. **Reward Shaping**: Dense rewards learn faster than sparse
5. **Constraint Handling**: Hard constraints via masking, soft via penalties
6. **Baseline Comparison**: Implement priority rules (EDD, SPT) for comparison

---

## Strengths & Limitations

**Strengths**:
- ✅ Generalizes to unseen problem instances
- ✅ Leverages graph structure naturally
- ✅ Handles dynamic environments (resource disruptions)
- ✅ Faster than exact methods for large instances

**Limitations**:
- ❌ Limited receptive field (3 GNN layers)
- ❌ Struggles with very long-range dependencies
- ❌ Requires significant training data
- ❌ No optimality guarantees

**Suggested Improvements** (from authors):
- Use deeper GNN architectures (MixHop, GraphSAINT)
- Add attention mechanisms for long-range dependencies
- Combine with supervised learning on optimal solutions
- Hybrid approaches: RL for hard decisions, heuristics for simple ones

---

## File Structure Reference

```
RL-RCPSP/
├── ppo.py                    # Main training loop (standard RCPSP)
├── ppo_for_variant.py        # Training for RCPSP-RD
├── Params.py                 # Hyperparameters
├── DataLoader.py             # Load PSPLIB instances
├── model/
│   ├── actor_critic.py       # Agent architecture
│   ├── graphcnn.py           # GNN feature extractor
│   └── mlp.py               # MLP components
├── rcpsp_simulator/
│   ├── normal_env.py         # Standard RCPSP environment
│   ├── skip_res_env.py       # RCPSP-RD environment
│   ├── jobdag.py            # Project graph structure
│   ├── node.py              # Task/activity class
│   └── normal_executor.py    # Simulation executor
└── test_rcpsp.py            # Evaluation script
```

---

## Quick Start for Adaptation

1. **Copy base environment**: `cp normal_env.py new_problem_env.py`
2. **Modify state/action/reward**: Update `reset()` and `step()`
3. **Adjust features**: Change feature dimensions in `actor_critic.py`
4. **Update data loader**: Create loader for your problem format
5. **Test with random policy**: Ensure environment works
6. **Train**: Run `ppo.py` with new environment
7. **Evaluate**: Compare against problem-specific heuristics
