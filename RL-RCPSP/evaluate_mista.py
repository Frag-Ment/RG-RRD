"""
Evaluate Trained MRCMPSP Agent on MISTA Benchmarks
Compare against baseline heuristics
"""
import torch
import numpy as np
import json
import time
from pathlib import Path
from MISTADataLoader import MISTADataLoader
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from model.hierarchical_agent import HierarchicalAgent


class BaselineHeuristics:
    """
    Baseline heuristic policies for MRCMPSP
    """

    @staticmethod
    def random_policy(state):
        """Random action selection"""
        if len(state.feasible_actions) == 0:
            return None
        return state.feasible_actions[np.random.randint(len(state.feasible_actions))]

    @staticmethod
    def fcfs_policy(state):
        """First-Come-First-Served: select first available action"""
        if len(state.feasible_actions) == 0:
            return None
        return state.feasible_actions[0]

    @staticmethod
    def edd_policy(state):
        """Earliest Due Date: prioritize projects with earliest deadline"""
        if len(state.feasible_actions) == 0:
            return None

        # Group actions by project
        projects_actions = {}
        for proj_id, task_id, mode_id in state.feasible_actions:
            if proj_id not in projects_actions:
                projects_actions[proj_id] = []
            projects_actions[proj_id].append((task_id, mode_id))

        # Find project with earliest deadline
        project_features = state.project_features
        min_deadline_proj = None
        min_deadline = float('inf')

        for proj_id in projects_actions.keys():
            if proj_id < len(project_features):
                deadline = project_features[proj_id][4]  # deadline is 5th feature
                if deadline < min_deadline:
                    min_deadline = deadline
                    min_deadline_proj = proj_id

        if min_deadline_proj is None:
            min_deadline_proj = list(projects_actions.keys())[0]

        # Select first task from that project, fastest mode
        task_id, mode_id = projects_actions[min_deadline_proj][0]
        return (min_deadline_proj, task_id, mode_id)

    @staticmethod
    def spt_policy(state):
        """Shortest Processing Time: select task with shortest duration"""
        if len(state.feasible_actions) == 0:
            return None

        min_duration = float('inf')
        best_action = None

        for proj_id, task_id, mode_id in state.feasible_actions:
            # Get duration from node features
            # Feature 0 is average duration
            proj_features = state.project_node_features[proj_id]
            duration = proj_features[task_id][0]

            if duration < min_duration:
                min_duration = duration
                best_action = (proj_id, task_id, mode_id)

        return best_action if best_action else state.feasible_actions[0]

    @staticmethod
    def lpt_policy(state):
        """Longest Processing Time: select task with longest duration"""
        if len(state.feasible_actions) == 0:
            return None

        max_duration = -1
        best_action = None

        for proj_id, task_id, mode_id in state.feasible_actions:
            proj_features = state.project_node_features[proj_id]
            duration = proj_features[task_id][0]

            if duration > max_duration:
                max_duration = duration
                best_action = (proj_id, task_id, mode_id)

        return best_action if best_action else state.feasible_actions[0]


def evaluate_policy(env_class, instances, policy_fn, policy_name, max_steps=1000):
    """
    Evaluate a policy on a set of instances

    Args:
        env_class: Environment class
        instances: List of problem instances
        policy_fn: Policy function (state -> action)
        policy_name: Name for logging
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with results
    """
    results = []

    print(f"\nEvaluating {policy_name} on {len(instances)} instances...")

    for idx, instance in enumerate(instances):
        env = env_class()
        state = env.reset(instance['projects'], instance['shared_resources'])

        done = False
        steps = 0
        start_time = time.time()

        while not done and steps < max_steps:
            action = policy_fn(state)

            if action is None:
                break

            try:
                state, reward, done = env.step(action)
                steps += 1
            except Exception as e:
                print(f"  Warning: Error in instance {idx}: {e}")
                break

        runtime = time.time() - start_time
        makespan = env.executor.get_makespan() if done else float('inf')

        result = {
            'instance': instance.get('instance_file', f'instance_{idx}'),
            'set': instance.get('set', 'unknown'),
            'num_projects': len(instance['projects']),
            'total_tasks': sum(p['num_jobs']-2 for p in instance['projects']),
            'makespan': makespan,
            'steps': steps,
            'runtime': runtime,
            'completed': done
        }

        results.append(result)

        if (idx + 1) % 5 == 0:
            avg_makespan = np.mean([r['makespan'] for r in results if r['makespan'] != float('inf')])
            print(f"  Progress: {idx+1}/{len(instances)}, Avg makespan: {avg_makespan:.2f}")

    # Calculate statistics
    valid_results = [r for r in results if r['makespan'] != float('inf')]

    if len(valid_results) == 0:
        print(f"  ✗ {policy_name}: No valid results!")
        return {'policy': policy_name, 'results': results, 'avg_makespan': float('inf')}

    avg_makespan = np.mean([r['makespan'] for r in valid_results])
    std_makespan = np.std([r['makespan'] for r in valid_results])
    avg_steps = np.mean([r['steps'] for r in valid_results])
    avg_runtime = np.mean([r['runtime'] for r in valid_results])
    success_rate = len(valid_results) / len(results) * 100

    print(f"\n  ✓ {policy_name} Results:")
    print(f"    Average makespan: {avg_makespan:.2f} ± {std_makespan:.2f}")
    print(f"    Average steps: {avg_steps:.1f}")
    print(f"    Average runtime: {avg_runtime:.3f}s")
    print(f"    Success rate: {success_rate:.1f}%")

    return {
        'policy': policy_name,
        'results': results,
        'avg_makespan': avg_makespan,
        'std_makespan': std_makespan,
        'avg_steps': avg_steps,
        'avg_runtime': avg_runtime,
        'success_rate': success_rate
    }


def evaluate_rl_agent(model_path, instances, device='cuda:0'):
    """Evaluate trained RL agent"""

    print(f"Loading model from {model_path}...")

    # Create agent
    agent = HierarchicalAgent(
        project_feature_dim=8,
        project_hidden_dim=64,
        num_gnn_layers=3,
        gnn_hidden_dim=64,
        mlp_hidden_dim=32,
        node_feature_dim=9,
        num_mlp_layers=2,
        device=device
    ).to(device)

    # Load weights
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    print("✓ Model loaded successfully")

    # Define policy function
    def rl_policy(state):
        with torch.no_grad():
            return agent.get_action(state, greedy=True)

    # Evaluate
    return evaluate_policy(MRCMPSPEnvironment, instances, rl_policy, "RL-Hierarchical")


def main():
    """Main evaluation script"""

    print("="*70)
    print("MRCMPSP MISTA Benchmark Evaluation")
    print("="*70)

    # Configuration
    mista_data_root = 'Mista'
    test_set = 'B'  # Evaluate on Set B
    model_path = 'saves/mrcmpsp_mista_A_final.pt'  # Path to trained model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load test data
    print(f"\nLoading test instances from Set {test_set}...")
    loader = MISTADataLoader(data_root=mista_data_root)

    try:
        test_instances = loader.load_set(test_set)
        print(f"✓ Loaded {len(test_instances)} test instances")
    except Exception as e:
        print(f"✗ Error loading test set: {e}")
        print("Falling back to j10 instances...")
        test_instances = loader.load_single_project_set(10)[:10]
        print(f"✓ Loaded {len(test_instances)} j10 instances")

    # Get statistics
    stats = loader.get_statistics(test_instances)
    print(f"\nTest Set Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Evaluate baselines
    print("\n" + "="*70)
    print("Evaluating Baseline Heuristics")
    print("="*70)

    baseline_results = {}

    # Random
    baseline_results['random'] = evaluate_policy(
        MRCMPSPEnvironment, test_instances,
        BaselineHeuristics.random_policy, "Random"
    )

    # FCFS
    baseline_results['fcfs'] = evaluate_policy(
        MRCMPSPEnvironment, test_instances,
        BaselineHeuristics.fcfs_policy, "FCFS"
    )

    # EDD
    baseline_results['edd'] = evaluate_policy(
        MRCMPSPEnvironment, test_instances,
        BaselineHeuristics.edd_policy, "EDD"
    )

    # SPT
    baseline_results['spt'] = evaluate_policy(
        MRCMPSPEnvironment, test_instances,
        BaselineHeuristics.spt_policy, "SPT"
    )

    # LPT
    baseline_results['lpt'] = evaluate_policy(
        MRCMPSPEnvironment, test_instances,
        BaselineHeuristics.lpt_policy, "LPT"
    )

    # Evaluate RL agent (if model exists)
    print("\n" + "="*70)
    print("Evaluating RL Agent")
    print("="*70)

    if Path(model_path).exists():
        rl_results = evaluate_rl_agent(model_path, test_instances, device)
        baseline_results['rl'] = rl_results
    else:
        print(f"✗ Model not found: {model_path}")
        print("  Train a model first using: python train_mista.py")
        rl_results = None

    # Comparison Table
    print("\n" + "="*70)
    print("Results Comparison")
    print("="*70)

    print(f"\n{'Policy':<20} {'Avg Makespan':<15} {'Std':<10} {'Success Rate':<15}")
    print("-"*70)

    # Sort by makespan
    sorted_policies = sorted(baseline_results.items(),
                            key=lambda x: x[1]['avg_makespan'])

    for policy_name, result in sorted_policies:
        makespan = result['avg_makespan']
        std = result.get('std_makespan', 0)
        success = result.get('success_rate', 100)

        if makespan == float('inf'):
            print(f"{policy_name:<20} {'FAILED':<15} {'-':<10} {success:<15.1f}%")
        else:
            print(f"{policy_name:<20} {makespan:<15.2f} {std:<10.2f} {success:<15.1f}%")

    # Calculate improvements
    if rl_results and 'edd' in baseline_results:
        rl_makespan = rl_results['avg_makespan']
        edd_makespan = baseline_results['edd']['avg_makespan']

        if rl_makespan != float('inf') and edd_makespan != float('inf'):
            improvement = (edd_makespan - rl_makespan) / edd_makespan * 100
            print(f"\n✓ RL improvement over EDD: {improvement:.1f}%")

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / f'evaluation_{test_set}_{int(time.time())}.json'

    # Prepare output (convert numpy types to native Python)
    output_data = {}
    for policy_name, result in baseline_results.items():
        output_data[policy_name] = {
            'avg_makespan': float(result['avg_makespan']),
            'std_makespan': float(result.get('std_makespan', 0)),
            'avg_steps': float(result.get('avg_steps', 0)),
            'avg_runtime': float(result.get('avg_runtime', 0)),
            'success_rate': float(result.get('success_rate', 0)),
            'detailed_results': result['results']
        }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
