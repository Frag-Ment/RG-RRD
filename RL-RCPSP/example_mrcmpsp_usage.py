"""
Example usage of MRCMPSP implementation
Demonstrates common use cases
"""
import torch
import numpy as np
from MRCMPSPDataLoader import MRCMPSPDataLoader, create_mrcmpsp_dataloaders
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from model.hierarchical_agent import HierarchicalAgent


def example_1_generate_data():
    """Example 1: Generate and inspect MRCMPSP data"""
    print("="*60)
    print("Example 1: Generate MRCMPSP Data")
    print("="*60)

    # Create data loader
    loader = MRCMPSPDataLoader(
        num_instances=10,
        num_projects=3,
        tasks_per_project=15,
        num_modes=3,
        shared_resources=[15, 15, 15, 15],
        seed=42
    )

    print(f"\nGenerated {len(loader)} instances")
    print(f"Statistics: {loader.get_statistics()}")

    # Inspect one instance
    instance = loader.read_instance(0)

    print(f"\nInstance details:")
    print(f"  Number of projects: {len(instance['projects'])}")
    print(f"  Shared resources: {instance['shared_resources']}")

    for i, project in enumerate(instance['projects']):
        print(f"\n  Project {i}:")
        print(f"    ID: {project['project_id']}")
        print(f"    Priority: {project['priority']:.2f}")
        print(f"    Deadline: {project['deadline']:.1f}")
        print(f"    Tasks: {len(project['nodes_info']) - 2}")  # Exclude dummy nodes
        print(f"    Adjacency shape: {project['adj_mat'].shape}")

        # Show first task modes
        first_task = project['nodes_info'][1]  # Skip dummy start
        print(f"    First task modes:")
        for mode in first_task['modes']:
            print(f"      Mode {mode['mode_id']}: Duration={mode['duration']}, "
                  f"Resources={mode['resources']}")

    # Save data
    loader.save('data/mrcmpsp_example.pkl')
    print("\n✓ Data saved to data/mrcmpsp_example.pkl")


def example_2_run_environment():
    """Example 2: Run environment with random policy"""
    print("\n" + "="*60)
    print("Example 2: Run MRCMPSP Environment")
    print("="*60)

    # Create instance
    loader = MRCMPSPDataLoader(
        num_instances=1,
        num_projects=2,
        tasks_per_project=10,
        num_modes=2
    )
    instance = loader.read_instance(0)

    # Initialize environment
    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    print(f"\nInitial state:")
    print(f"  Active projects: {state.active_projects}")
    print(f"  Feasible actions: {len(state.feasible_actions)}")
    print(f"  Shared resources: {state.shared_resources}")

    # Run random policy
    done = False
    steps = 0
    total_reward = 0

    print("\nRunning random policy...")

    while not done and steps < 50:
        # Random action
        if len(state.feasible_actions) == 0:
            print("  No more feasible actions!")
            break

        action = state.feasible_actions[np.random.randint(len(state.feasible_actions))]

        # Execute
        state, reward, done = env.step(action)
        total_reward += reward
        steps += 1

        if steps % 10 == 0:
            info = env.get_info()
            print(f"  Step {steps}: Time={info['walltime']}, "
                  f"Tasks={info['completed_tasks']}/{info['total_tasks']}, "
                  f"Reward={reward:.2f}")

    # Final statistics
    print(f"\nEpisode finished:")
    print(f"  Steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Makespan: {env.executor.get_makespan()}")

    stats = env.executor.get_statistics()
    print(f"  Completed projects: {stats['completed_projects']}")
    print(f"  Total tasks: {stats['total_tasks']}")

    print("\n✓ Environment run complete")


def example_3_create_agent():
    """Example 3: Create and test agent"""
    print("\n" + "="*60)
    print("Example 3: Create Hierarchical Agent")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

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

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    print(f"\nAgent architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Test forward pass
    loader = MRCMPSPDataLoader(num_instances=1, num_projects=2, tasks_per_project=8, num_modes=2)
    instance = loader.read_instance(0)

    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    print(f"\nTesting forward pass...")

    with torch.no_grad():
        # Get action and value
        action, logprob, entropy, value = agent.get_action_and_value(state)

        print(f"  Action: {action}")
        print(f"  Log probability: {logprob.item():.4f}")
        print(f"  Entropy: {entropy.item():.4f}")
        print(f"  Value estimate: {value.item():.4f}")

        # Test greedy selection
        greedy_action = agent.get_action(state, greedy=True)
        print(f"  Greedy action: {greedy_action}")

    print("\n✓ Agent test complete")


def example_4_agent_rollout():
    """Example 4: Full episode with agent"""
    print("\n" + "="*60)
    print("Example 4: Agent Rollout")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create agent
    agent = HierarchicalAgent(
        project_feature_dim=8,
        project_hidden_dim=32,
        num_gnn_layers=2,
        gnn_hidden_dim=32,
        mlp_hidden_dim=16,
        node_feature_dim=9,
        device=device
    ).to(device)

    # Create environment
    loader = MRCMPSPDataLoader(num_instances=1, num_projects=3, tasks_per_project=12, num_modes=3)
    instance = loader.read_instance(0)

    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    print(f"\nRunning agent policy (untrained)...")
    print(f"Problem: {len(instance['projects'])} projects, "
          f"{sum(len(p['nodes_info'])-2 for p in instance['projects'])} total tasks")

    # Run episode
    done = False
    steps = 0
    episode_data = {
        'states': [],
        'actions': [],
        'rewards': [],
        'values': []
    }

    while not done and steps < 100:
        with torch.no_grad():
            action, logprob, entropy, value = agent.get_action_and_value(state)

            if action is None:
                print("  No feasible actions remaining")
                break

            episode_data['states'].append(state)
            episode_data['actions'].append(action)
            episode_data['values'].append(value.item())

            state, reward, done = env.step(action)
            episode_data['rewards'].append(reward)

            steps += 1

            if steps % 10 == 0:
                info = env.get_info()
                print(f"  Step {steps}: Projects active={info['num_active_projects']}, "
                      f"Tasks={info['completed_tasks']}/{info['total_tasks']}, "
                      f"Time={info['walltime']}")

    # Results
    print(f"\nEpisode results:")
    print(f"  Steps taken: {steps}")
    print(f"  Makespan: {env.executor.get_makespan()}")
    print(f"  Average reward: {np.mean(episode_data['rewards']):.2f}")
    print(f"  Average value estimate: {np.mean(episode_data['values']):.2f}")

    # Action distribution
    projects_selected = [a[0] for a in episode_data['actions']]
    print(f"\n  Project selection distribution:")
    for proj_id in range(len(instance['projects'])):
        count = projects_selected.count(proj_id)
        print(f"    Project {proj_id}: {count} times ({100*count/steps:.1f}%)")

    print("\n✓ Rollout complete")


def example_5_compare_policies():
    """Example 5: Compare random vs greedy agent"""
    print("\n" + "="*60)
    print("Example 5: Policy Comparison")
    print("="*60)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Create agent
    agent = HierarchicalAgent(
        project_feature_dim=8,
        project_hidden_dim=32,
        num_gnn_layers=2,
        gnn_hidden_dim=32,
        mlp_hidden_dim=16,
        node_feature_dim=9,
        device=device
    ).to(device)

    # Test on multiple instances
    loader = MRCMPSPDataLoader(num_instances=5, num_projects=2, tasks_per_project=10, num_modes=2)

    random_makespans = []
    greedy_makespans = []

    print("\nTesting on 5 instances...")

    for i in range(5):
        instance = loader.read_instance(i)

        # Random policy
        env = MRCMPSPEnvironment()
        state = env.reset(instance['projects'], instance['shared_resources'])
        done = False
        steps = 0

        while not done and steps < 100:
            if len(state.feasible_actions) == 0:
                break
            action = state.feasible_actions[np.random.randint(len(state.feasible_actions))]
            state, reward, done = env.step(action)
            steps += 1

        random_makespans.append(env.executor.get_makespan())

        # Greedy agent policy
        env = MRCMPSPEnvironment()
        state = env.reset(instance['projects'], instance['shared_resources'])
        done = False
        steps = 0

        with torch.no_grad():
            while not done and steps < 100:
                action = agent.get_action(state, greedy=True)
                if action is None:
                    break
                state, reward, done = env.step(action)
                steps += 1

        greedy_makespans.append(env.executor.get_makespan())

        print(f"  Instance {i}: Random={random_makespans[i]}, "
              f"Greedy Agent={greedy_makespans[i]}")

    print(f"\nAverage makespans:")
    print(f"  Random policy: {np.mean(random_makespans):.2f} ± {np.std(random_makespans):.2f}")
    print(f"  Greedy agent (untrained): {np.mean(greedy_makespans):.2f} ± {np.std(greedy_makespans):.2f}")

    improvement = (np.mean(random_makespans) - np.mean(greedy_makespans)) / np.mean(random_makespans) * 100
    print(f"  Improvement: {improvement:.1f}%")

    print("\n✓ Comparison complete")
    print("  Note: Agent is untrained - with training expect 20-30% improvement over random")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MRCMPSP Usage Examples")
    print("="*60)

    try:
        example_1_generate_data()
        example_2_run_environment()
        example_3_create_agent()
        example_4_agent_rollout()
        example_5_compare_policies()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Run training: python ppo_mrcmpsp.py")
        print("  2. Monitor with TensorBoard: tensorboard --logdir=runs")
        print("  3. Check MRCMPSP_README.md for detailed documentation")

    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        import traceback
        traceback.print_exc()
