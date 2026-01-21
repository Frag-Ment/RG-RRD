"""
Test script for MRCMPSP implementation
Validates environment, agent, and training components
"""
import torch
import numpy as np
from MRCMPSPDataLoader import MRCMPSPDataLoader
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from model.hierarchical_agent import HierarchicalAgent


def test_data_loader():
    """Test data loader"""
    print("Testing Data Loader...")

    loader = MRCMPSPDataLoader(
        num_instances=5,
        num_projects=2,
        tasks_per_project=10,
        num_modes=3
    )

    assert len(loader) == 5, "Wrong number of instances"

    instance = loader.next_instance()
    assert 'projects' in instance
    assert len(instance['projects']) == 2
    assert instance['shared_resources'].shape == (4,)

    print("✓ Data loader test passed")


def test_environment():
    """Test MRCMPSP environment"""
    print("\nTesting Environment...")

    # Create instance
    loader = MRCMPSPDataLoader(
        num_instances=1,
        num_projects=2,
        tasks_per_project=5,
        num_modes=2
    )
    instance = loader.read_instance(0)

    # Create environment
    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    # Check state structure
    assert hasattr(state, 'project_features')
    assert hasattr(state, 'feasible_actions')
    assert len(state.project_graphs) == 2
    assert len(state.active_projects) == 2

    print(f"  Initial feasible actions: {len(state.feasible_actions)}")
    print(f"  Active projects: {state.active_projects}")

    # Take a few steps
    steps = 0
    done = False

    while not done and steps < 20:
        if len(state.feasible_actions) == 0:
            print("  Warning: No feasible actions")
            break

        # Random action
        action = state.feasible_actions[0]
        state, reward, done = env.step(action)

        steps += 1
        print(f"  Step {steps}: Action {action}, Reward {reward:.2f}, Done {done}")

    print(f"  Completed {steps} steps")
    print(f"  Final makespan: {env.executor.get_makespan()}")

    print("✓ Environment test passed")


def test_agent():
    """Test hierarchical agent"""
    print("\nTesting Hierarchical Agent...")

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")

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

    print(f"  Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Create environment and state
    loader = MRCMPSPDataLoader(
        num_instances=1,
        num_projects=2,
        tasks_per_project=5,
        num_modes=2
    )
    instance = loader.read_instance(0)

    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    # Test forward pass
    with torch.no_grad():
        # Get action
        action, logprob, entropy, value = agent.get_action_and_value(state)

        print(f"  Selected action: {action}")
        print(f"  Log prob: {logprob.item():.4f}")
        print(f"  Entropy: {entropy.item():.4f}")
        print(f"  Value: {value.item():.4f}")

        # Get value only
        value_only = agent.get_value(state)
        print(f"  Value (direct): {value_only.item():.4f}")

        # Greedy action
        greedy_action = agent.get_action(state, greedy=True)
        print(f"  Greedy action: {greedy_action}")

    print("✓ Agent test passed")


def test_full_episode():
    """Test full episode with agent"""
    print("\nTesting Full Episode...")

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
    loader = MRCMPSPDataLoader(
        num_instances=1,
        num_projects=2,
        tasks_per_project=10,
        num_modes=3
    )
    instance = loader.read_instance(0)

    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    # Run episode
    done = False
    steps = 0
    total_reward = 0

    print("  Running episode...")

    while not done and steps < 100:
        with torch.no_grad():
            action = agent.get_action(state, greedy=False)

            if action is None:
                print("  Agent returned None action - no feasible actions")
                break

            state, reward, done = env.step(action)
            total_reward += reward
            steps += 1

        if steps % 10 == 0:
            info = env.get_info()
            print(f"    Step {steps}: Time {info['walltime']}, "
                  f"Completed {info['completed_tasks']}/{info['total_tasks']}")

    print(f"\n  Episode finished:")
    print(f"    Steps: {steps}")
    print(f"    Total reward: {total_reward:.2f}")
    print(f"    Makespan: {env.executor.get_makespan()}")
    print(f"    Completed tasks: {env.executor.project_manager.completed_tasks}")

    print("✓ Full episode test passed")


def test_training_step():
    """Test one training step"""
    print("\nTesting Training Step...")

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

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # Create environment
    loader = MRCMPSPDataLoader(num_instances=2, num_projects=2, tasks_per_project=8, num_modes=2)
    instance = loader.read_instance(0)

    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])

    # Collect some transitions
    states = []
    actions_taken = []
    rewards_collected = []

    for _ in range(5):
        with torch.no_grad():
            action = agent.get_action(state, greedy=False)

            if action is None:
                break

            states.append(state)
            actions_taken.append(action)

            state, reward, done = env.step(action)
            rewards_collected.append(reward)

            if done:
                break

    if len(states) == 0:
        print("  No transitions collected, skipping training step")
        return

    # Compute loss
    total_loss = 0

    for i in range(len(states)):
        _, logprob, entropy, value = agent.get_action_and_value(states[i], actions_taken[i])

        # Dummy loss (just to test backward pass)
        loss = -logprob + 0.5 * (value ** 2) - 0.01 * entropy
        total_loss += loss

    avg_loss = total_loss / len(states)

    print(f"  Collected {len(states)} transitions")
    print(f"  Average loss: {avg_loss.item():.4f}")

    # Backward pass
    optimizer.zero_grad()
    avg_loss.backward()
    optimizer.step()

    print("  Backward pass successful")
    print("✓ Training step test passed")


if __name__ == "__main__":
    print("="*60)
    print("MRCMPSP Implementation Test Suite")
    print("="*60)

    try:
        test_data_loader()
        test_environment()
        test_agent()
        test_full_episode()
        test_training_step()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
