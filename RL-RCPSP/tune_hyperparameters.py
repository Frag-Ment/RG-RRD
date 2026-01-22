"""
Automatic Hyperparameter Tuning for MRCMPSP on MISTA Benchmarks

Supports:
- Grid Search: Test all combinations
- Random Search: Test random combinations
- Automatic tracking and comparison
- Best configuration selection
"""
import os
import json
import time
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import copy

from model.hierarchical_agent import HierarchicalAgent
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from MISTADataLoader import MISTADataLoader


class HyperparameterConfig:
    """Define hyperparameter search space"""

    # Architecture parameters
    GNN_LAYERS = [2, 3, 4]
    GNN_HIDDEN_DIM = [32, 64, 128]
    MLP_HIDDEN_DIM = [16, 32, 64]
    PROJECT_HIDDEN_DIM = [32, 64, 128]

    # Training parameters
    LEARNING_RATE = [1e-5, 3e-5, 1e-4]
    CLIP_COEF = [0.1, 0.15, 0.2]
    ENT_COEF = [0.005, 0.01, 0.02]
    VF_COEF = [0.3, 0.5, 0.7]

    # PPO parameters
    NUM_ENVS = [2, 4, 8]
    NUM_STEPS = [64, 128, 256]
    MINIBATCH_SIZE = [32, 64, 128]
    UPDATE_EPOCHS = [3, 4, 5]

    # Fixed parameters (don't tune these)
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    MAX_GRAD_NORM = 0.5

    @classmethod
    def get_default_config(cls):
        """Get default configuration (baseline)"""
        return {
            'gnn_layers': 3,
            'gnn_hidden_dim': 64,
            'mlp_hidden_dim': 32,
            'project_hidden_dim': 64,
            'learning_rate': 3e-5,
            'clip_coef': 0.1,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'num_envs': 4,
            'num_steps': 128,
            'minibatch_size': 64,
            'update_epochs': 4
        }

    @classmethod
    def get_quick_search_space(cls):
        """Quick search space for fast tuning"""
        return {
            'gnn_layers': [2, 3],
            'gnn_hidden_dim': [32, 64],
            'mlp_hidden_dim': [16, 32],
            'project_hidden_dim': [32, 64],
            'learning_rate': [3e-5, 1e-4],
            'clip_coef': [0.1, 0.15],
            'ent_coef': [0.01, 0.02],
            'vf_coef': [0.5],
            'num_envs': [4],
            'num_steps': [128],
            'minibatch_size': [64],
            'update_epochs': [4]
        }

    @classmethod
    def get_full_search_space(cls):
        """Full search space for comprehensive tuning"""
        return {
            'gnn_layers': cls.GNN_LAYERS,
            'gnn_hidden_dim': cls.GNN_HIDDEN_DIM,
            'mlp_hidden_dim': cls.MLP_HIDDEN_DIM,
            'project_hidden_dim': cls.PROJECT_HIDDEN_DIM,
            'learning_rate': cls.LEARNING_RATE,
            'clip_coef': cls.CLIP_COEF,
            'ent_coef': cls.ENT_COEF,
            'vf_coef': cls.VF_COEF,
            'num_envs': cls.NUM_ENVS,
            'num_steps': cls.NUM_STEPS,
            'minibatch_size': cls.MINIBATCH_SIZE,
            'update_epochs': cls.UPDATE_EPOCHS
        }


def generate_grid_search_configs(search_space):
    """Generate all combinations for grid search"""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)

    return configs


def generate_random_search_configs(search_space, n_configs=20, seed=42):
    """Generate random configurations"""
    random.seed(seed)
    np.random.seed(seed)

    configs = []
    for _ in range(n_configs):
        config = {}
        for key, values in search_space.items():
            config[key] = random.choice(values)
        configs.append(config)

    return configs


def train_with_config(config, train_instances, val_instances,
                       timesteps=50000, device='cuda:0', save_dir='hp_tuning'):
    """
    Train model with given hyperparameter configuration

    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*70}")
    print(f"Training with config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"{'='*70}")

    # Create directories
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Create config hash for unique identification
    config_str = '_'.join([f"{k}{v}" for k, v in sorted(config.items())])
    config_hash = str(hash(config_str))[-8:]

    try:
        # Create agent
        agent = HierarchicalAgent(
            project_feature_dim=8,
            project_hidden_dim=config['project_hidden_dim'],
            num_gnn_layers=config['gnn_layers'],
            gnn_hidden_dim=config['gnn_hidden_dim'],
            mlp_hidden_dim=config['mlp_hidden_dim'],
            node_feature_dim=9,
            num_mlp_layers=2,
            device=device
        ).to(device)

        optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)

        # Initialize environments
        envs = []
        for i in range(config['num_envs']):
            instance = train_instances[i % len(train_instances)]
            env = MRCMPSPEnvironment()
            state = env.reset(instance['projects'], instance['shared_resources'])
            envs.append(env)

        # Storage
        num_steps = config['num_steps']
        num_envs = config['num_envs']

        obs = [[None for j in range(num_envs)] for i in range(num_steps)]
        actions = [[None for j in range(num_envs)] for i in range(num_steps)]
        logprobs = torch.zeros((num_steps, num_envs)).to(device)
        rewards = torch.zeros((num_steps, num_envs)).to(device)
        dones = torch.zeros((num_steps, num_envs)).to(device)
        values = torch.zeros((num_steps, num_envs)).to(device)

        # Initialize states
        all_next_obs = [envs[i].executor.get_state() for i in range(num_envs)]
        next_done = torch.zeros(num_envs).to(device)

        # Training metrics
        train_losses = []
        train_rewards = []

        batch_size = num_steps * num_envs
        num_updates = timesteps // batch_size

        print(f"Training for {num_updates} updates...")

        for update in tqdm(range(1, num_updates + 1), desc="Training"):
            # Collect rollout
            for step in range(num_steps):
                obs[step] = copy.deepcopy(all_next_obs)
                dones[step] = copy.deepcopy(next_done)

                # Get actions
                with torch.no_grad():
                    for env_idx in range(num_envs):
                        try:
                            action, logprob, _, value = agent.get_action_and_value(all_next_obs[env_idx])
                            values[step][env_idx] = value.flatten()
                            actions[step][env_idx] = action
                            logprobs[step][env_idx] = logprob
                        except:
                            actions[step][env_idx] = None
                            logprobs[step][env_idx] = 0
                            values[step][env_idx] = 0

                # Execute actions
                all_next_obs = []
                next_done = []

                for env_idx in range(num_envs):
                    action = actions[step][env_idx]

                    if action is None:
                        reward = -10
                        done = 1
                        instance = train_instances[random.randint(0, len(train_instances)-1)]
                        next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])
                    else:
                        try:
                            next_obs, reward, done = envs[env_idx].step(action)
                            reward = reward / 10.0

                            if done:
                                instance = train_instances[random.randint(0, len(train_instances)-1)]
                                next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])
                        except:
                            reward = -10
                            done = 1
                            instance = train_instances[random.randint(0, len(train_instances)-1)]
                            next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])

                    all_next_obs.append(next_obs)
                    rewards[step][env_idx] = torch.tensor(reward).to(device).view(-1)
                    next_done.append(done)

                next_done = torch.tensor(next_done).to(device)

            # Compute advantages
            with torch.no_grad():
                all_next_values = []
                for env_idx in range(num_envs):
                    try:
                        next_value = agent.get_value(all_next_obs[env_idx])
                        all_next_values.append(next_value)
                    except:
                        all_next_values.append(torch.tensor(0.0, device=device))

                all_next_values = torch.stack(all_next_values).to(device).flatten()

                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0

                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = all_next_values
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]

                    delta = rewards[t] + 0.99 * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam

                returns = advantages + values

            # Flatten batch
            b_obs = []
            b_actions = []
            for row_idx in range(len(obs)):
                for col_idx in range(len(obs[0])):
                    b_obs.append(obs[row_idx][col_idx])
                    b_actions.append(actions[row_idx][col_idx])

            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Update policy
            b_inds = np.arange(batch_size)

            for epoch in range(config['update_epochs']):
                np.random.shuffle(b_inds)

                for start in range(0, batch_size, config['minibatch_size']):
                    end = start + config['minibatch_size']
                    mb_inds = b_inds[start:end]

                    newlogprobs = []
                    newvalues = []
                    entropies = []

                    for idx in mb_inds:
                        if b_actions[idx] is None:
                            newlogprobs.append(torch.tensor(0.0, device=device))
                            newvalues.append(b_values[idx])
                            entropies.append(torch.tensor(0.0, device=device))
                        else:
                            try:
                                _, new_lp, ent, new_v = agent.get_action_and_value(
                                    b_obs[idx], b_actions[idx]
                                )
                                newlogprobs.append(new_lp)
                                newvalues.append(new_v.flatten())
                                entropies.append(ent)
                            except:
                                newlogprobs.append(torch.tensor(0.0, device=device))
                                newvalues.append(b_values[idx])
                                entropies.append(torch.tensor(0.0, device=device))

                    newlogprob = torch.stack(newlogprobs)
                    newvalue = torch.stack(newvalues)
                    entropy = torch.stack(entropies)

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - config['ent_coef'] * entropy_loss + v_loss * config['vf_coef']

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

                    train_losses.append(loss.item())

            train_rewards.append(rewards.mean().item())

        # Evaluate on validation set
        print("\nEvaluating on validation set...")
        val_makespans = []

        for instance in val_instances[:5]:  # Test on 5 instances
            env = MRCMPSPEnvironment()
            state = env.reset(instance['projects'], instance['shared_resources'])

            done = False
            steps = 0

            while not done and steps < 500:
                with torch.no_grad():
                    action = agent.get_action(state, greedy=True)
                    if action is None:
                        break
                    state, reward, done = env.step(action)
                    steps += 1

            if done:
                val_makespans.append(env.executor.get_makespan())

        avg_val_makespan = np.mean(val_makespans) if len(val_makespans) > 0 else float('inf')

        # Save model
        model_path = save_dir / f'model_{config_hash}.pt'
        torch.save(agent.state_dict(), model_path)

        results = {
            'config': config,
            'config_hash': config_hash,
            'val_makespan_mean': float(avg_val_makespan),
            'val_makespan_std': float(np.std(val_makespans)) if len(val_makespans) > 0 else 0.0,
            'train_loss_mean': float(np.mean(train_losses)),
            'train_reward_mean': float(np.mean(train_rewards)),
            'val_success_rate': len(val_makespans) / min(5, len(val_instances)) * 100,
            'model_path': str(model_path),
            'num_parameters': sum(p.numel() for p in agent.parameters())
        }

        print(f"\n✓ Results:")
        print(f"  Val makespan: {avg_val_makespan:.2f}")
        print(f"  Success rate: {results['val_success_rate']:.1f}%")

        return results

    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()

        return {
            'config': config,
            'config_hash': config_hash,
            'val_makespan_mean': float('inf'),
            'error': str(e)
        }


def run_hyperparameter_search(search_type='random', n_configs=10, timesteps_per_config=50000):
    """
    Run hyperparameter search

    Args:
        search_type: 'grid', 'random', or 'quick'
        n_configs: Number of configurations (for random search)
        timesteps_per_config: Training timesteps per configuration
    """
    print("="*70)
    print("Automatic Hyperparameter Tuning for MRCMPSP")
    print("="*70)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Load data
    print("\nLoading MISTA data...")
    loader = MISTADataLoader(data_root='Mista')

    try:
        train_instances = loader.load_set('A')
        val_instances = loader.load_set('B')
        print(f"✓ Train: {len(train_instances)} instances")
        print(f"✓ Val: {len(val_instances)} instances")
    except:
        print("Warning: Could not load Set A/B, using j10")
        train_instances = loader.load_single_project_set(10)[:20]
        val_instances = loader.load_single_project_set(10)[20:30]

    # Generate configurations
    print(f"\nGenerating configurations ({search_type})...")

    if search_type == 'quick':
        search_space = HyperparameterConfig.get_quick_search_space()
        configs = generate_random_search_configs(search_space, n_configs=min(n_configs, 10))
    elif search_type == 'grid':
        search_space = HyperparameterConfig.get_quick_search_space()
        configs = generate_grid_search_configs(search_space)
    elif search_type == 'random':
        search_space = HyperparameterConfig.get_full_search_space()
        configs = generate_random_search_configs(search_space, n_configs=n_configs)
    else:
        raise ValueError(f"Unknown search_type: {search_type}")

    print(f"✓ Generated {len(configs)} configurations")

    # Add default config as baseline
    default_config = HyperparameterConfig.get_default_config()
    configs = [default_config] + configs

    print(f"\n{'='*70}")
    print(f"Will train {len(configs)} configurations")
    print(f"Timesteps per config: {timesteps_per_config:,}")
    print(f"Estimated time: {len(configs) * timesteps_per_config / 50000 * 0.5:.1f} hours")
    print(f"{'='*70}")

    # Train each configuration
    results = []
    save_dir = Path('hp_tuning_results')
    save_dir.mkdir(exist_ok=True)

    for i, config in enumerate(configs):
        print(f"\n{'#'*70}")
        print(f"Configuration {i+1}/{len(configs)}")
        print(f"{'#'*70}")

        result = train_with_config(
            config, train_instances, val_instances,
            timesteps=timesteps_per_config,
            device=device,
            save_dir='hp_tuning_models'
        )

        results.append(result)

        # Save intermediate results
        results_file = save_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Progress: {i+1}/{len(configs)} configs completed")

    # Analyze results
    print(f"\n{'='*70}")
    print("Hyperparameter Search Complete!")
    print(f"{'='*70}")

    # Sort by validation makespan
    valid_results = [r for r in results if r['val_makespan_mean'] != float('inf')]
    valid_results.sort(key=lambda x: x['val_makespan_mean'])

    print(f"\nTop 5 Configurations:")
    print(f"{'='*70}")

    for i, result in enumerate(valid_results[:5]):
        print(f"\nRank {i+1}:")
        print(f"  Val Makespan: {result['val_makespan_mean']:.2f} ± {result['val_makespan_std']:.2f}")
        print(f"  Config:")
        for key, value in result['config'].items():
            print(f"    {key}: {value}")

    # Save final results
    final_results = {
        'search_type': search_type,
        'n_configs': len(configs),
        'timesteps_per_config': timesteps_per_config,
        'all_results': results,
        'best_config': valid_results[0]['config'] if len(valid_results) > 0 else None,
        'best_makespan': valid_results[0]['val_makespan_mean'] if len(valid_results) > 0 else None
    }

    final_file = save_dir / f'hp_search_{search_type}_{int(time.time())}.json'
    with open(final_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n✓ Results saved to: {final_file}")

    # Save best config separately
    if len(valid_results) > 0:
        best_config_file = save_dir / 'best_config.json'
        with open(best_config_file, 'w') as f:
            json.dump(valid_results[0]['config'], f, indent=2)
        print(f"✓ Best config saved to: {best_config_file}")

    return final_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Hyperparameter tuning for MRCMPSP')
    parser.add_argument('--search_type', type=str, default='quick',
                        choices=['quick', 'random', 'grid'],
                        help='Type of search: quick (10 configs), random (N configs), grid (all combos)')
    parser.add_argument('--n_configs', type=int, default=10,
                        help='Number of configurations for random search')
    parser.add_argument('--timesteps', type=int, default=50000,
                        help='Training timesteps per configuration')

    args = parser.parse_args()

    results = run_hyperparameter_search(
        search_type=args.search_type,
        n_configs=args.n_configs,
        timesteps_per_config=args.timesteps
    )

    print("\n✓ Hyperparameter tuning complete!")
