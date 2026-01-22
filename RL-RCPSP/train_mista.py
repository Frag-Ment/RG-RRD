"""
PPO Training for MRCMPSP using MISTA Benchmark Data
Real-world benchmark instances from MISTA database
"""
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import copy
import json

from model.hierarchical_agent import HierarchicalAgent
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from MISTADataLoader import MISTADataLoader

# Hyperparameters
class MRCMPSPArgs:
    # MISTA Data Configuration
    use_mista = True
    mista_data_root = 'Mista'
    train_set = 'A'  # Set A for training (10 instances)
    val_set = 'B'    # Set B for validation (10 instances)
    use_j10_augmentation = False  # Add j10 instances for data augmentation
    j10_instances = 20  # Number of j10 instances to add if augmentation enabled

    # Training
    num_envs = 4  # Parallel environments
    num_steps = 128  # Steps per rollout
    total_timesteps = 500000  # Increased for real benchmarks
    learning_rate = 3e-5
    anneal_lr = True

    # PPO
    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.1
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    update_epochs = 4
    minibatch_size = 64
    norm_adv = True
    clip_vloss = True
    target_kl = None

    # Model architecture
    gnn_layers = 3
    gnn_hidden_dim = 64
    mlp_hidden_dim = 32
    project_hidden_dim = 64
    node_feature_dim = 9
    project_feature_dim = 8

    # System
    cuda = True
    seed = 42
    torch_deterministic = True

    # Logging
    experiment_name = f"mrcmpsp_mista_{train_set}"
    save_interval = 25  # Save model every N updates

args = MRCMPSPArgs()


def make_env(instance):
    """Create environment with given instance"""
    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])
    return env, state


def compute_greedy_makespan(agent, instances, num_test=None):
    """Evaluate agent with greedy policy"""
    if num_test is None:
        num_test = len(instances)

    total_makespan = 0
    successful_runs = 0

    for i in range(min(num_test, len(instances))):
        instance = instances[i]
        env = MRCMPSPEnvironment()
        state = env.reset(instance['projects'], instance['shared_resources'])

        done = False
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            with torch.no_grad():
                action = agent.get_action(state, greedy=True)

                if action is None:
                    break

                state, reward, done = env.step(action)
                steps += 1

        if done:
            total_makespan += env.executor.get_makespan()
            successful_runs += 1

    if successful_runs == 0:
        return float('inf')

    return total_makespan / successful_runs


# Setup
print("="*60)
print(f"Training MRCMPSP with MISTA Benchmarks")
print(f"Train Set: {args.train_set}, Val Set: {args.val_set}")
print("="*60)

writer = SummaryWriter(f"runs/{args.experiment_name}_{time.time()}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|"
                                               for key, value in vars(args).items()])),
)

# Seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
print(f"Using device: {device}")

# Load MISTA data
print("\n" + "="*60)
print("Loading MISTA Benchmark Data")
print("="*60)

loader = MISTADataLoader(data_root=args.mista_data_root)

try:
    train_instances = loader.load_set(args.train_set)
    print(f"✓ Loaded {len(train_instances)} training instances from Set {args.train_set}")
except Exception as e:
    print(f"✗ Error loading train set {args.train_set}: {e}")
    print(f"  Falling back to j10 instances...")
    train_instances = loader.load_single_project_set(10)[:50]
    print(f"✓ Loaded {len(train_instances)} j10 instances for training")

try:
    val_instances = loader.load_set(args.val_set)
    print(f"✓ Loaded {len(val_instances)} validation instances from Set {args.val_set}")
except Exception as e:
    print(f"✗ Error loading val set {args.val_set}: {e}")
    val_instances = []

# Optional: Augment with j10 instances
if args.use_j10_augmentation:
    print(f"\nAugmenting with {args.j10_instances} j10 instances...")
    j10_instances = loader.load_single_project_set(10)[:args.j10_instances]
    train_instances = train_instances + j10_instances
    print(f"✓ Total training instances: {len(train_instances)}")

# Get statistics
stats = loader.get_statistics(train_instances)
print(f"\nTraining Set Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Initialize environments with first instances
print("\n" + "="*60)
print("Initializing Environments")
print("="*60)

envs = []
all_ini_states = []

for i in range(args.num_envs):
    instance = train_instances[i % len(train_instances)]
    env, ini_state = make_env(instance)
    envs.append(env)
    all_ini_states.append(ini_state)
    print(f"  Env {i}: {len(instance['projects'])} projects, "
          f"{sum(p['num_jobs']-2 for p in instance['projects'])} total tasks")

# Create agent
print("\n" + "="*60)
print("Creating Hierarchical Agent")
print("="*60)

agent = HierarchicalAgent(
    project_feature_dim=args.project_feature_dim,
    project_hidden_dim=args.project_hidden_dim,
    num_gnn_layers=args.gnn_layers,
    gnn_hidden_dim=args.gnn_hidden_dim,
    mlp_hidden_dim=args.mlp_hidden_dim,
    node_feature_dim=args.node_feature_dim,
    num_mlp_layers=2,
    critic_hidden_dim=64,
    device=device
).to(device)

total_params = sum(p.numel() for p in agent.parameters())
print(f"  Total parameters: {total_params:,}")

optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# Storage
obs = [[None for j in range(args.num_envs)] for i in range(args.num_steps)]
actions = [[None for j in range(args.num_envs)] for i in range(args.num_steps)]
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# Training loop
global_step = 0
start_time = time.time()
all_next_obs = copy.deepcopy(all_ini_states)
next_done = torch.zeros(args.num_envs).to(device)

batch_size = args.num_steps * args.num_envs
num_updates = args.total_timesteps // batch_size

os.makedirs('saves', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Track training metrics
training_history = {
    'updates': [],
    'train_makespan': [],
    'val_makespan': [],
    'policy_loss': [],
    'value_loss': [],
    'entropy': []
}

print("\n" + "="*60)
print(f"Starting Training for {num_updates} Updates")
print(f"Batch size: {batch_size}, Minibatch size: {args.minibatch_size}")
print("="*60 + "\n")

for update in tqdm(range(1, num_updates + 1), desc="Training"):
    # Learning rate annealing
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    # Collect rollout
    for step in range(args.num_steps):
        global_step += args.num_envs
        obs[step] = copy.deepcopy(all_next_obs)
        dones[step] = copy.deepcopy(next_done)

        # Get actions from agent
        with torch.no_grad():
            for env_idx in range(args.num_envs):
                try:
                    action, logprob, _, value = agent.get_action_and_value(all_next_obs[env_idx])

                    values[step][env_idx] = value.flatten()
                    actions[step][env_idx] = action
                    logprobs[step][env_idx] = logprob
                except Exception as e:
                    # If agent fails, use None action
                    actions[step][env_idx] = None
                    logprobs[step][env_idx] = 0
                    values[step][env_idx] = 0

        # Execute actions
        all_next_obs = []
        next_done = []

        for env_idx in range(args.num_envs):
            action = actions[step][env_idx]

            # Handle no feasible action case
            if action is None:
                # Episode ended prematurely
                reward = -10  # Small penalty
                done = 1
                # Load new instance
                instance = train_instances[random.randint(0, len(train_instances)-1)]
                next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])
            else:
                try:
                    next_obs, reward, done = envs[env_idx].step(action)

                    # Normalize reward
                    reward = reward / 10.0

                    if done:
                        # Load next instance
                        instance = train_instances[random.randint(0, len(train_instances)-1)]
                        next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])

                except Exception as e:
                    # Handle errors gracefully
                    reward = -10
                    done = 1
                    instance = train_instances[random.randint(0, len(train_instances)-1)]
                    next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])

            all_next_obs.append(next_obs)
            rewards[step][env_idx] = torch.tensor(reward).to(device).view(-1)
            next_done.append(done)

        next_done = torch.tensor(next_done).to(device)

    # Compute advantages (GAE)
    with torch.no_grad():
        all_next_values = []
        for env_idx in range(args.num_envs):
            try:
                next_value = agent.get_value(all_next_obs[env_idx])
                all_next_values.append(next_value)
            except:
                all_next_values.append(torch.tensor(0.0, device=device))

        all_next_values = torch.stack(all_next_values).to(device).flatten()

        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0

        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = all_next_values
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]

            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

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

    # Optimize policy and value network
    b_inds = np.arange(batch_size)
    clipfracs = []

    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)

        for start in range(0, batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            # Recompute log probs and values
            newlogprobs = []
            newvalues = []
            entropies = []

            for idx in mb_inds:
                if b_actions[idx] is None:
                    # Skip invalid actions
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

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]

            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

        if args.target_kl is not None and approx_kl > args.target_kl:
            break

    # Logging and evaluation
    if update % args.save_interval == 0:
        print(f"\n[Update {update}/{num_updates}] Evaluating...")

        # Save model
        save_path = f'saves/{args.experiment_name}_update{update}.pt'
        torch.save(agent.state_dict(), save_path)
        print(f"  ✓ Model saved: {save_path}")

        # Evaluate
        train_makespan = compute_greedy_makespan(agent, train_instances, num_test=5)
        print(f"  Train makespan (5 instances): {train_makespan:.2f}")

        if len(val_instances) > 0:
            val_makespan = compute_greedy_makespan(agent, val_instances, num_test=len(val_instances))
            print(f"  Val makespan ({len(val_instances)} instances): {val_makespan:.2f}")
        else:
            val_makespan = train_makespan

        # Log to tensorboard
        writer.add_scalar("eval/train_makespan", train_makespan, update)
        writer.add_scalar("eval/val_makespan", val_makespan, update)

        # Save to history
        training_history['updates'].append(update)
        training_history['train_makespan'].append(train_makespan)
        training_history['val_makespan'].append(val_makespan)
        training_history['policy_loss'].append(pg_loss.item())
        training_history['value_loss'].append(v_loss.item())
        training_history['entropy'].append(entropy_loss.item())

    # Log metrics every update
    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

print("\n" + "="*60)
print("✓ Training Complete!")
print("="*60)

# Final evaluation
if len(val_instances) > 0:
    print("\nFinal evaluation on validation set...")
    final_val_makespan = compute_greedy_makespan(agent, val_instances, num_test=len(val_instances))
    print(f"  Val makespan: {final_val_makespan:.2f}")
    writer.add_scalar("eval/final_val_makespan", final_val_makespan, num_updates)

# Save final model
final_save_path = f'saves/{args.experiment_name}_final.pt'
torch.save(agent.state_dict(), final_save_path)
print(f"\n✓ Final model saved: {final_save_path}")

# Save training history
history_path = f'results/{args.experiment_name}_history.json'
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"✓ Training history saved: {history_path}")

writer.close()

print("\n" + "="*60)
print("Training completed successfully!")
print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
print("="*60)
