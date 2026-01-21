"""
PPO Training for MRCMPSP (Multi-Resource-Constrained Multi-Project Scheduling Problem)
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

from model.hierarchical_agent import HierarchicalAgent
from mrcmpsp_simulator.mrcmpsp_env import MRCMPSPEnvironment
from MRCMPSPDataLoader import create_mrcmpsp_dataloaders

# Hyperparameters
class MRCMPSPArgs:
    # Environment
    num_projects = 3
    tasks_per_project = 20
    num_modes = 3

    # Training
    num_envs = 4  # Parallel environments
    num_steps = 128  # Steps per rollout
    total_timesteps = 300000
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
    experiment_name = f"mrcmpsp_p{num_projects}_t{tasks_per_project}_m{num_modes}"
    save_interval = 50  # Save model every N updates


args = MRCMPSPArgs()


def make_env(instance):
    """Create environment with given instance"""
    env = MRCMPSPEnvironment()
    state = env.reset(instance['projects'], instance['shared_resources'])
    return env, state


def compute_greedy_makespan(agent, dataloader, num_test=20):
    """Evaluate agent with greedy policy"""
    total_makespan = 0

    for i in range(num_test):
        instance = dataloader.read_instance(i)
        env = MRCMPSPEnvironment()
        state = env.reset(instance['projects'], instance['shared_resources'])

        done = False
        while not done:
            with torch.no_grad():
                action = agent.get_action(state, greedy=True)

                if action is None:
                    break

                state, reward, done = env.step(action)

        total_makespan += env.executor.get_makespan()

    return total_makespan / num_test


# Setup
print(f"Training MRCMPSP with {args.num_projects} projects, "
      f"{args.tasks_per_project} tasks/project, {args.num_modes} modes")

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

# Create dataloaders
print("Generating datasets...")
dataloader_train, dataloader_val, dataloader_test = create_mrcmpsp_dataloaders(
    num_projects=args.num_projects,
    tasks_per_project=args.tasks_per_project,
    num_modes=args.num_modes
)

print(f"Train: {len(dataloader_train)} instances")
print(f"Val: {len(dataloader_val)} instances")
print(f"Test: {len(dataloader_test)} instances")

# Initialize environments
print("Initializing environments...")
first_instance = dataloader_train.read_instance(0)
envs = []
all_ini_states = []

for i in range(args.num_envs):
    env, ini_state = make_env(first_instance)
    envs.append(env)
    all_ini_states.append(ini_state)

# Create agent
print("Creating hierarchical agent...")
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

print(f"\nStarting training for {num_updates} updates...")
print(f"Batch size: {batch_size}, Minibatch size: {args.minibatch_size}")

for update in tqdm(range(1, num_updates + 1)):
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
                action, logprob, _, value = agent.get_action_and_value(all_next_obs[env_idx])

                values[step][env_idx] = value.flatten()
                actions[step][env_idx] = action
                logprobs[step][env_idx] = logprob

        # Execute actions
        all_next_obs = []
        next_done = []

        for env_idx in range(args.num_envs):
            action = actions[step][env_idx]

            # Handle no feasible action case
            if action is None:
                # Episode ended prematurely
                reward = 0
                done = 1
                instance = dataloader_train.next_instance()
                next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])
            else:
                try:
                    next_obs, reward, done = envs[env_idx].step(action)

                    # Normalize reward
                    reward = reward / 10.0

                    if done:
                        # Load next instance
                        instance = dataloader_train.next_instance()
                        next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])

                except Exception as e:
                    print(f"Error in env {env_idx}: {e}")
                    reward = -100
                    done = 1
                    instance = dataloader_train.next_instance()
                    next_obs = envs[env_idx].reset(instance['projects'], instance['shared_resources'])

            all_next_obs.append(next_obs)
            rewards[step][env_idx] = torch.tensor(reward).to(device).view(-1)
            next_done.append(done)

        next_done = torch.tensor(next_done).to(device)

    # Compute advantages (GAE)
    with torch.no_grad():
        all_next_values = []
        for env_idx in range(args.num_envs):
            next_value = agent.get_value(all_next_obs[env_idx])
            all_next_values.append(next_value)
        all_next_values = torch.tensor(all_next_values).to(device).flatten()

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
                    _, new_lp, ent, new_v = agent.get_action_and_value(
                        b_obs[idx], b_actions[idx]
                    )
                    newlogprobs.append(new_lp)
                    newvalues.append(new_v.flatten())
                    entropies.append(ent)

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
        print(f"\n[Update {update}] Evaluating...")

        # Save model
        save_path = f'saves/{args.experiment_name}_update{update}.pt'
        torch.save(agent.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Evaluate
        train_makespan = compute_greedy_makespan(agent, dataloader_train, num_test=10)
        val_makespan = compute_greedy_makespan(agent, dataloader_val, num_test=10)

        print(f"Train makespan: {train_makespan:.2f}")
        print(f"Val makespan: {val_makespan:.2f}")

        writer.add_scalar("eval/train_makespan", train_makespan, update)
        writer.add_scalar("eval/val_makespan", val_makespan, update)

    # Log metrics
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

print("\n✓ Training complete!")

# Final evaluation
print("\nFinal evaluation on test set...")
test_makespan = compute_greedy_makespan(agent, dataloader_test, num_test=30)
print(f"Test makespan: {test_makespan:.2f}")
writer.add_scalar("eval/test_makespan", test_makespan, num_updates)

# Save final model
final_save_path = f'saves/{args.experiment_name}_final.pt'
torch.save(agent.state_dict(), final_save_path)
print(f"Final model saved to {final_save_path}")

writer.close()
