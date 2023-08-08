# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import os
import random
import time
from Params import args
from model.actor_critic import Agent
from rcpsp_simulator.skip_res_env import Skip_res_environment
from DataLoader import Dataloader
from dataset_info import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import copy
from test_funcs.compute_greedy_makespan import greedy_average_makespan
from tqdm import tqdm



def make_env(n_env, adj, fea, resource, variant):
    all_env = []
    all_ini_state = []
    for i in range(n_env):
        env = Skip_res_environment()
        ini_state = env.reset(adj, fea, resource, variant)
        all_env.append(env)
        all_ini_state.append(ini_state)
    return all_env, all_ini_state


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer



writer = SummaryWriter(f"runs/{args.use_dataset} {time.time()}+{args.gnn_layer}{args.fea_extract_layer}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# ini dataset
repeat_times = args.repeat_times
if args.use_dataset == 'res_U1_30':
    dataloader_train = Dataloader(train_set_U1_30, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_U1_30)
    dataloader_test = Dataloader(test_set_U1_30)
elif args.use_dataset == 'res_U1_60':
    dataloader_train = Dataloader(train_set_U1_60, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_U1_60)
    dataloader_test = Dataloader(test_set_U1_60)
elif args.use_dataset == 'res_U1_90':
    dataloader_train = Dataloader(train_set_U1_90, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_U1_90)
    dataloader_test = Dataloader(test_set_U1_90)
elif args.use_dataset == 'res_U1_120':
    dataloader_train = Dataloader(train_set_U1_120, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_U1_120)
    dataloader_test = Dataloader(test_set_U1_120)
elif args.use_dataset == 'res_U1_mix':
    dataloader_train = Dataloader(train_set_U1_mix, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_U1_mix)
    dataloader_test = Dataloader(test_set_U1_mix)


if args.use_dataset == 'res_B1_30':
    dataloader_train = Dataloader(train_set_B1_30, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_B1_30)
    dataloader_test = Dataloader(test_set_B1_30)
elif args.use_dataset == 'res_B1_60':
    dataloader_train = Dataloader(train_set_B1_60, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_B1_60)
    dataloader_test = Dataloader(test_set_B1_60)
elif args.use_dataset == 'res_B1_90':
    dataloader_train = Dataloader(train_set_B1_90, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_B1_90)
    dataloader_test = Dataloader(test_set_B1_90)
elif args.use_dataset == 'res_B1_120':
    dataloader_train = Dataloader(train_set_B1_120, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_B1_120)
    dataloader_test = Dataloader(test_set_B1_120)
elif args.use_dataset == 'res_B1_mix':
    dataloader_train = Dataloader(train_set_B1_mix, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_B1_mix)
    dataloader_test = Dataloader(test_set_B1_mix)

if args.use_dataset == 'res_Exp_30':
    dataloader_train = Dataloader(train_set_Exp_30, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_Exp_30)
    dataloader_test = Dataloader(test_set_Exp_30)
elif args.use_dataset == 'res_Exp_60':
    dataloader_train = Dataloader(train_set_Exp_60, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_Exp_60)
    dataloader_test = Dataloader(test_set_Exp_60)
elif args.use_dataset == 'res_Exp_90':
    dataloader_train = Dataloader(train_set_Exp_90, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_Exp_90)
    dataloader_test = Dataloader(test_set_Exp_90)
elif args.use_dataset == 'res_Exp_120':
    dataloader_train = Dataloader(train_set_Exp_120, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_Exp_120)
    dataloader_test = Dataloader(test_set_Exp_120)
elif args.use_dataset == 'res_Exp_mix':
    dataloader_train = Dataloader(train_set_Exp_mix, repeat_times=repeat_times)
    dataloader_val = Dataloader(val_set_Exp_mix)
    dataloader_test = Dataloader(test_set_Exp_mix)

# env setup
adj, fea, res, var = dataloader_train.read_instance(0)
envs, all_ini_states = make_env(args.num_envs, adj, fea, res, var)

agent = Agent(num_layers=args.gnn_layer, learn_eps=False, num_mlp_layers_feature_extract=args.fea_extract_layer,
        num_mlp_layers_actor=2, hidden_dim_actor=32,
        num_mlp_layers_critic=2, hidden_dim_critic=32,
        input_dim=7, hidden_dim=64, neighbor_pooling_type='sum', device='cuda:0').to(device)

# 读取模型
# agent.load_state_dict(torch.load('saves/model.pt'))
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
# data shape info

# use list to storage state
obs = [[0 for j in range(args.num_envs)] for i in range(args.num_steps)]
actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.int64).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)


# TRY NOT TO MODIFY: start the game
# modified obs
global_step = 0
start_time = time.time()

all_next_obs = copy.deepcopy(all_ini_states)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size

print('start training!')
for update in tqdm(range(1, num_updates + 1)):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        # attention storage method of obs!
        obs[step] = copy.deepcopy(all_next_obs)
        dones[step] = copy.deepcopy(next_done)

        # ALGO LOGIC: action logic
        with torch.no_grad():
            # multi env
            for env_idx in range(args.num_envs):

                action, logprob, _, value = agent.get_action_and_value(all_next_obs[env_idx])
                values[step][env_idx] = value.flatten()
                actions[step][env_idx] = action
                logprobs[step][env_idx] = logprob

        # TRY NOT TO MODIFY: execute program and log data.
        all_next_obs = []
        next_done = []

        for env_idx in range(args.num_envs):
            # print('***')
            a = copy.deepcopy(actions[step][env_idx].cpu().numpy())
            next_obs, reward, done = envs[env_idx].step(a)


            reward = - reward / 10
            if done == 1:
                adj, fea, res, var = dataloader_train.next_instance()
                # print('training instance idx:', dataloader_train.now_instance_idx)
            ################################
                next_obs = envs[env_idx].reset(adj, fea, res, var)

            all_next_obs.append(next_obs)
            rewards[step][env_idx] = torch.tensor(reward).to(device).view(-1)
            next_done.append(done)
        next_done = torch.tensor(next_done).to(device)


    # bootstrap value if not done
    # ************************************************
    with torch.no_grad():
        all_next_values = []
        for env_idx in range(args.num_envs):
            next_value = agent.get_value(all_next_obs[env_idx])
            all_next_values.append(next_value)
        all_next_values = torch.tensor(all_next_values).to(device)

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


    # flatten the batch
    # 把所有历史按顺序排列变成一行
    # 需要对state数据做特殊处理
    b_obs = []
    for row_idx in range(len(obs)):
        for col_idx in range(len(obs[0])):
            b_obs.append(obs[row_idx][col_idx])

    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in range(args.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            # mb_inds为索引，
            mb_inds = b_inds[start:end]

            # 从这里开始改
            newlogprob = torch.zeros(len(mb_inds)).to(device)
            newvalue = torch.zeros(len(mb_inds)).to(device)
            entropy = torch.zeros(len(mb_inds)).to(device)
            for idx, item in enumerate(mb_inds):
                _, new_lp, ent, new_v = agent.get_action_and_value(b_obs[item], b_actions[item])
                newlogprob[idx] = new_lp
                newvalue[idx] = new_v
                entropy[idx] = ent

            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
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

        if args.target_kl is not None:
            if approx_kl > args.target_kl:
                break

    # save model and test
    if update % 20 == 0:
        print('saved')
        save_model_name = 'var_saves/'+ args.use_dataset + '_' + str(update) + str(time.time())+ '_'+str(args.gnn_layer)+str(args.fea_extract_layer)+'.pt'
        torch.save(agent, save_model_name)
        # compute average makespan
        train_makespan = greedy_average_makespan(agent, dataloader_train)
        val_makespan = greedy_average_makespan(agent, dataloader_val)

        writer.add_scalar("losses/train_average_makespan", train_makespan, update)
        writer.add_scalar("losses/val_average_makespan", val_makespan, update)


    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

writer.close()
