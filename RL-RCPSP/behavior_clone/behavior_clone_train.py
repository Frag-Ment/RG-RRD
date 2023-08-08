import torch
import numpy as np
from behavior_clone.actor_critic_for_clone import Agent
# from behavior_clone.clone_compute_mse import compute_mse
from torch.utils.tensorboard import SummaryWriter
import copy
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default='small_training_set')
parser.add_argument("--learn_eps", type=bool, default=False)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--num_mlp_layers_feature_extract", type=int, default=2)
parser.add_argument("--num_mlp_layers_actor", type=int, default=2)
parser.add_argument("--hidden_dim_actor", type=int, default=32)
parser.add_argument("--num_mlp_layers_critic", type=int, default=2)
parser.add_argument("--hidden_dim_critic", type=int, default=32)
parser.add_argument("--input_dim", type=int, default=7)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--neighbor_pooling_type", type=str, default='sum')

parser.add_argument("--v_coef", type=float, default=0.5)
clone_args = parser.parse_args()


device = 'cuda'
instance_type = 30

run_name = f"{clone_args.exp_name}_{int(time.time())}"
writer = SummaryWriter(f"./behavior_clone/runs2/{run_name}")

# read instance
instance_info = np.load('./PSPLIB_dataset/problems_{}.npy'.format(instance_type), allow_pickle=True)
# not use variant, transform to steady
instance_idx = 0
resource = instance_info[instance_idx][2]
variant = np.stack([resource for i in range(1000)])

# state
adj_dataset = np.load('./behavior_clone/instance{}_dataset/adj_dataset.npy'.format(instance_type), allow_pickle=True)
fea_dataset = np.load('./behavior_clone/instance{}_dataset/fea_dataset.npy'.format(instance_type), allow_pickle=True)
resource_dataset = np.load('./behavior_clone/instance{}_dataset/resource_dataset.npy'.format(instance_type), allow_pickle=True)

# action and reward
runable_dataset = np.load('./behavior_clone/instance{}_dataset/runable_dataset.npy'.format(instance_type), allow_pickle=True)
action_dataset = np.load('./behavior_clone/instance{}_dataset/30optslu.npy'.format(instance_type), allow_pickle=True)
reward_dataset = np.load('./behavior_clone/instance{}_dataset/reward_dataset.npy'.format(instance_type), allow_pickle=True)

# 基本信息
# 但是，如果使用不同大小的instance进行训练，整体的逻辑都要改动
num_instance = adj_dataset.shape[0]
num_nodes = adj_dataset.shape[-1]
num_data = num_instance * num_nodes
# 把state各个成分拍平
adj_dataset = adj_dataset.reshape(num_instance*num_nodes, num_nodes, num_nodes)
fea_dataset = fea_dataset.reshape(num_instance*num_nodes, num_nodes, 7)
resource_dataset = resource_dataset.reshape(num_instance*num_nodes, 4)

# 准备数据, 一共有num_nodes + 1个动作，因为不算第一个虚拟节点，但是算最后一个虚拟节点
# 还是说根本不应该算上最后一个虚拟节点？
# 将action转为one hot编码，然后与输出的概率分布做loss
# 将reward进行反向计算，然后与输出的价值函数做loss
# 然后将两个loss相加做综合loss

# 转换并存储one hot编码
runable_dataset = runable_dataset.reshape(-1)
action_dataset = action_dataset.reshape(-1)
one_hot_dataset = []
for global_idx in range(runable_dataset.shape[0]):
    action_idx = runable_dataset[global_idx].index(action_dataset[global_idx])
    one_hot = np.zeros(len(runable_dataset[global_idx]))
    one_hot[action_idx] = 1
    one_hot_dataset.append(copy.deepcopy(one_hot))
# 由于计算仅需0.009秒，我觉得不需要再另外储存成文件了

# 反向计算reward
# 如果是skip env, 下面需要改一下range中取值方式
if reward_dataset is list:
    raise ValueError('reward_dataset是个列表，不是个数组。如果用skip env需要更改下面逻辑')

return_dataset = np.zeros_like(reward_dataset)
for row_idx in range(reward_dataset.shape[0]):
    return_ = 0
    for step in reversed(range(0, reward_dataset[row_idx].shape[0])):
        return_ = return_ + reward_dataset[row_idx][step]
        return_dataset[row_idx][step] = return_
return_dataset = return_dataset.reshape(-1)

# one_hot_dataset = torch.FloatTensor(one_hot_dataset).to(device)
# 将所有return除以10，便于训练
return_dataset = torch.FloatTensor(return_dataset).to(device) / 10

# 训练
graph_pool = torch.full(size=[1, adj_dataset.shape[-1]], fill_value=1/adj_dataset.shape[-1], dtype=torch.float32, device=device)

agent = Agent(num_layers=clone_args.num_layers,
              learn_eps=clone_args.learn_eps,
              num_mlp_layers_feature_extract=clone_args.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=clone_args.num_mlp_layers_actor,
              hidden_dim_actor=clone_args.hidden_dim_actor,
              num_mlp_layers_critic=clone_args.num_mlp_layers_critic,
              hidden_dim_critic=clone_args.hidden_dim_critic,
              input_dim=clone_args.input_dim, hidden_dim=clone_args.hidden_dim,
              neighbor_pooling_type=clone_args.neighbor_pooling_type,
              device='cuda:0').to(device)


num_epoch = 3000
batch_size = num_instance * 2
num_batch = num_nodes * num_instance / batch_size

cross_ent = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(agent.parameters(), lr=1e-5, eps=1e-5)

for epoch in tqdm(range(num_epoch)):

    all_idx = np.arange(num_data)
    np.random.shuffle(all_idx)

    for start in range(0, num_data, batch_size):
        end = start + batch_size
        b_inds = all_idx[start:end]

        all_loss = torch.zeros(len(b_inds)).to(device)
        all_action_loss = torch.zeros(len(b_inds)).to(device)
        all_value_loss = torch.zeros(len(b_inds)).to(device)

        for i, idx in enumerate(b_inds):
            pi, value = agent.get_pi_and_value(adj_dataset[idx],
                                               fea_dataset[idx],
                                               resource_dataset[idx],
                                               runable_dataset[idx],
                                               graph_pool)

            target_one_hot = torch.FloatTensor(one_hot_dataset[idx]).to(device)
            target_value = return_dataset[idx]

            # 计算loss
            action_loss = cross_ent(pi, target_one_hot)
            value_loss = 0.5 * ((value - target_value) ** 2)

            all_action_loss[i] = action_loss
            all_value_loss[i] = value_loss

        # 反向传播
        a_loss = all_action_loss.mean()
        v_loss = all_value_loss.mean()
        loss = a_loss + clone_args.v_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

    if epoch % 5 == 0:
        torch.save(agent.state_dict(), f'./behavior_clone/saves/{clone_args.exp_name}_model.pt')
        # makespan_mae = compute_mse(0, 70)
    writer.add_scalar("losses/combined_loss", loss.item(), epoch)
    writer.add_scalar("losses/value_loss", v_loss.item(), epoch)
    writer.add_scalar("losses/action_loss", a_loss.item(), epoch)
    # writer.add_scalar("losses/makespan_mae", makespan_mae, epoch)


writer.close()



