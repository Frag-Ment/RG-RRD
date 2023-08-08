import random
import time
# from behavior_clone_train import clone_args
from behavior_clone.actor_critic_for_clone import Agent
from rcpsp_simulator.normal_env import Normal_environment
import numpy as np
import torch
from torch.distributions.categorical import Categorical


device = torch.device('cuda:0')

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
# torch.backends.cudnn.deterministic = clone_args.torch_deterministic

def test_train(test_start_idx, test_end_idx):

    # env setup
    all_info = np.load('../PSPLIB_dataset/problems_30.npy', allow_pickle=True)

    instance_idx = 0
    resource = all_info[instance_idx][2]
    variant = np.stack([resource for ss in range(1000)])

    n_nodes = all_info[instance_idx][1].shape[0] - 2
    graph_pool = torch.full(size=[1, n_nodes], fill_value=1/n_nodes, dtype=torch.float32, device='cuda:0')

    env = Normal_environment()

    agent = Agent(num_layers=3, learn_eps=False, num_mlp_layers_feature_extract=2,
            num_mlp_layers_actor=2, hidden_dim_actor=32,
            num_mlp_layers_critic=2, hidden_dim_critic=32,
            input_dim=7, hidden_dim=64, neighbor_pooling_type='sum', device='cuda:0').to(device)

    # 读取模型
    agent.load_state_dict(torch.load('../behavior_clone/saves/small_training_set_model.pt'))

    all_data_store = []
    idx = -1 + test_start_idx
    for instance_idx in range(test_start_idx, test_end_idx):
        idx += 1
        resource = all_info[instance_idx][2]
        variant = np.stack([resource for item in range(1000)])

        state = env.reset(all_info[instance_idx][0], all_info[instance_idx][1], all_info[instance_idx][2], variant)
        for step in range(100000):
            # 由于结构变化，需要重新准备网络输入，并处理输出
            adj_mat = state[0]
            fea_mat = state[1]
            resource_mat = state[2]
            runable_nodes_idx = state[3]
            #
            pi, value = agent.get_pi_and_value(adj_mat,
                                               fea_mat,
                                               resource_mat,
                                               runable_nodes_idx,
                                               graph_pool)

            _, sample_idx = pi.squeeze().max(0)
            action = runable_nodes_idx[sample_idx]

            # print(action)
            # print(action.cpu().numpy())
            # print(int(action.cpu().numpy()))
            state, reward, done = env.step(action)
            if done == 1:
                break

        # print(i, env.executor.walltime)
        all_data_store.append([int(idx), int(env.executor.walltime)])

    all_data_store = np.stack(all_data_store)
    np.savetxt('../behavior_clone/clone_walltime.txt', all_data_store)


test_train(0, 70)