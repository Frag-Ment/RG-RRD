import random
import time
from Params import args
from model.actor_critic import Agent
from rcpsp_simulator.normal_env import Normal_environment
from rcpsp_simulator.skip_env import Skip_environment
import numpy as np
import torch
import copy

device = torch.device('cuda:0')

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

def test_train(save_model_name, dataloader):
    dataloader_for_test = copy.deepcopy(dataloader)
    dataloader_for_test.repeat_times = 1

    env = Skip_environment()
    agent = Agent(num_layers=4, learn_eps=False, num_mlp_layers_feature_extract=6,
            num_mlp_layers_actor=2, hidden_dim_actor=32,
            num_mlp_layers_critic=2, hidden_dim_critic=32,
            input_dim=7, hidden_dim=64, neighbor_pooling_type='sum', device='cuda:0').to(device)

    # 读取模型
    agent.load_state_dict(torch.load(save_model_name))
    all_data_store = []

    for instance_idx in range(dataloader.num_instances):
        adj, fea, res, var = dataloader.next_instance()

        state = env.reset(adj, fea, res, var)
        for step in range(100000):
            action, _, _, _ = agent.get_action_and_value(state, greedy_test=True)
            # print(action)
            # print(action.cpu().numpy())
            # print(int(action.cpu().numpy()))
            state, reward, done = env.step(action.cpu().numpy())
            if done == 1:
                break

        # print(i, env.executor.walltime)
        all_data_store.append([int(instance_idx), int(env.executor.walltime)])

    all_data_store = np.stack(all_data_store)
    np.savetxt('./30NB.txt', all_data_store)


