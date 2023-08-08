import numpy as np
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

def greedy_average_makespan(agent, dataloader, normal_env=False):
    dataloader_for_test = copy.deepcopy(dataloader)
    dataloader_for_test.repeat_times = 1
    if normal_env == True:
        env = Normal_environment()
    else:
        env = Skip_environment()
    agent = agent

    all_data_store = []
    start_time = time.time()
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

        all_data_store.append(int(env.executor.walltime))
    end_time = time.time()
    average_makespan = sum(all_data_store) / len(all_data_store)
    return average_makespan



