import numpy as np
import random
import time
from Params import args
from model.actor_critic import Agent
from rcpsp_simulator.normal_env import Normal_environment
from rcpsp_simulator.skip_env import Skip_environment
from rcpsp_simulator.normal_res_env import Normal_res_environment
from rcpsp_simulator.skip_res_env import Skip_res_environment
import numpy as np
import torch
import copy


device = torch.device('cuda:0')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

def sample_average_makespan(save_model_name, dataloader, n_sampels=30, normal_env=False, variant=False, greedy_test=False):
    dataloader_for_test = copy.deepcopy(dataloader)
    dataloader_for_test.repeat_times = 1

    if normal_env == True:
        if variant == False:
            env = Normal_environment()
        else:
            env = Normal_res_environment()

    else:
        if variant == False:
            env = Skip_environment()
        else:
            env = Skip_res_environment()

    # agent = Agent(num_layers=3, learn_eps=False, num_mlp_layers_feature_extract=2,
    #               num_mlp_layers_actor=2, hidden_dim_actor=32,
    #               num_mlp_layers_critic=2, hidden_dim_critic=32,
    #               input_dim=7, hidden_dim=64, neighbor_pooling_type='sum', device='cuda:0').to(device)
    #
    # # 读取模型
    # agent.load_state_dict(torch.load(save_model_name))

    agent = torch.load(save_model_name)

    all_data_store = []
    all_time_list = []
    for instance_idx in range(dataloader.num_instances):

        # print('sampling instance', instance_idx)
        sample_result = []
        start_time = time.time()
        for n_sample in range(n_sampels):
            adj, fea, res, var = dataloader.read_instance(instance_idx)
            state = env.reset(adj, fea, res, var)

            for step in range(100000):
                action, _, _, _ = agent.get_action_and_value(state, greedy_test=greedy_test)
                # print(action)
                # print(action.cpu().numpy())
                # print(int(action.cpu().numpy()))
                state, reward, done = env.step(action.cpu().numpy())
                if done == 1:
                    sample_result.append(int(env.executor.walltime))
                    all_time_list.append(time.time() - start_time)
                    break

        # print(sample_result)
        # print('makespan:', min(sample_result))
        all_data_store.append(min(sample_result))

    average_makespan = sum(all_data_store) / len(all_data_store)
    return average_makespan, sum(all_time_list)/(dataloader.num_instances * n_sampels)


