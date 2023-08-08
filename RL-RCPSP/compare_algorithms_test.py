from Priority_rules.LFT_and_correspond_agent import LFT_agent, LST_agent
from Priority_rules.SPT_agent import SPT_agent
from Priority_rules.GRPW_agent import GRPW_agent
from Priority_rules.MTS_agent import MTS_agent

from DataLoader import Dataloader
from dataset_info import *
from rcpsp_simulator.normal_env import Normal_environment
from rcpsp_simulator.normal_res_env import Normal_res_environment
import time
import copy


# test all the compare algorithms

# rcpsp dataset
test_instances = [test_set_30, test_set_60, test_set_90, test_set_Exp_120, test_set_mix]

# srcpsp dataset
var_test_instances = [test_set_U1_30, test_set_U1_60, test_set_U1_90, test_set_U1_120, test_set_U1_mix,
                      test_set_B1_30, test_set_B1_60, test_set_B1_90, test_set_B1_120, test_set_B1_mix,
                      test_set_Exp_30, test_set_Exp_60, test_set_Exp_90, test_set_Exp_120, test_set_Exp_mix]



agent_list = [SPT_agent, LFT_agent, LST_agent, GRPW_agent, MTS_agent]
if_variant = False

if if_variant != True:
    all_instances = test_instances
else:
    all_instances = var_test_instances

for test_instance in all_instances:
    print('test_instances', test_instance)
    if if_variant != True:
        use_env = Normal_environment()
    else:
        use_env = Normal_res_environment()

    for agent in agent_list:
        dataloader = Dataloader(test_instance)
        n_instances = dataloader.num_instances
        env = copy.deepcopy(use_env)
        makespan_list = []
        start_time = time.time()

        for i in range(n_instances):
            adj, fea, res, var = dataloader.next_instance()
            state = env.reset(adj, fea, res, var)
            while 1:
                jobdag = env.executor.jobdag
                action = agent(jobdag, state)

                state, _, done = env.step(action)
                if done == 1:
                    makespan_list.append(env.executor.walltime)
                    break

        print(f'{agent.__name__} average makespan:', sum(makespan_list) / len(makespan_list))
        print('average time:', (time.time()-start_time)/n_instances)




    # # env = Normal_environment()
    # env = Normal_res_environment()
    # SPT_makespan_list = []
    # start_time = time.time()
    # for i in range(n_instances):
    #     adj, fea, res, var = dataloader.next_instance()
    #     state = env.reset(adj, fea, res, var)
    #     while 1:
    #         jobdag = env.executor.jobdag
    #         action = SPT_agent(jobdag, state)
    #
    #         state, _, done = env.step(action)
    #         if done == 1:
    #             SPT_makespan_list.append(env.executor.walltime)
    #             break
    #
    # print('SPT average makespan:', sum(SPT_makespan_list) / len(SPT_makespan_list))
    # print('average time:', (time.time()-start_time)/n_instances)
    #
    #
    # # env = Normal_environment()
    # env = Normal_res_environment()
    # SPT_makespan_list = []
    # start_time = time.time()
    # for i in range(n_instances):
    #     adj, fea, res, var = dataloader.next_instance()
    #     state = env.reset(adj, fea, res, var)
    #     while 1:
    #         jobdag = env.executor.jobdag
    #         action = SPT_agent(jobdag, state)
    #
    #         state, _, done = env.step(action)
    #         if done == 1:
    #             SPT_makespan_list.append(env.executor.walltime)
    #             break
    #
    # print('SPT average makespan:', sum(SPT_makespan_list) / len(SPT_makespan_list))
    # print('average time:', (time.time()-start_time)/n_instances)
    #
    #
