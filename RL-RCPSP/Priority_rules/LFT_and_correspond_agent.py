import numpy as np
import time
import copy
from Priority_rules.LFT_functions import get_init_frontier, get_bottom_up_paths, generate_CPM
from rcpsp_simulator.jobdag import Jobdag

# 之前误以为这个是CPM，但实际上这个是LFT
# LFT, latest finishing time
def LFT_agent(jobdag, state):
    runable_nodes_idx = state[3]
    depth = 256
    msg_mat, msg_mask = get_bottom_up_paths(jobdag, depth)
    CPM_list = generate_CPM(jobdag, msg_mat, msg_mask)
    priority_list = CPM_list[runable_nodes_idx]
    return runable_nodes_idx[np.argmax(priority_list)]

# LST,latest starting time  = LFT - dj
def LST_agent(jobdag, state):
    runable_nodes_idx = state[3]
    # runable_nodes_idx = state
    depth = 256
    msg_mat, msg_mask = get_bottom_up_paths(jobdag, depth)
    CPM_list = generate_CPM(jobdag, msg_mat, msg_mask)
    priority_list = CPM_list[runable_nodes_idx]
    # print(priority_list)
    self_duration_list = []
    for node_idx in runable_nodes_idx:
        self_duration_list.append(jobdag.nodes[node_idx].task_duration)
    # print(self_duration_list)
    return runable_nodes_idx[np.argmax(np.array(priority_list) - np.array(self_duration_list))]


# MSLK: Minimum slack  = LFT - EFT
# 这个需要传入当前env对未来进行模拟才能计算
# def MSKL_agent(jobdag, state, env):
#     runable_nodes_idx = state[3]
#     depth = 256
#     msg_mat, msg_mask = get_bottom_up_paths(jobdag, depth)
#     CPM_list = generate_CPM(jobdag, msg_mat, msg_mask)
#     # LFT:
#     priority_list = CPM_list[runable_nodes_idx]
#     # EFT: 最早优先可完成时间
#     EFT_list = []
#     for node_idx in runable_nodes_idx:
#         simulate_env = copy.deepcopy(env)
#         simulate_env.step(node_idx)
#         EFT_list.append(simulate_env.walltime)
#
#     return runable_nodes_idx[np.argmax(priority_list - np.array(EFT_list))]





### for test###
# all_info = np.load('../PSPLIB_dataset/problems_30.npy', allow_pickle=True)
# # shape: (480, 1000, 4)
# all_resource_variant = np.load('../PSPLIB_dataset/variant/variant_30_B1.npy', allow_pickle=True)
#
# instance_idx = 1
# adj_mat = all_info[instance_idx][0]
# fea_mat = all_info[instance_idx][1]
# resource_capacity = all_info[instance_idx][2]
# jd = Jobdag(adj_mat, fea_mat, resource_capacity)
#
# m = LST_agent(jd, [0, 1, 2, 3])
# print(m)

# a = np.array([1, 2, 3])
# b = np.array([0, 1, 0])
# c = a-b
# print(list(c))