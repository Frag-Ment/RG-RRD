import numpy as np
from rcpsp_simulator.jobdag import Jobdag

# 所有对比算法通过读取jobdag来计算哪个节点优先
# 不如统一输入为：jobdag, state，需要手动env.executor.jobdag取出jobdag
# shortest processing time
def SPT_agent(jobdag, state):
    runable_nodes_idx = state[3]
    each_node_time = [jobdag.nodes[item].task_duration for item in runable_nodes_idx]
    return runable_nodes_idx[each_node_time.index(min(each_node_time))]

def LPT_agent(jobdag, state):
    runable_nodes_idx = state[3]
    each_node_time = [jobdag.nodes[item].task_duration for item in runable_nodes_idx]
    return runable_nodes_idx[each_node_time.index(max(each_node_time))]

# def SPT_agent(jobdag, runable_nodes_idx):
#     # runable_nodes_idx = state[3]
#     each_node_time = [jobdag.nodes[item].task_duration for item in runable_nodes_idx]
#     print(each_node_time)
#     return runable_nodes_idx[each_node_time.index(min(each_node_time))]

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
# action = SPT_agent(jd, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# breakpoint()
# print('***')