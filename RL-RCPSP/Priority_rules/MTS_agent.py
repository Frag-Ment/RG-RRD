import numpy as np
from rcpsp_simulator.jobdag import Jobdag

# 子节点最多的节点
# Most total successor
def MTS_agent(jobdag, state):
    runable_nodes_idx = state[3]
    # runable_nodes_idx = state
    # 找出每个节点的子节点数目
    n_child_node_list = []
    for node_idx in runable_nodes_idx:
        n_child_node_list.append(len(jobdag.nodes[node_idx].child_nodes))
    # 当列表中有多个相同最大值时，list.index方法默认取第一个值的index
    return runable_nodes_idx[n_child_node_list.index(max(n_child_node_list))]


###for test###
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
# action = MTS_agent(jd, [0, 1, 2])
# breakpoint()
# print('***')
#

