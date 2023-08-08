import numpy as np
from rcpsp_simulator.jobdag import Jobdag

# 该节点duration加上其所有子节点duration最长
# Greatest rank positional weight
def GRPW_agent(jobdag, state):
    runable_nodes_idx = state[3]
    # runable_nodes_idx = state
    GRPW_list = []
    for node_idx in runable_nodes_idx:
        self_duration = jobdag.nodes[node_idx].task_duration
        sum_child_duration = 0
        for child_idx in jobdag.nodes[node_idx].child_nodes:
            sum_child_duration += jobdag.nodes[child_idx].task_duration
        GRPW_list.append(self_duration+sum_child_duration)
    # print(GRPW_list)
    return runable_nodes_idx[GRPW_list.index(max(GRPW_list))]

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
# action = GRPW_agent(jd, [0, 1, 2])
# breakpoint()
# print('***')
