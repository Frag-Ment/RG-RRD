# import numpy as np
from rcpsp_simulator.node import *
import copy
# 先考虑如何把信息放进程序
# 然后考虑程序如何具体执行

# 以后均使用邻接矩阵表示nodes之间的关系
# 在初始化任务时， 需要给定每个节点的自身属性和邻接矩阵。节点的idx需和矩阵对应
class Jobdag(object):
    def __init__(self, adj_mat, nodes_information, resource_exec):
        # 自身信息
        # self.idx:自身索引
        self.idx = None
        self.nodes = []

        self.original_adj_mat = copy.deepcopy(adj_mat)
        self.original_feature_mat = copy.deepcopy(nodes_information)

        temp = np.delete(self.original_adj_mat, [0, -1], axis=0)
        temp = np.delete(temp, [0, -1], axis=1)
        self.adj_mat = copy.deepcopy(temp)
        self.feature_mat = copy.deepcopy(np.delete(self.original_feature_mat, [0, -1], axis=0))


        self.reversed_adj_mat = copy.deepcopy(self.adj_mat.T)
        # adj_mat有多少列，输入父代和子代节点时需要用到
        self.adj_mat_line_shape = self.adj_mat.shape[0]

        self.resource_exec = copy.deepcopy(resource_exec)

        # 状态信息
        self.nodes_all_done = False
        self.num_nodes = None
        self.done_nodes = []

        # 将信息装入每个节点
        for idx, info in zip(range(self.feature_mat.shape[0]), self.feature_mat):
            task_duration, resource1, resource2, resource3, resource4, _, _ = info
            new_node = Node(idx, task_duration, resource1, resource2, resource3, resource4)
            self.nodes.append(new_node)

        # 设定每个节点的父节点和子节点信息
        # 使用列表记录父节点或子节点的idx
        for node_idx, node in enumerate(self.nodes):
            node.child_nodes = [idx for idx, var in zip(range(self.adj_mat_line_shape), self.adj_mat[node.idx]) if var == 1 and idx != node_idx]
            node.parent_nodes = [idx for idx, var in zip(range(self.adj_mat_line_shape), self.reversed_adj_mat[node.idx]) if var == 1 and idx != node_idx]

        # 统计一下节点总数
        self.num_nodes = len(self.nodes)










