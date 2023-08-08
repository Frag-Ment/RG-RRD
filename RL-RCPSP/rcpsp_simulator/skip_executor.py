from rcpsp_simulator.task import Task
import copy
from collections import namedtuple
import numpy as np


class Skip_executor(object):
    def __init__(self, jobdag, resource_variant):

        self.jobdag = jobdag

        # 原始资源矩阵，不会变
        self.original_resource_exec = copy.deepcopy(jobdag.resource_exec)
        # 资源矩阵, 和jobdag的资源矩阵是同一地址，所以不用deepcopy
        self.resource_exec = jobdag.resource_exec
        # 资源变动矩阵，储存在executor中而不是jobdag中
        self.resource_variant = copy.deepcopy(resource_variant)
        # 时间信息
        self.walltime = 0

        # 可运行节点初始化
        self.runable_nodes_idx = self.ini_runable_nodes_idx()

        # 正在执行的节点的列表
        self.running_tasks = []
        # 已经执行完成的任务的列表
        self.done_tasks = []

        # 节点的特征矩阵，同样是“一维数组”
        self.feature_mat = jobdag.feature_mat
        # 已经完成的节点的数量，用于判断所有的节点是否全部完成，由于第1个和最后1个节点不存在，因此初始值为2
        self.complete_node = 0
        # 所有活动的加工顺序
        self.action_sequence = []

        # 初始化mask
        mask = np.zeros(self.jobdag.num_nodes)
        mask[self.runable_nodes_idx] = 1
        self.now_action_mask = mask


    # 检查所有可以执行的节点，返回可执行节点的列表
    # 检查所有叶子节点
    # 叶子节点的条件：1.全部父节点已经执行完毕，或者没有父节点 2.不需要满足资源充足的条件
    # 这样看上去只需要小修一下下面的程序
    # 但是还有另一种思路，首先拿到第一层节点加入可执行列表，然后在每一次有节点完成后，将完成节点的子节点加入可执行列表
    # 这样的话效率更高，就按这个写吧，顺便也把后面的环境重写了，不然看着糟心

    def ini_runable_nodes_idx(self):
        # idx = np.where(self.jobdag.adj_mat[0] == 1)[0].tolist()
        # idx.remove(0)
        idx = [0, 1, 2]
        return idx

    def check_runable_nodes(self):
        # runable nodes idx里储存的是已经解锁的节点
        # 需要检查三项，1.满足资源条件 2.父节点等于已经完成的父节点 3.此节点没有执行，不能是正在执行或者已经完成
        runable_nodes_idx = []
        runable_nodes = []
        for idx in self.runable_nodes_idx:
            node = self.jobdag.nodes[idx]
            # 检查资源条件
            re1 = self.resource_exec[0] - node.resource1
            re2 = self.resource_exec[1] - node.resource2
            re3 = self.resource_exec[2] - node.resource3
            re4 = self.resource_exec[3] - node.resource4
            # 检查约束条件
            parent_node_set = set(node.parent_nodes)
            completed_parent_set = set(node.completed_parent_nodes)
            # 还有一个条件，就是这个节点不能正在执行，不能已经执行，只能是未执行状态
            # 否则这个节点还是会被assign task，导致同一节点的重复assign

            if re1>=0 and re2>=0 and re3>=0 and re4>=0 and parent_node_set==completed_parent_set and node.condition =='not_start':
                runable_nodes_idx.append(idx)
                runable_nodes.append(node)

        # 然后更新mask
        mask = np.zeros(self.jobdag.num_nodes)
        mask[self.runable_nodes_idx] = 1
        self.now_action_mask = mask
        return runable_nodes_idx, runable_nodes

        # 指派任务，输入想要执行节点的索引和指针，生成任务信息，调整资源，调整任务状态。一次只能指派一个任务
        # 节点索引的格式：[node索引]
    def assign_task(self, node_idx):

        assert node_idx in self.runable_nodes_idx, 'can not assign input node idx'
        self.action_sequence.append(node_idx)
        # 先把这个节点从可执行节点移除，注意
        self.runable_nodes_idx.remove(node_idx)

        node = self.jobdag.nodes[node_idx]
        # 生成new_task
        new_task = Task(node_idx, node, self.walltime)
        self.running_tasks.append(new_task)
        # 改变节点状态
        node.condition = 'running'
        # 改变特征矩阵，把矩阵对应数设为1（代表正在执行）
        self.feature_mat[node_idx][5] = 1
        # 减去资源
        self.resource_exec[0] -= node.resource1
        self.resource_exec[1] -= node.resource2
        self.resource_exec[2] -= node.resource3
        self.resource_exec[3] -= node.resource4
    # 推进时间函数
    # 将时间向前推进1
    def advance_time(self):
        # 推进时间
        # if self.walltime == 800:
        #     print(1)
        self.walltime += 1

        # 不能直接移除！！！会导致Bug，还不会报错。为了找这个bug我用了一晚上！！！
        wait_remove_task = []
        ####
        for task in self.running_tasks:
            # 同时减去特征矩阵中的对应的时间 ###
            self.feature_mat[task.node_idx][0] -= 1
            # task时间记录加一
            task.have_run_time += 1
            # 如果该task已完成，进行下面操作

            if self.feature_mat[task.node_idx][0] == 0:
                wait_remove_task.append(task)

        for task in wait_remove_task:
            # 解锁新节点的操作应该在assign task时就操作，而不是等到有节点完成时操作
            # 后来多了个新的列表储存已经完成的父节点
            for child_node in self.jobdag.nodes[task.node_idx].child_nodes:
                self.jobdag.nodes[child_node].completed_parent_nodes.append(task.node_idx)

            # 移除操作
            self.running_tasks.remove(task)
            self.done_tasks.append(task)
            task.finish_time = self.walltime
            task.during_time = task.finish_time - task.start_time
            # 设置对应节点的condition
            task.node.condition = 'done'
            # 返还资源
            self.resource_exec[0] += task.node.resource1
            self.resource_exec[1] += task.node.resource2
            self.resource_exec[2] += task.node.resource3
            self.resource_exec[3] += task.node.resource4
            # 将特征矩阵one hot编码设置为01(代表已完成)
            self.feature_mat[task.node_idx][5] = 0
            self.feature_mat[task.node_idx][6] = 1
            # 已完成的节点数+1
            self.complete_node += 1