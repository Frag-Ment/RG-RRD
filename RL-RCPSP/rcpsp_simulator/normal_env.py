import copy
from rcpsp_simulator.jobdag import Jobdag
import numpy as np
from collections import namedtuple
from rcpsp_simulator.normal_executor import Normal_executor
from rcpsp_simulator.task import Task


# 各个类的包含关系：
# Env>Executor>Jobdag_map>Jobdag>node
# Env>Executor>task

# Env的功能：next_state, reward, done = env.step(action)
# Env的功能：state = env.reset()
# 数据格式：
# state:  特征矩阵， 要执行的节点（二维数组，Index0：jobdag编号  Index1:可执行节点编号）， 资源和工人状态， 邻接矩阵从env的self中读取


class Normal_environment(object):
    def __init__(self):
        self.executor = None
        # 邻接矩阵的储存格式：idx0:jobdag编号 idx1:邻接矩阵
        # self.reverse_adj_mat = []
        self.adj_mat = None
        self.tep = namedtuple('state', ['adj_mat', 'feature_mat', 'resource_exec',
                                        'runable_nodes_idx', 'action_mask'])
        # # 拿到资源和工人状态表 ###资源和工人状态是不断变化的！这两行要重写！
        # self.workers_exec = executor.workers_exec
        # self.stations_exec = executor.stations_exec
        self.last_decision_time = 0


    def reset(self, adj_mat, nodes_information, resource, resource_variant):

        # 1.重置env的状态, 也就是重新装载所有数据， 所以需要输入任务信息
        jobdag = Jobdag(adj_mat, nodes_information, resource)
        executor = Normal_executor(jobdag, resource_variant)
        self.executor = executor
        self.last_decision_time = 0
        self.adj_mat = self.executor.jobdag.adj_mat
        # 2.返回初始state
        state = self.tep(self.adj_mat, self.executor.feature_mat, self.executor.resource_exec,
                         self.executor.runable_nodes_idx, self.executor.now_action_mask)
        return state

    def step(self, action):

        action = int(action)
        assert action in self.executor.runable_nodes_idx
        # 先检查一下这个节点是不是最后一个节点，如果是，需要尽可能早执行，然后assign task
        # 如果是最后一个节点，它的父节点都已经assign task，只要什么都不干再等一会就做好了
        if len(self.executor.action_sequence) == self.executor.jobdag.num_nodes - 1:

            self.executor.action_sequence.append(action)
            self.last_decision_time = self.executor.walltime

            while (self.executor.resource_exec[0] - self.executor.jobdag.nodes[action].resource1 < 0 or
                   self.executor.resource_exec[1] - self.executor.jobdag.nodes[action].resource2 < 0 or
                   self.executor.resource_exec[2] - self.executor.jobdag.nodes[action].resource3 < 0 or
                   self.executor.resource_exec[3] - self.executor.jobdag.nodes[action].resource4 < 0 or
                   # 检查这个节点的所有前置节点是否完成
                   set(self.executor.jobdag.nodes[action].parent_nodes) !=
                   set(self.executor.jobdag.nodes[action].completed_parent_nodes)):

                self.executor.advance_time()

            self.executor.assign_task(action)

            while(self.executor.complete_node != self.executor.jobdag.num_nodes):
                self.executor.advance_time()

            self.executor.feature_mat[-1][-1] = 1
            self.executor.feature_mat[-1][-2] = 0

            return self.tep(self.adj_mat, self.executor.feature_mat, self.executor.resource_exec,
                            self.executor.runable_nodes_idx, np.zeros(self.executor.jobdag.num_nodes)), self.executor.walltime - self.last_decision_time, 1

        # action是jobdag中的索引，是1个数字
        # 这个action不一定能立刻执行，但是一定要优先执行
        # 检查现在是否有充足资源执行这个action，如果不行，就推进时间，直到有充足资源
        # 除了资源，我忽略了一个条件，就是它的前置节点要真正执行完毕才可以开始执行它
        # 资源要充足，前置节点要完成，然后才能开始这个节点，如果打不到要求就

        while (self.executor.resource_exec[0] - self.executor.jobdag.nodes[action].resource1 < 0 or
               self.executor.resource_exec[1] - self.executor.jobdag.nodes[action].resource2 < 0 or
               self.executor.resource_exec[2] - self.executor.jobdag.nodes[action].resource3 < 0 or
               self.executor.resource_exec[3] - self.executor.jobdag.nodes[action].resource4 < 0 or
        # 检查这个节点的所有前置节点是否完成
            set(self.executor.jobdag.nodes[action].parent_nodes) !=
            set(self.executor.jobdag.nodes[action].completed_parent_nodes)):
            self.executor.advance_time()

        self.executor.assign_task(action)

        # 解锁新节点操作，在assign_task之后，后续的节点其实已经算是解锁了
        # 找到该节点的子节点，并将自身idx加入子节点的已完成父节点集合中，最后检查该子节点是否解锁
        # 但是要注意，最后一个节点是不能执行的，所以不能把最后一个节点放进去
        for child_node in self.executor.jobdag.nodes[action].child_nodes:
            self.executor.jobdag.nodes[child_node].unlocked_parent_nodes.append(action)
            if set(self.executor.jobdag.nodes[child_node].unlocked_parent_nodes) == \
                    set(self.executor.jobdag.nodes[child_node].parent_nodes) :
                self.executor.runable_nodes_idx.append(self.executor.jobdag.nodes[child_node].idx)
        # 然后再更新mask
        mask = np.zeros(self.executor.jobdag.num_nodes)
        mask[self.executor.runable_nodes_idx] = 1
        self.executor.now_action_mask = mask

        # this_decision_time = self.executor.walltime
        reward = self.executor.walltime - self.last_decision_time
        self.last_decision_time = self.executor.walltime

        return self.tep(self.adj_mat, self.executor.feature_mat, self.executor.resource_exec,
                        self.executor.runable_nodes_idx, self.executor.now_action_mask), reward, 0

        # if len(self.executor.runable_nodes_idx) > 0:
        #     return self.tep(self.adj_mat, self.executor.resource_exec, self.executor.feature_mat,
        #                     self.executor.runable_nodes_idx), reward, 0
        # # 检查节点是否已经全部做完
        # # runable_node_idx是一直都有的，直到最后做完的时刻，所以只需要考虑最后时刻不出bug
        # # 当没有可进行的节点时，代表刚刚assign的已经是最后一个节点   错误
        # # 也有可能是父节点全部都在加工，子节点尚未解锁
        #
        # else:
        #     while len(self.executor.runable_nodes_idx) == 0:
        #         self.executor.advance_time()
        #         if self.executor.complete_node == self.executor.jobdag.num_nodes:
        #             return self.tep(self.adj_mat, self.executor.resource_exec, self.executor.feature_mat,
        #                             self.executor.runable_nodes_idx), self.executor.walltime - this_decision_time, 1
        #
        #     return self.tep(self.adj_mat, self.executor.resource_exec, self.executor.feature_mat,
        #                     self.executor.runable_nodes_idx), self.executor.walltime - this_decision_time, 0

            # while (self.executor.complete_node != self.executor.jobdag.num_nodes):
            #     self.executor.advance_time()
            #
            # return self.tep(self.adj_mat, self.executor.resource_exec, self.executor.feature_mat,
            #                 self.executor.runable_nodes_idx), self.executor.walltime - this_decision_time, 1
