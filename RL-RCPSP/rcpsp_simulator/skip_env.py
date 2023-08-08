import copy
from rcpsp_simulator.jobdag import Jobdag
import numpy as np
from collections import namedtuple
from rcpsp_simulator.skip_executor import Skip_executor
from rcpsp_simulator.task import Task


# 各个类的包含关系：
# Env>Executor>Jobdag_map>Jobdag>node
# Env>Executor>task

# Env的功能：next_state, reward, done = env.step(action)
# Env的功能：state = env.reset()
# 数据格式：
# state:  特征矩阵， 要执行的节点（二维数组，Index0：jobdag编号  Index1:可执行节点编号）， 资源和工人状态， 邻接矩阵从env的self中读取


class Skip_environment(object):
    def __init__(self):
        self.executor = None
        # 邻接矩阵的储存格式：idx0:jobdag编号 idx1:邻接矩阵
        self.adj_mat = None
        # self.reverse_adj_mat = []

        self.tep = namedtuple('state', ['adj_mat', 'feature_mat', 'resource_exec',
                                        'runable_nodes_idx', 'action_mask'])
        # # 拿到资源和工人状态表 ###资源和工人状态是不断变化的！这两行要重写！
        # self.workers_exec = executor.workers_exec
        # self.stations_exec = executor.stations_exec
        self.last_decision_time = 0

    def check_state(self):
        # 该函数作用是返回下面的state
        adj_mat_state = self.adj_mat
        # reverse_adj_mat_state = self.reverse_adj_mat
        feature_mat_state = self.executor.feature_mat
        resource_exec_state = self.executor.resource_exec
        return adj_mat_state,  feature_mat_state, resource_exec_state

    def reset(self, adj_mat, nodes_information, resource, resource_variant):

        # 1.重置env的状态, 也就是重新装载所有数据， 所以需要输入任务信息
        jobdag = Jobdag(adj_mat, nodes_information, resource)
        executor = Skip_executor(jobdag, resource_variant)
        self.executor = executor
        self.adj_mat = copy.deepcopy(self.executor.jobdag.adj_mat)
        self.last_decision_time = 0

        # 2.返回初始state
        state = self.tep(self.adj_mat, self.executor.feature_mat, self.executor.resource_exec,
                         self.executor.runable_nodes_idx, self.executor.now_action_mask)
        return state

    def step(self, action):
        action = int(action)

        assert action in self.executor.runable_nodes_idx

        # if len(self.executor.action_sequence) == self.executor.jobdag.num_nodes - 1:
        #     return self.return_last_state(action)
        ##### 在此程序中，下面三条命令一般是在一起的
        self.executor.assign_task(action)
        self.add_action_to_runable_idx(action)
        runable_nodes_idx, runable_nodes = self.executor.check_runable_nodes()
        #####

        while len(runable_nodes) == 1:
            # 如果是最后一个节点
            # action sequence肯定要改
            if len(self.executor.action_sequence) == self.executor.jobdag.num_nodes - 1:
                return self.return_last_state(runable_nodes_idx[0])

            self.executor.assign_task(runable_nodes_idx[0])
            self.add_action_to_runable_idx(runable_nodes_idx[0])

            self.executor.advance_time()
            runable_nodes_idx, runable_nodes = self.executor.check_runable_nodes()

        if len(runable_nodes) > 1:
            adj_mat_state, feature_mat_state, resource_exec_state,  = self.check_state()
            state = self.tep(adj_mat_state, feature_mat_state, resource_exec_state,
                              runable_nodes_idx, self.executor.now_action_mask)

            reward = self.executor.walltime - self.last_decision_time
            self.last_decision_time = self.executor.walltime
            done = 0
            return state, reward, done

        while len(runable_nodes) == 0:
            ### 先判断一下是否已经做完所有节点
            if self.executor.complete_node == self.executor.jobdag.num_nodes:
            # 注意判断条件，由于第一个和最后一个是虚拟节点，因此可能需要修改判断标准
                adj_mat_state, feature_mat_state, resource_exec_state = self.check_state()
                state = self.tep(adj_mat_state, feature_mat_state, resource_exec_state,
                                  runable_nodes_idx, self.executor.now_action_mask)

                done = 1
                # 计算reward
                reward = self.executor.walltime - self.last_decision_time
                self.last_decision_time = self.executor.walltime
                return state, reward, done

            # 然后推进时间
            self.executor.advance_time()
            runable_nodes_idx, runable_nodes = self.executor.check_runable_nodes()
            # 推进时间后出现三种情况 1.可执行节点数等于1  2.大于1  3.等于0
            # 节点数为1
            while len(runable_nodes) == 1:
                if len(self.executor.action_sequence) == self.executor.jobdag.num_nodes - 1:
                    return self.return_last_state(runable_nodes_idx[0])
                self.executor.assign_task(runable_nodes_idx[0])
                self.add_action_to_runable_idx(runable_nodes_idx[0])
                self.executor.advance_time()

                runable_nodes_idx, runable_nodes = self.executor.check_runable_nodes()
            # 节点数大于1
            if len(runable_nodes) > 1:
                adj_mat_state, feature_mat_state, resource_exec_state = self.check_state()
                state = self.tep(adj_mat_state, feature_mat_state, resource_exec_state,
                                  runable_nodes_idx, self.executor.now_action_mask)

                reward = self.executor.walltime - self.last_decision_time
                self.last_decision_time = self.executor.walltime
                done = 0
                return state, reward, done

    def add_action_to_runable_idx(self, action):
        for child_node in self.executor.jobdag.nodes[action].child_nodes:
            self.executor.jobdag.nodes[child_node].unlocked_parent_nodes.append(action)
            if set(self.executor.jobdag.nodes[child_node].unlocked_parent_nodes) == set(self.executor.jobdag.nodes[child_node].parent_nodes) :
                self.executor.runable_nodes_idx.append(self.executor.jobdag.nodes[child_node].idx)


    def return_last_state(self, action):
        # if action == 28:
        #     print(action)

        if len(self.executor.action_sequence) == self.executor.jobdag.num_nodes - 1:
            self.executor.assign_task(action)

            while self.executor.complete_node != self.executor.jobdag.num_nodes:
                self.executor.advance_time()


            return self.tep(self.adj_mat, self.executor.feature_mat, self.executor.resource_exec,
                            self.executor.runable_nodes_idx, np.zeros(self.executor.jobdag.num_nodes)), self.executor.walltime - self.last_decision_time, 1
