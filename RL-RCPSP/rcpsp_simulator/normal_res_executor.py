from rcpsp_simulator.task import Task
import copy
from collections import namedtuple
import numpy as np


class Normal_res_executor(object):
    def __init__(self, jobdag, resource_variant):

        self.jobdag = jobdag

        # 原始资源矩阵，不会变
        self.original_resource_exec = copy.deepcopy(jobdag.resource_exec)
        # 当前剩余资源矩阵, 和jobdag的资源矩阵是同一地址，改变时会同时变，所以不用deepcopy
        self.resource_exec = jobdag.resource_exec
        # 当前消耗资源矩阵，通过合并所有task消耗资源得出
        self.consume_resource = np.array([0, 0, 0, 0])
        # 资源变动矩阵，储存在executor中而不是jobdag中
        self.resource_variant = copy.deepcopy(resource_variant)
        # 时间信息
        self.walltime = 0

        # 可运行节点初始化
        self.runable_nodes_idx = [0, 1, 2]

        # 正在执行的节点的列表
        self.running_tasks = []
        # 已经执行完成的任务的列表
        self.done_tasks = []
        # 所有中断的任务
        self.break_tasks = []

        # 节点的特征矩阵，同样是“一维数组”
        self.feature_mat = jobdag.feature_mat
        # 已经完成的节点的数量
        self.complete_node = 0
        self.action_sequence = []

        self.break_flag = False
        self.break_time = 0
        self.future_resources = []


    # 检查所有可以执行的节点，返回可执行节点的列表
    # 检查所有叶子节点
    # 叶子节点的条件：1.全部父节点已经执行完毕，或者没有父节点 2.不需要满足资源充足的条件
    # 这样看上去只需要小修一下下面的程序
    # 但是还有另一种思路，首先拿到第一层节点加入可执行列表，然后在每一次有节点完成后，将完成节点的子节点加入可执行列表
    # 这样的话效率更高，就按这个写吧，顺便也把后面的环境重写了，不然看着糟心


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
        # 资源消耗矩阵更改
        self.consume_resource[0] += node.resource1
        self.consume_resource[1] += node.resource2
        self.consume_resource[2] += node.resource3
        self.consume_resource[3] += node.resource4

    # 推进时间函数
    # 将时间向前推进1
    def advance_time(self):
        # 推进时间
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
            # 对应的，资源消耗矩阵更改
            self.consume_resource[0] -= task.node.resource1
            self.consume_resource[1] -= task.node.resource2
            self.consume_resource[2] -= task.node.resource3
            self.consume_resource[3] -= task.node.resource4
            # 将特征矩阵one hot编码设置为01(代表已完成)
            self.feature_mat[task.node_idx][5] = 0
            self.feature_mat[task.node_idx][6] = 1
            # 已完成的节点数+1
            self.complete_node += 1

        # 下面是资源变动模块
        # resource_variant中保存的是每个时刻，每种资源的总量
        # 下一时刻的资源总量就是resource_variant
        # 有两种可能， 一种是下一时刻的资源总量仍旧能支持现有所有节点， 另一种是无法支持
        # 计算下一时刻的资源矩阵:下一时刻的总资源-消耗资源
        # if self.walltime > 900:
        #     print(1)
        self.resource_exec = self.resource_variant[self.walltime] - self.consume_resource

        # 如果出现了负资源，需要break
        if all(i>0 for i in self.resource_exec):
            self.break_flag = False
        else:
            self.break_flag = True
            self.break_time += 1
            self.break_operation()
        return self.break_flag

    def break_operation(self):
        # print('break')
        # 重置当前资源矩阵和资源消耗矩阵
        self.resource_exec = self.resource_variant[self.walltime]
        self.consume_resource = np.array([0, 0, 0, 0])

        # 重置所有task
        # 需要重置runable_nodes_idx
        # runable_nodes_idx更新的逻辑是，assign task后，遍历action的子节点，然后在子节点的unlock属性中加入action节点
        # 然后，对照该子节点的unlock节点和它的所有父节点，如果一致，就能够判断该节点已经解锁，将其加入可执行节点列表
        # 如果做逆向的话，就将这些加入的unlock节点全部抹除
        # 遍历每个中断的节点，这些节点在assign task时已经进行了解锁节点操作，找到这些解锁的节点然后删除该中断节点

        wait_remove_task = copy.deepcopy(self.running_tasks)
        for task in wait_remove_task:
            self.runable_nodes_idx.append(task.node_idx)
            task.break_info = True
            self.break_tasks.append(task)

            task.finish_time = self.walltime
            task.during_time = task.finish_time - task.start_time
            # 设置对应节点的condition
            task.node.condition = 'break'
            self.jobdag.nodes[task.node_idx].condition = 'not_start'

            # 将特征矩阵one hot编码设置为00，代表未进行
            self.feature_mat[task.node_idx][5] = 0
            self.feature_mat[task.node_idx][6] = 0

            # 将所有子节点中的unlock信息全部重置
            for child_node_idx in self.jobdag.nodes[task.node_idx].child_nodes:
                # 将子节点中的unlock信息去除
                self.jobdag.nodes[child_node_idx].unlocked_parent_nodes.remove(task.node_idx)
                # 如果子节点还在runable node idx中，将其去除
                if child_node_idx in self.runable_nodes_idx:
                    self.runable_nodes_idx.remove(child_node_idx)

        self.running_tasks = []




