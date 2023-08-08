import numpy as np
# 定义节点
# 节点的属性有 1.时间 2.工人数 3.工种 4.工位  前节点列表  是否完工
class Node(object):
    def __init__(self, idx, task_duration, resource1, resource2, resource3, resource4):
        # 自身属性
        self.idx = idx
        self.task_duration = task_duration
        self.rest_time = task_duration

        self.resource1 = resource1
        self.resource2 = resource2
        self.resource3 = resource3
        self.resource4 = resource4

        # 节点相关
        self.parent_nodes = []
        self.child_nodes = []
        self.descendant_nodes = []

        # 已经加工完成的父节点，在节点完成后加入
        self.completed_parent_nodes = []
        # 意思是正在加工和已经加工完成的父节点，在assign task之后就加入
        self.unlocked_parent_nodes = []

        # 自身状态 有三种：'running' 'done' 'not_start'
        self.condition = 'not_start'
        # 这个工序是否能执行，能为1，不能为0
        self.if_could_run = None
        # 这个参数是为了在检查可执行节点时节省步骤，None为未知，False为不需要检查，True为需要检查
        # 称为需要检查标志
        self.if_need_check = None




    # # 改变节点的状态
    # def set_condition(self, condition):


