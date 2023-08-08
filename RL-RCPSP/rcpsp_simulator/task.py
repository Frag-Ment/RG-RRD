import numpy as np


class Task(object):
    def __init__(self, node_idx, node, start_time):
        # node_idx： [node索引]
        self.node_idx = int(node_idx)

        self.start_time = start_time
        self.node = node

        self.finish_time = None
        self.during_time = None

        self.break_info = False

        self.have_run_time = 0

#     def task_break(self, break_info):
# # break_info:[第几次break的]
