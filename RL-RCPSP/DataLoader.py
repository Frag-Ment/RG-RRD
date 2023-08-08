import numpy as np
import time
import copy
# 问题有4种类型，30，60，90，120
# 每种问题一共480个案例
# 每种问题都有5种资源扰动：U1 U2 B1 B2 Exp 或许不扰动也可以视为一种扰动, 那么一共6种。
# 由于不扰动的环境本身也用不到resource variant这个参数，那么默认是U1好了
# 编码方案如下


class Dataloader():
    def __init__(self, dataset_info, repeat_times=1):
        self.dataset_info = dataset_info
        self.ins_dataset = []
        self.adj_dataset = []
        self.fea_dataset = []
        self.res_dataset = []
        self.var_dataset = []
        self.now_instance_idx = 0
        self.num_instances = None
        self.dataset_loop_times = 0
        # 调用next_instance时重复一个例子多少次
        self.repeat_times = repeat_times
        self.need_repeat_times = self.repeat_times
        self.generate_dataset()

    def generate_dataset(self):
        for command in self.dataset_info:
            instance_size = command[0]
            start_idx = command[1]
            end_idx = command[2]
            variant_type = command[3]
            if variant_type == 'None':
                variant_type = 'B1'
            instances = np.load(f'./PSPLIB_dataset/problems_{instance_size}.npy', allow_pickle=True)
            variants = np.load(f'./PSPLIB_dataset/variant/variant_{instance_size}_{variant_type}.npy', allow_pickle=True)
            for idx in range(start_idx, end_idx):
                self.ins_dataset.append(copy.deepcopy(instances[idx]))
                self.adj_dataset.append(copy.deepcopy(instances[idx][0]))
                self.fea_dataset.append(copy.deepcopy(instances[idx][1]))
                self.res_dataset.append(copy.deepcopy(instances[idx][2]))
                self.var_dataset.append(copy.deepcopy(variants[idx]))
        self.num_instances = len(self.ins_dataset)


    def next_instance(self):
        if self.need_repeat_times == 0:
            self.need_repeat_times = self.repeat_times
            self.now_instance_idx += 1

        if self.now_instance_idx > self.num_instances - 1:
            self.now_instance_idx = 0
            self.dataset_loop_times += 1
        self.need_repeat_times -= 1

        return self.adj_dataset[self.now_instance_idx], self.fea_dataset[self.now_instance_idx],\
               self.res_dataset[self.now_instance_idx], self.var_dataset[self.now_instance_idx]

    def read_instance(self, idx):
        self.now_instance_idx = idx
        return self.adj_dataset[self.now_instance_idx], self.fea_dataset[self.now_instance_idx], \
               self.res_dataset[self.now_instance_idx], self.var_dataset[self.now_instance_idx]

