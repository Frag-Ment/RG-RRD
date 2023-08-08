from test_funcs.compute_greedy_makespan import greedy_average_makespan
from test_funcs.sample_average_makespan import sample_average_makespan
from dataset_info import *
from DataLoader import Dataloader
from rcpsp_simulator.skip_env import Skip_environment
from rcpsp_simulator.skip_res_env import Skip_res_environment
from rcpsp_simulator.skip_env import Skip_environment
import numpy as np
import time
import torch

# U1
res_U1_30 = './test_models/res_U1_30.pt'
res_U1_60 = './test_models/res_U1_60.pt'
res_U1_90 = './test_models/res_U1_90.pt'
res_U1_120 = './test_models/res_U1_120.pt'
res_U1_mix = './test_models/res_U1_mix.pt'

dataloader_test_U1_30 = Dataloader(test_set_U1_30)
dataloader_test_U1_60 = Dataloader(test_set_U1_60)
dataloader_test_U1_90 = Dataloader(test_set_U1_90)
dataloader_test_U1_120 = Dataloader(test_set_U1_120)
dataloader_test_U1_mix = Dataloader(test_set_U1_mix)

# B1
res_B1_30 = './test_models/res_B1_30.pt'
res_B1_60 = './test_models/res_B1_60.pt'
res_B1_90 = './test_models/res_B1_90.pt'
res_B1_120 = './test_models/res_B1_120.pt'
res_B1_mix = './test_models/res_B1_mix.pt'

dataloader_test_B1_30 = Dataloader(test_set_B1_30)
dataloader_test_B1_60 = Dataloader(test_set_B1_60)
dataloader_test_B1_90 = Dataloader(test_set_B1_90)
dataloader_test_B1_120 = Dataloader(test_set_B1_120)
dataloader_test_B1_mix = Dataloader(test_set_B1_mix)

# Exp
res_Exp_30 = './test_models/res_Exp_30.pt'
res_Exp_60 = './test_models/res_Exp_60.pt'
res_Exp_90 = './test_models/res_Exp_90.pt'
res_Exp_120 = './test_models/res_Exp_120.pt'
res_Exp_mix = './test_models/res_Exp_mix.pt'

dataloader_test_Exp_30 = Dataloader(test_set_Exp_30)
dataloader_test_Exp_60 = Dataloader(test_set_Exp_60)
dataloader_test_Exp_90 = Dataloader(test_set_Exp_90)
dataloader_test_Exp_120 = Dataloader(test_set_Exp_120)
dataloader_test_Exp_mix = Dataloader(test_set_Exp_mix)



# U1
all_U1_model = [res_U1_30, res_U1_60, res_U1_90, res_U1_120, res_U1_mix]
all_U1_dataloader = [dataloader_test_U1_30, dataloader_test_U1_60, dataloader_test_U1_90,
                  dataloader_test_U1_120, dataloader_test_U1_mix]
# B1
all_B1_model = [res_B1_30, res_B1_60, res_B1_90, res_B1_120, res_B1_mix]
all_B1_dataloader = [dataloader_test_B1_30, dataloader_test_B1_60, dataloader_test_B1_90,
                  dataloader_test_B1_120, dataloader_test_B1_mix]
# Exp
all_Exp_model = [res_Exp_30, res_Exp_60, res_Exp_90, res_Exp_120, res_Exp_mix]
all_Exp_dataloader = [dataloader_test_Exp_30, dataloader_test_Exp_60, dataloader_test_Exp_90,
                  dataloader_test_Exp_120, dataloader_test_Exp_mix]


n_samples = 30
greedy_test = False
normal_env = False
if_variant = True

# test function
test_model = all_B1_model
test_dataloader = all_B1_dataloader

for model, dataloader in zip(test_model, test_dataloader):
    print(model)
    average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                          normal_env=normal_env, variant=if_variant,
                                                          greedy_test=greedy_test)
    print(average_makespan, average_time)

