from test_funcs.compute_greedy_makespan import greedy_average_makespan
from test_funcs.sample_average_makespan import sample_average_makespan
from dataset_info import *
from DataLoader import Dataloader
from rcpsp_simulator.skip_env import Skip_environment
import numpy as np
import time

model_30 = './test_models/set_30.pt'
model_60 = './test_models/set_60.pt'
model_90 = './test_models/set_90.pt'
model_120 = './test_models/set_120.pt'
model_mix = './test_models/set_mix.pt'

dataloader_test_30 = Dataloader(test_set_30)
dataloader_test_60 = Dataloader(test_set_60)
dataloader_test_90 = Dataloader(test_set_90)
dataloader_test_120 = Dataloader(test_set_120)
dataloader_test_mix = Dataloader(test_set_mix)

n_samples = 30
greedy_test = False
normal_env = False
if_variant = False


## normal test ###
all_model = [model_30, model_60, model_90, model_120, model_mix]
all_dataloader = [dataloader_test_30, dataloader_test_60, dataloader_test_90,
                  dataloader_test_120, dataloader_test_mix]

# ## train on small instances and test on large instances ###
# all_model = [model_30, model_30, model_30, model_30]
# all_dataloader = [dataloader_test_60, dataloader_test_90,
#                   dataloader_test_120, dataloader_test_mix]

# ## train on large instances and test on small instances ###
# all_model = [model_90, model_90, model_90]
# all_dataloader = [dataloader_test_30, dataloader_test_60, dataloader_test_mix]

# test sample time
# all_model = [model_60]
# all_dataloader = [dataloader_test_60]

for model, dataloader in zip(all_model, all_dataloader):
    print(model)
    average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                             normal_env=normal_env, variant=if_variant,
                                                             greedy_test=greedy_test)
    print(average_makespan, average_time)


### test sample time ##
# sample_times_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# all_model = [model_60]
# all_dataloader = [dataloader_test_60]
#
# for model, dataloader in zip(all_model, all_dataloader):
#     for n_samples in sample_times_list:
#         print(n_samples)
#         average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
#                                                                  normal_env=normal_env, variant=if_variant,
#                                                                  greedy_test=greedy_test)
#         print(average_makespan, average_time)