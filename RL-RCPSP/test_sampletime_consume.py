from test_funcs.sample_average_makespan import sample_average_makespan

from DataLoader import Dataloader
from dataset_info import *

all_sample_times = [1, 10, 20, 30, 40, 50, 60]

model30 = './test_models/set_30.pt'
model60 = './test_models/set_60.pt'
model90 = './test_models/set_90.pt'

dataloader_test_30 = Dataloader(test_set_30)
dataloader_test_60 = Dataloader(test_set_60)
dataloader_test_90 = Dataloader(test_set_90)


all_dataloaders = [dataloader_test_30, dataloader_test_60, dataloader_test_90]


greedy_test = False
normal_env = False
if_variant = False

for n_samples in all_sample_times:
    model = model30
    dataloader = dataloader_test_30
    print('model', model)
    print('n_samples', n_samples)
    average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                             normal_env=normal_env, variant=if_variant,
                                                             greedy_test=greedy_test)
    print('time', average_time)



for n_samples in all_sample_times:
    model = model60
    dataloader = dataloader_test_60
    print('model', model)
    print('n_samples', n_samples)
    average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                             normal_env=normal_env, variant=if_variant,
                                                             greedy_test=greedy_test)
    print('time', average_time)


for n_samples in all_sample_times:
    model = model90
    dataloader = dataloader_test_90
    print('model', model)
    print('n_samples', n_samples)
    average_makespan, average_time = sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                             normal_env=normal_env, variant=if_variant,
                                                             greedy_test=greedy_test)
    print('time', average_time)


# m_30 = [0.0627, 0.2766, 0.5238, 0.7201, 0.9401, 1.1575, 1.3938]
# m_60 = [0.1041, 0.5732, 1.0989, 1.6225, 2.1797, 2.6998, 3.1089]
# m_90 = [0.1683, 0.9249, 1.7769, 2.6277, 3.4841, 4.4243, 5.2254]