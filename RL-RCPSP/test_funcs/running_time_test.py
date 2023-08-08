from test_funcs import sample_average_makespan
from DataLoader import Dataloader
from dataset_info import *

all_sample_times = [1, 10, 20, 30, 40, 50, 60]

model30 = './test_models/set_30'
model60 = './test_models/set_60'
model90 = './test_models/set_90'

dataloader_test_30 = Dataloader(test_set_30)
dataloader_test_60 = Dataloader(test_set_60)
dataloader_test_90 = Dataloader(test_set_90)

all_model = [model30, model60, model90]
all_dataloaders = [dataloader_test_30, dataloader_test_60, dataloader_test_90]

normal_env = False
if_variant = False
greedy_test = False

for n_samples, model, dataloader in zip(all_sample_times, all_model, all_dataloaders):
    sample_average_makespan(model, dataloader, n_sampels=n_samples,
                                                             normal_env=normal_env, variant=if_variant,
                                                             greedy_test=greedy_test)