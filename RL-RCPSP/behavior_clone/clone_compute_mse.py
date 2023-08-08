import numpy as np
from behavior_clone.clone_test_train import test_train


def compute_mse(test_start_idx, test_end_idx):
    test_train(test_start_idx, test_end_idx)

    opt = np.loadtxt('../30随机.txt', usecols=1)
    our = np.loadtxt('../behavior_clone/clone_walltime.txt', usecols=1)
    worst = np.loadtxt('../30随机上界.txt', usecols=1)
    average = np.loadtxt('../30平均.txt', usecols=1)

    # print('opt len:', opt.shape[0])
    # print('our len:', our.shape[0])
    # print('worst len:', worst.shape[0])

    num_data = min(opt.shape[0], our.shape[0], worst.shape[0], average.shape[0])
    print('use len:', num_data)

    our_opt_mse = np.mean(our - opt[test_start_idx:test_end_idx])
    # worst_opt_mse = np.mean(worst[test_start_idx:test_end_idx] - opt[test_start_idx:test_end_idx])
    average_opt_mse = np.mean(average[test_start_idx:test_end_idx] - opt[test_start_idx:test_end_idx])
    print('our_opt_mse', our_opt_mse)
    # print('worst_opt_mse', worst_opt_mse)
    print('average_opt_mse', average_opt_mse)
    return our_opt_mse

compute_mse(0, 70)