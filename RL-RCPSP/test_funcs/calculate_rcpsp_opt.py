import numpy as np

j30_opt_path = '../PSPLIB_dataset/opt_30.npy'
j60_opt_path = '../PSPLIB_dataset/opt_60.npy'
j90_opt_path = '../PSPLIB_dataset/opt_90.npy'
j120_opt_path = '../PSPLIB_dataset/opt_120.npy'

all_path = [j30_opt_path, j60_opt_path, j90_opt_path, j120_opt_path]

mean_list = []
for path in all_path:
    data = np.load(path, allow_pickle=True)
    print(np.mean(data[0: 90]))
    mean_list.append(np.mean(data[0: 90]))

print(mean_list)
breakpoint()
