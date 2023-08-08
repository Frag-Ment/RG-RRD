import numpy as np
from rcpsp_simulator.skip_env import Skip_environment
import time

# read instance
all_info = np.load('../PSPLIB_dataset/problems_60.npy', allow_pickle=True)
# read variant information
# variant = np.load('./PSPLIB_dataset/variant/variant_30_B1.npy')

# not use variant, transform to steady
instance_idx = 4
resource = all_info[instance_idx][2]
variant = np.stack([resource for i in range(1000)])

# jobdag = env.executor.jobdag_map
# msg_mat, msg_mask = get_bottom_up_paths(env.executor.jobdag_map, max_depth=256)
# print(msg_mask)

def sample():
    data = []
    for instance_idx in range(480):
        all_time = []
        now_time = time.time()
        for lunshu in range(100):
            env = Skip_environment()
            resource = all_info[instance_idx][2]
            variant = np.stack([resource for ss in range(1000)])
            state = env.reset(all_info[instance_idx][0], all_info[instance_idx][1], all_info[instance_idx][2], variant)
            all_reward = 0
            for i in range(1000):
                # action = state[3][0]
                # print(action)
                # 随机策略
                n_options = len(state[3])
                action_idx = np.random.randint(0, n_options)
                # action_idx = 0
                action = state[3][action_idx]
                # print(action)
                action = np.array(action, dtype='int64')
                state, reward, done = env.step(action)
                # print(state)
                all_reward += reward
                # print(state)
                # print('single reward', reward)
                # print(done)
                # print(done)
                if done == 1:
                    break

            # print('complete time', env.executor.walltime)
            # print('all reward', all_reward)
            all_time.append(env.executor.walltime)
        all = np.stack(all_time)
        print(instance_idx, np.min(all))
        # print('消耗时间：', time.time() - now_time)
        data.append([instance_idx, np.min(all)])

sample()








