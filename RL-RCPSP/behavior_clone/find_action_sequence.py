import numpy as np
import time
from rcpsp_simulator.skip_env import Skip_environment

# 1.找到最优或者较优解，使用skip env
# 2.使用normal env制作数据集 输入：state 输出 action reward


instance_type = 30
# read instance
all_info = np.load('../PSPLIB_dataset/problems_{}.npy'.format(instance_type), allow_pickle=True)
opt_time = np.loadtxt('../{}随机.txt'.format(instance_type), usecols=1)


# variant是不使用的，所以随便输入一个就行
instance_idx = 0
resource = all_info[instance_idx][2]
variant = np.stack([resource for i in range(1000)])

def generate_behavior_cloneing_dataset():
    standard_solution = []
    for instance_idx in range(0, 80):
        print(instance_idx)
        standard_time = opt_time[instance_idx]
        all_time = []
        now_time = time.time()

        for lunshu in range(1000000000000):
            if lunshu > 900000000000:
                print('error')
            env = Skip_environment()
            resource = all_info[instance_idx][2]
            variant = np.stack([resource for ss in range(1000)])
            state = env.reset(all_info[instance_idx][0], all_info[instance_idx][1], all_info[instance_idx][2], variant)
            all_reward = 0
            for i in range(1000):
                # 随机动作
                n_options = len(state[3])
                action_idx = np.random.randint(0, n_options)
                action = state[3][action_idx]

                action = np.array(action, dtype='int64')
                state, reward, done = env.step(action)

                all_reward += reward
                if done == 1:
                    break
            if env.executor.walltime == standard_time:
                standard_solution.append(env.executor.action_sequence)
                break
    standard_solution = np.stack(standard_solution)
    # print(standard_solution)
    np.save('../behavior_clone/instance30_dataset/{}随机解.npy'.format(instance_type), standard_solution)

generate_behavior_cloneing_dataset()

# sol = np.load('../behavior_clone/instance30_dataset/{}随机解.npy'.format(instance_type), allow_pickle=True)
# print(sol[0])
