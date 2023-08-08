from rcpsp_simulator.normal_env import Normal_environment
import numpy as np
import pickle
import copy

# type of instance
instance_type = 30

# read instance
all_info = np.load('../PSPLIB_dataset/problems_{}.npy'.format(instance_type), allow_pickle=True)
action_sequence = np.load('../behavior_clone/instance30_dataset/{}随机解.npy'.format(instance_type), allow_pickle=True)

# not use variant, transform to steady
instance_idx = 4
resource = all_info[instance_idx][2]
variant = np.stack([resource for i in range(1000)])


def generate_behavior_cloning_dataset():
    adj_dataset = []
    fea_dataset = []
    resource_dataset = []
    runable_dataset = []
    reward_dataset = []
    for instance_idx in range(0, 1):
        adj_list = []
        fea_list = []
        resource_list = []
        runable_list = []
        reward_list = []
        print('instance idx', instance_idx)
        for lunshu in range(1):
            env = Normal_environment()
            resource = all_info[instance_idx][2]
            variant = np.stack([resource for ss in range(1000)])
            state = env.reset(all_info[instance_idx][0], all_info[instance_idx][1], all_info[instance_idx][2], variant)
            for step in range(1000000):
                adj_list.append(copy.deepcopy(state[0]))
                fea_list.append(copy.deepcopy(state[1]))
                resource_list.append(copy.deepcopy(state[2]))
                runable_list.append(copy.deepcopy(state[3]))

                action = action_sequence[instance_idx][step]
                state, reward, done = env.step(int(action))

                # mask校验
                # have_mask = state[-1]
                # runable_nodes_idx = state[-2]
                # mask = np.zeros(32)
                # mask[runable_nodes_idx] = 1
                # for i in range(32):
                #     assert mask[i] == have_mask[i]

                reward_list.append(copy.deepcopy(reward))
                if done == 1:
                    print('makespan: ', env.executor.walltime)
                    break

        adj_dataset.append(adj_list)
        fea_dataset.append(fea_list)
        resource_dataset.append(resource_list)
        runable_dataset.append(runable_list)
        reward_dataset.append(reward_list)

    np.save('../behavior_clone/instance{}_dataset/adj_dataset.npy'.format(instance_type), np.stack(adj_dataset))
    np.save('../behavior_clone/instance{}_dataset/fea_dataset.npy'.format(instance_type), np.stack(fea_dataset))
    np.save('../behavior_clone/instance{}_dataset/resource_dataset.npy'.format(instance_type), np.stack(resource_dataset))
    np.save('../behavior_clone/instance{}_dataset/runable_dataset.npy'.format(instance_type), np.stack(runable_dataset))
    np.save('../behavior_clone/instance{}_dataset/reward_dataset.npy'.format(instance_type), np.stack(reward_dataset))


generate_behavior_cloning_dataset()

# fea = np.load('../behavior_clone/instance30_dataset/fea_dataset.npy', allow_pickle=True)
# print(fea)

# 生成action的one hot编码



