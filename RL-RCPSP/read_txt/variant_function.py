import numpy as np
import copy
import time

# 找到资源的最低限度
def find_LR(feature_mat):
    n_nodes = len(feature_mat)
    LR1 = 0
    LR2 = 0
    LR3 = 0
    LR4 = 0
    for i in range(0, n_nodes):
        if feature_mat[i][1] > LR1:
            LR1 = feature_mat[i][1]
        if feature_mat[i][2] > LR2:
            LR2 = feature_mat[i][2]
        if feature_mat[i][3] > LR3:
            LR3 = feature_mat[i][3]
        if feature_mat[i][4] > LR4:
            LR4 = feature_mat[i][4]
    LR_list = [LR1, LR2, LR3, LR4]
    return LR_list

def generate_varient_matrix(feature_mat, resource_capacity, type):
    LR = find_LR(feature_mat)
    expect = resource_capacity
    if type == 'U1':
        LR1 = LR[0]
        R1 = expect[0]

        if LR1 == 2*R1-LR1:
            distribute1 = np.ones([1000, 1]) * LR1
        else:
            distribute1 = np.random.uniform(LR1, 2*R1-LR1, [1000, 1])
            distribute1 = np.round(distribute1)

        LR2 = LR[1]
        R2 = expect[1]
        if LR2 == 2*R2-LR2:
            distribute2 = np.ones([1000, 1]) * LR2
        else:
            distribute2 = np.random.uniform(LR2, 2*R2-LR2, [1000, 1])
            distribute2 = np.round(distribute2)

        LR3 = LR[2]
        R3 = expect[2]
        if LR3 == 2*R3-LR3:
            distribute3 = np.ones([1000, 1]) * LR3
        else:
            distribute3 = np.random.uniform(LR3, 2*R3-LR3, [1000, 1])
            distribute3 = np.round(distribute3)

        LR4 = LR[3]
        R4 = expect[3]
        if LR4 == 2*R4-LR4:
            distribute4 = np.ones([1000, 1]) * LR4
        else:
            distribute4 = np.random.uniform(LR4, 2*R4-LR4, [1000, 1])
            distribute4 = np.round(distribute4)

        variant_matrix = np.hstack([distribute1, distribute2, distribute3, distribute4])
        return variant_matrix
########################################
    if type == 'U2':
        LR1 = LR[0]
        R1 = expect[0]


        if (R1 +LR1)/2 == (3*R1-LR1)/2:
            distribute1 = np.ones([1000, 1]) * LR1
        else:
            distribute1 = np.random.uniform((R1 +LR1)/2, (3*R1-LR1)/2, [1000, 1])
            distribute1 = np.round(distribute1)

        for i in range(1, 4):
            LR1 = LR[i]
            R1 = expect[i]
            if (R1 + LR1) / 2 == (3 * R1 - LR1) / 2:
                distribute = np.ones([1000, 1]) * LR1
            else:
                distribute = np.random.uniform((R1 + LR1) / 2, (3 * R1 - LR1) / 2, [1000, 1])
                distribute = np.round(distribute)
            variant_matrix = np.hstack([distribute1, distribute])
            distribute1 = variant_matrix

        return variant_matrix
###################################################
    if type == 'Exp':
        LR1 = LR[0]
        R1 = expect[0]
        distribute1 = np.random.exponential(scale=R1, size=[1000, 1])
        distribute1 = np.round(distribute1)

        for i in range(0, 1000):
            if distribute1[i][0] < LR1:
                distribute1[i][0] = LR1

        for j in range(1, 4):
            LR1 = LR[j]
            R1 = expect[j]
            distribute = np.random.exponential(scale=R1, size=[1000, 1])
            distribute = np.round(distribute)
            for i in range(0, 1000):
                if distribute[i][0] < LR1:
                    distribute[i][0] = LR1
            variant_matrix = np.hstack([distribute1, distribute])
            distribute1 = variant_matrix
        return variant_matrix
###################################################
    if type == 'B1':
        LR1 = LR[0]
        R1 = expect[0]
        Range = 2*R1 - 2*LR1
        if Range == 0:
            distribute1 = np.ones([1000, 1]) * LR1
        else:
            distribute1 = np.random.beta(a=1, b=1, size=[1000, 1]) * Range + LR1
            distribute1 = np.round(distribute1)

        for j in range(1, 4):
            LR1 = LR[j]
            R1 = expect[j]
            Range = 2 * R1 - 2 * LR1

            if Range == 0:
                distribute = np.ones([1000, 1]) * LR1
            else:
                distribute = np.random.beta(a=1, b=1, size=[1000, 1]) * Range + LR1
                distribute = np.round(distribute)

            variant_matrix = np.hstack([distribute1, distribute])
            distribute1 = variant_matrix
        return variant_matrix

    if type == 'B2':
        LR1 = LR[0]
        R1 = expect[0]
        Range = 2*R1 - 2*LR1
        if Range == 0:
            distribute1 = np.ones([1000, 1]) * LR1
        else:
            distribute1 = np.random.beta(a=11/2, b=11/2, size=[1000, 1]) * Range + LR1
            distribute1 = np.round(distribute1)

        for j in range(1, 4):
            LR1 = LR[j]
            R1 = expect[j]
            Range = 2 * R1 - 2 * LR1

            if Range == 0:
                distribute = np.ones([1000, 1]) * LR1
            else:
                distribute = np.random.beta(a=11/2, b=11/2, size=[1000, 1]) * Range + LR1
                distribute = np.round(distribute)

            variant_matrix = np.hstack([distribute1, distribute])
            distribute1 = variant_matrix
        return variant_matrix


# 生成资源矩阵
all_info = np.load('../PSPLIB_dataset/problems_120.npy', allow_pickle=True)
all_type = ['U1', 'U2', 'B1', 'B2', 'Exp']

for variant_type in all_type:
    variant_mat = []
    for instance_idx in range(all_info.shape[0]):
        feature_mat = all_info[instance_idx][1]
        resource = all_info[instance_idx][2]
        variant_matrix = generate_varient_matrix(feature_mat, resource, type=variant_type)
        variant_mat.append(copy.deepcopy(variant_matrix))
    variant_mat = np.stack(variant_mat)
    np.save(f'../PSPLIB_dataset/variant/variant_120_{variant_type}.npy', variant_mat)

# m = np.load('../PSPLIB_dataset/variant/variant_120_B1.npy', allow_pickle=True)
# min()
# breakpoint()