import numpy as np
import scipy


# 生成扰动特征矩阵，在使用时取出扰动矩阵对应t时刻扰动，和原资源数量相加即可得到变动后资源数量
# type:uniform/exponential/beta

def generate_variant_matrix(type, n_resource, low_bound, high_bound,
                            variance=None, random_seed=None, a=None, b=None, scale=None):
    if random_seed:
        np.random.seed(random_seed)
    # 均匀分布
    if type == 'uniform':
        resource_variant = np.random.randint(low_bound, high_bound, size=(1000, n_resource))

    # 正态分布
    if type == 'normal':
        resource_variant = np.random.normal(loc=(low_bound + high_bound) * 0.5, scale=variance, size=(1000, n_resource))

    # beta分布
    if type == 'beta':
        resource_variant = np.random.beta(a, b, size=(1000, n_resource)).round()

    # 指数分布
    if type == 'exponetial':
        resource_variant = np.random.exponential(scale, size=(1000, n_resource))

    return resource_variant

path = 'C:/Users/Fragment/Desktop/benchmark/resource_variant1.npy'
all_variant_mat = []
for i in range(1000):
    all_variant_mat.append(generate_variant_matrix('uniform', 4, -3, 3, random_seed=1))

np.save(path, all_variant_mat)