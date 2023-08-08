import numpy as np

problems = []

for a in range(1, 49):
    for b in range(1, 11):
        location = '../PSPLIB_dataset/j60.sm/j60{}_{}.sm'.format(a, b)
        f = open(location, 'r')
        data = f.readlines()

        information = []
        adj_mat = np.zeros(3844).reshape(62, 62)
        resource_mat = np.zeros(434).reshape(62, 7)
        resource_capacity_mat = np.zeros(4)
        # 读取邻接矩阵 #
        for line_num in range(18, 80):
            b = data[line_num].split()
            # if b[0] is not str:
            #     print('error:read wrong line', b[0])
            length = len(b)
            idx = int(b[0]) - 1

            for i in range(3, length):
                successor = int(b[i]) - 1
                adj_mat[idx][successor] = 1

        adj_mat = adj_mat + np.eye(adj_mat.shape[0])

        # 读取时间及资源矩阵 格式为：时间 4种资源
        # 然后在每行的最后加入一个0，表示活动状态为未执行
        for line_num in range(84, 146):
            b = data[line_num].split()
            idx = int(b[0]) - 1

            for i in range(2, 7):
                resource = int(b[i])
                resource_mat[idx][i-2] = resource
        # 对于第一个活动把状态设为01，表示已经执行完毕
        resource_mat[0][-1] = 1

        # 读取资源容量矩阵 #
        b = data[149].split()
        for i in range(0, 4):
            resource_capacity = int(b[i])
            resource_capacity_mat[i] = resource_capacity
        #
        information.append(adj_mat)
        information.append(resource_mat)
        information.append(resource_capacity_mat)

        problems.append(information)


pro_ary = np.array(problems, dtype=object)

# 保存文件
path = '../PSPLIB_dataset/problems_60.npy'

# file = open(path, 'w')
# file.write(pro_ary)
# file.close

np.save(path, pro_ary)
# b = np.load(path, allow_pickle=True)
# print(b[1][0][0])