import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pydot

all_info = np.load('./PSPLIB_dataset/problems_30.npy', allow_pickle=True)

instance_idx = 0
adj_mat = all_info[instance_idx][0] - np.eye(all_info[instance_idx][0].shape[0])
nodes_information = all_info[instance_idx][1]
resource = all_info[instance_idx][2]
resource_variant = [resource for item in range(1000)]

walltime = 1

def condition_visualize(adj_mat, nodes_information, resource, walltime):
    color_list = []
    # 设置颜色
    for row in nodes_information:
        # 未执行
        if row[5] == 0 and row[6] == 0:
            color_list.append('red')
        # 执行中
        if row[5] == 1 and row[6] == 0:
            color_list.append('blue')
        # 已完成
        if row[5] == 0 and row[6] == 1:
            color_list.append('green')

    # 设置显示信息
    labels = {}
    for i, row in enumerate(nodes_information):
        labels[i] = '{}'.format(i) + str(row[:5])

    G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph)
    pos = nx.nx_pydot.pydot_layout(G, prog='dot')

    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_color=color_list)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)

    time = walltime
    resource = 'Resource: ' + str(resource)
    walltime = 'Walltime: ' + str(walltime)
    plt.text(0, 40, resource, color='red', fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='bottom')
    plt.text(0, 0, walltime, color='red', fontsize=10, fontweight='bold', horizontalalignment='left', verticalalignment='bottom')
    plt.show()



