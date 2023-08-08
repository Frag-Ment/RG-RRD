import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.signal

# ## action loss of different layers ###
# original_fig = '../behavior_clone_data/original.csv'
# fea_layer_3 = '../behavior_clone_data/fea_layer3_aloss.csv'
# fea_layer_4 = '../behavior_clone_data/fea_layer4_aloss.csv'
# fea_layer_5 = '../behavior_clone_data/fea_layer5_aloss.csv'
# fea_layer_6 = '../behavior_clone_data/fea_layer6_aloss.csv'
#
# df = pd.read_csv(original_fig)
# layer_3 = pd.read_csv(fea_layer_3)
# layer_4 = pd.read_csv(fea_layer_4)
# layer_5 = pd.read_csv(fea_layer_5)
# layer_6 = pd.read_csv(fea_layer_6)
#
# plt.figure()
#
# layer_2 = scipy.signal.savgol_filter(df.iloc[:, 2], 25, 3)
# layer_3 = scipy.signal.savgol_filter(layer_3.iloc[:, 2], 25, 3)
# layer_4 = scipy.signal.savgol_filter(layer_4.iloc[:, 2], 25, 3)
# layer_5 = scipy.signal.savgol_filter(layer_5.iloc[:, 2], 25, 3)
# layer_6 = scipy.signal.savgol_filter(layer_6.iloc[:, 2], 25, 3)
#
# plt.plot()
# plt.plot(layer_2, label='layer 2', color='blue')
# plt.plot(layer_3, label='layer 3', color='orange')
# plt.plot(layer_4, label='layer 4', color='green')
# plt.plot(layer_5, label='layer 5', color='red')
# plt.plot(layer_6, label='layer 6', color='purple')
#
#
# # plt.title('original training curve', fontsize= 16)
# plt.xlabel('every 5 iterations', fontsize=14)
# plt.ylabel('action loss', fontsize=14)
# plt.legend()
# plt.show()


### action loss of different gnn layers ###
# layer_2 = '../behavior_clone_data/original.csv'
# layer_3 = '../behavior_clone_data/gnn_layer3_aloss.csv'
# layer_4 = '../behavior_clone_data/gnn_layer4_aloss.csv'
# layer_5 = '../behavior_clone_data/gnn_layer5_aloss.csv'
# layer_6 = '../behavior_clone_data/gnn_layer6_aloss.csv'
#
# layer_2 = pd.read_csv(layer_2)
# layer_3 = pd.read_csv(layer_3)
# layer_4 = pd.read_csv(layer_4)
# layer_5 = pd.read_csv(layer_5)
# layer_6 = pd.read_csv(layer_6)
#
# plt.figure()
#
# layer_2 = scipy.signal.savgol_filter(layer_2.iloc[:, 2], 25, 3)
# layer_3 = scipy.signal.savgol_filter(layer_3.iloc[:, 2], 25, 3)
# layer_4 = scipy.signal.savgol_filter(layer_4.iloc[:, 2], 25, 3)
# layer_5 = scipy.signal.savgol_filter(layer_5.iloc[:, 2], 25, 3)
# layer_6 = scipy.signal.savgol_filter(layer_6.iloc[:, 2], 25, 3)
#
# plt.plot()
# plt.plot(layer_2, label='layer 2', color='blue')
# plt.plot(layer_3, label='layer 3', color='orange')
# plt.plot(layer_4, label='layer 4', color='green')
# plt.plot(layer_5, label='layer 5', color='red')
# plt.plot(layer_6, label='layer 6', color='purple')
#
# # plt.title('original training curve', fontsize= 18)
# plt.xlabel('every 5 iterations', fontsize=14)
# plt.ylabel('action loss', fontsize=14)
# plt.legend()
# plt.show()


# ## action loss of comibined params ###
# original_fig = '../behavior_clone_data/g3_5_aloss.csv'
# fea_layer_3 = '../behavior_clone_data/g4_4_aloss.csv'
# fea_layer_4 = '../behavior_clone_data/g4_5_aloss.csv'
# fea_layer_5 = '../behavior_clone_data/g5_5_aloss.csv'
#
#
# df = pd.read_csv(original_fig)
# layer_3 = pd.read_csv(fea_layer_3)
# layer_4 = pd.read_csv(fea_layer_4)
# layer_5 = pd.read_csv(fea_layer_5)
#
# plt.figure()
#
# layer_2 = scipy.signal.savgol_filter(df.iloc[:, 2], 25, 3)
# layer_3 = scipy.signal.savgol_filter(layer_3.iloc[:, 2], 25, 3)
# layer_4 = scipy.signal.savgol_filter(layer_4.iloc[:, 2], 25, 3)
# layer_5 = scipy.signal.savgol_filter(layer_5.iloc[:, 2], 25, 3)
#
# plt.plot()
# plt.plot(layer_2, label='gnn 3, gnn hidden 5', color='blue')
# plt.plot(layer_3, label='gnn 4, gnn hidden 4', color='orange')
# plt.plot(layer_4, label='gnn 4, gnn hidden 5', color='green')
# plt.plot(layer_5, label='gnn 5, gnn hidden 5', color='red')
#
# # plt.title('original training curve', fontsize= 16)
# plt.xlabel('every 5 iterations', fontsize=14)
# plt.ylabel('action loss', fontsize=14)
# plt.legend()
# plt.show()
#
# ## critic loss of comibined params ###
# original_fig = '../behavior_clone_data/g3_5_vloss.csv'
# fea_layer_3 = '../behavior_clone_data/g4_4_vloss.csv'
# fea_layer_4 = '../behavior_clone_data/g5_5_vloss.csv'
# fea_layer_5 = '../behavior_clone_data/g4_5_vloss.csv'
#
# df = pd.read_csv(original_fig)
# layer_3 = pd.read_csv(fea_layer_3)
# layer_4 = pd.read_csv(fea_layer_4)
# layer_5 = pd.read_csv(fea_layer_5)
#
# plt.figure()
#
# layer_2 = scipy.signal.savgol_filter(df.iloc[:, 2], 25, 3)
# layer_3 = scipy.signal.savgol_filter(layer_3.iloc[:, 2], 25, 3)
# layer_4 = scipy.signal.savgol_filter(layer_4.iloc[:, 2], 25, 3)
# layer_5 = scipy.signal.savgol_filter(layer_5.iloc[:, 2], 25, 3)
#
# plt.plot()
# plt.plot(layer_2, label='gnn 3, gnn hidden 5', color='blue')
# plt.plot(layer_3, label='gnn 4, gnn hidden 4', color='orange')
# plt.plot(layer_4, label='gnn 4, gnn hidden 5', color='green')
# plt.plot(layer_5, label='gnn 5, gnn hidden 5', color='red')
#
# # plt.title('original training curve', fontsize= 16)
# plt.xlabel('every 5 iterations', fontsize=14)
# plt.ylabel('value loss', fontsize=14)
# plt.legend()
# plt.show()



## repeat times  ###
original_fig = '../behavior_clone_data/repeat_50.csv'
fea_layer_3 = '../behavior_clone_data/random.csv'
fea_layer_4 = '../behavior_clone_data/once.csv'


layer_2 = pd.read_csv(original_fig)
layer_3 = pd.read_csv(fea_layer_3)
layer_4 = pd.read_csv(fea_layer_4)


plt.figure()

# layer_2 = scipy.signal.savgol_filter(layer_2.iloc[:, 2], 25, 3)
# layer_3 = scipy.signal.savgol_filter(layer_3.iloc[:, 2],  25, 3)
# layer_4 = scipy.signal.savgol_filter(layer_4.iloc[:, 2], 25, 3)


plt.plot()



plt.plot(layer_3.iloc[:, 1], layer_3.iloc[:, 2], label='method 1', color='orange')
plt.plot(layer_4.iloc[:, 1], layer_4.iloc[:, 2], label='method 2', color='green')
plt.plot(layer_2.iloc[:, 1], layer_2.iloc[:, 2], label='method 3', color='blue')

# plt.title('original training curve', fontsize= 16)
plt.xlabel('iterations', fontsize=14)
plt.ylabel('average makespan', fontsize=14)
plt.legend()
plt.show()