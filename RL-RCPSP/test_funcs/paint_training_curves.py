import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.signal


set_30 = '../behavior_clone_data/set_30.csv'
set_60 = '../behavior_clone_data/set_60.csv'
set_90 = '../behavior_clone_data/set_90.csv'
set_120 = '../behavior_clone_data/set_120.csv'
set_mix = '../behavior_clone_data/set_mix.csv'
set_30 = pd.read_csv(set_30)
set_60 = pd.read_csv(set_60)
set_90 = pd.read_csv(set_90)
set_120 = pd.read_csv(set_120)
set_mix = pd.read_csv(set_mix)

exp_30 = pd.read_csv('../behavior_clone_data/Exp_30.csv')
exp_60 = pd.read_csv('../behavior_clone_data/Exp_60.csv')
exp_90 = pd.read_csv('../behavior_clone_data/Exp_90.csv')
exp_120 = pd.read_csv('../behavior_clone_data/Exp_120.csv')
exp_mix = pd.read_csv('../behavior_clone_data/Exp_mix.csv')

u_30 = pd.read_csv('../behavior_clone_data/u_30.csv')
u_60 = pd.read_csv('../behavior_clone_data/u_60.csv')
u_90 = pd.read_csv('../behavior_clone_data/u_90.csv')
u_120 = pd.read_csv('../behavior_clone_data/u_120.csv')
u_mix = pd.read_csv('../behavior_clone_data/u_mix.csv')

b_30 = pd.read_csv('../behavior_clone_data/b_30.csv')
b_60 = pd.read_csv('../behavior_clone_data/b_60.csv')
b_90 = pd.read_csv('../behavior_clone_data/b_90.csv')
b_120 = pd.read_csv('../behavior_clone_data/b_120.csv')
b_mix = pd.read_csv('../behavior_clone_data/b_mix.csv')






plt.figure(figsize=(8, 6))
plt.plot(b_120.iloc[:, 2], color='red', linewidth=3)

plt.title('instance j120', fontsize=18)
plt.xlabel('every 20 iterations', fontsize=18)
plt.ylabel('average makespan', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()
