import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import scipy.signal

## sample times ###

labels = ['greedy', '10', '20', '30', '40', '50', '60']
y30 = [59.28, 57.23, 56.97, 56.88, 56.88, 56.88, 56.88]
y60 = (81.57, 79.46, 78.78, 78.36, 78.26, 78.26, 78.20)
y90 = (95.83, 94.55, 93.55, 93.05, 92.90, 92.85, 92.83)

plt.figure()

total_width = 0.8  # the width of the bars

x = np.arange(len(labels))

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2


fig, ax = plt.subplots()
rects0 = ax.bar(x -width, y30, width, label='j30')
rects1 = ax.bar(x , y60, width, label='j60')
rects2 = ax.bar(x + width, y90, width, label='j90')


plt.ylim(55, 100)
plt.ylabel('average makespan', fontsize=14)

ax.set_ylabel('average makespan', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.legend()
plt.show()



## running times ###

x = ('greedy', '10', '20', '30', '40', '50', '60')
y30 = [0.0627, 0.2766, 0.5238, 0.7201, 0.9401, 1.1575, 1.3938]
y60 = [0.1041, 0.5732, 1.0989, 1.6225, 2.1797, 2.6998, 3.1089]
y90 = [0.1683, 0.9249, 1.7769, 2.6277, 3.4841, 4.4243, 5.2254]


plt.figure()

total_width = 0.8  # the width of the bars

x = np.arange(len(labels))

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2


fig, ax = plt.subplots()
rects0 = ax.bar(x -width, y30, width, label='j30')
rects1 = ax.bar(x , y60, width, label='j60')
rects2 = ax.bar(x + width, y90, width, label='j90')


# plt.ylim(55, 100)
plt.ylabel('average time(s)', fontsize=14)

ax.set_ylabel('average time', fontsize=14)

ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.legend()
plt.show()
