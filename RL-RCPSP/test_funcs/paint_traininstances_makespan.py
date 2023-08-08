import numpy as np
import matplotlib.pyplot as plt


## running times ###

labels = ['60', '90', '120', '300']
x = ('60', '90', '120', '300')
y30 = [59.92, 57.84,  57.61, 57.58]
y60 = [84.21, 79.37, 78.71, 78.65]
y90 = [104.21, 98.75, 97.77, 97.63]


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


# plt.ylabel('average makespan on validation dataset', fontsize=14)
ax.set_ylabel('best average makespan on validation dataset', fontsize=12)
ax.set_xlabel('number of instances used for training', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)

plt.ylim(50, 110)

plt.legend()
plt.show()
