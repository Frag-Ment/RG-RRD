import numpy as np

# location = '../PSPLIB_dataset/j30opt.sm'
# start_row = 22
# end_row = 502
# opt = []

# location = '../PSPLIB_dataset/j60lb.sm'
# start_row = 26
# end_row = 435
# opt = []

location = '../PSPLIB_dataset/j90lb.sm'
start_row = 25
end_row = 470
opt = []

# location = '../PSPLIB_dataset/j120lb.sm'
# start_row = 11
# end_row = 195
# opt = []

f = open(location, 'r')
data = f.readlines()

for i in range(start_row, end_row):
    # if read j30opt, column is 2
    opt.append(int(data[i].split()[3]))

print(opt)
print(len(opt))

np.save('../PSPLIB_dataset/opt_90.npy', opt)


