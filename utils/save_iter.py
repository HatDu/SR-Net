import numpy as np
import matplotlib.pyplot as plt
import os

root_dir = 'exp_dir/TOF_NET/ITER/6XXXX/data/'
sv_dir = 'exp_dir/TOF_NET/ITER/6XXXX/imgs/'

data_list = {}
for fname in sorted(os.listdir(root_dir)):
    key = fname.replace('npy', 'png')
    data_list[key] = np.load(os.path.join(root_dir, fname))
print(data_list.keys())

for k in data_list.keys():
    data = data_list[k]
    print(data.shape)
    data = data[20]
    data = np.transpose(data, (1, 2, 0))
    data = np.sqrt((data**2).sum(-1))
    plt.imsave(os.path.join(sv_dir, k), data, cmap='gray')