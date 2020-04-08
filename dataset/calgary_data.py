import os

import numpy as np
from torch.utils.data import Dataset


class MRI_DATA(Dataset):
    def __init__(self, root, transform=None, train=True, gap=1):
        super().__init__()
        self.root = root
        self.data_list = os.listdir(root)
        self.transform = transform
        self.train = train
        self.gap = gap

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        kspace = np.load(os.path.join(self.root, self.data_list[i]))
        info = {'fname': self.data_list[i]}
        if self.train != True:
            hs = 16*self.gap
            size = kspace.shape[0]//2
            kspace = kspace[max(size-hs, 0):min(size+hs, kspace.shape[0])]
            # print(kspace.shape)
            
        data = self.transform(kspace, self.data_list[i]) if self.transform else kspace
        
        return data, info
