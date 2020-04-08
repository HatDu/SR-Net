import torch
import numpy as np


class DataTransform:
    def __init__(self, mask_func, use_seed=False, max_frames=-1, gap=1):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.max_frames = max_frames
        self.gap = gap

    def __call__(self, kspace, fname, train=True):
        '''
        params:
            kspace: ndarray, float32, (n, x, y, 2)  
        returns:
            im_und(...): n, x, y, 2
        '''
        # for different thickness
        start = 0
        if train:
            start = np.random.randint(0, self.gap)
        slices_mask = [i for i in range(start, kspace.shape[0], self.gap)]
        kspace = kspace[slices_mask]
        kspace = torch.tensor(kspace).float()
        kspace = (kspace - kspace.mean()) / kspace.std()

        # normalize
        if self.max_frames>0:
            n_seq = kspace.size(0)
            if n_seq > self.max_frames:
                start = np.random.randint(n_seq-self.max_frames)
                kspace = kspace[start: start + self.max_frames, ...]
        
        image = kspace.ifft(2, True)

        # undersample

        seed = None if not self.use_seed else tuple(map(ord, fname))
        [mask, pdf] = self.mask_func(np.array(image.size()[:-1]), seed)
        mask = torch.tensor(mask).float().unsqueeze(-1)

        k_und = kspace*mask
        im_und = k_und.ifft(2, True)

        return im_und, k_und, image, mask, kspace, pdf