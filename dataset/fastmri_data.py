import pathlib
import random

import h5py
from torch.utils.data import Dataset
import numpy as np
import torch
class MRI_DATA(Dataset):
    def __init__(self, root, transform, sample_rate=0.5, resolution=320, acquisition=['CORPD_FBK', 'CORPDFS_FBK']):
        self.transform = transform
        self.acquisition = acquisition
        self.recons_key = 'reconstruction_esc'
        self.resolution = resolution

        files = sorted(list(pathlib.Path(root).iterdir()))
        pd_list = []
        pdfs_list = []
        for fname in sorted(files):
            with h5py.File(fname, 'r') as data:
                acq = data.attrs['acquisition'] if 'acquisition' in data.attrs else 'None'
                if acq not in self.acquisition:
                    continue
                if acq == 'CORPD_FBK':
                    pd_list.append(fname)
                elif acq == 'CORPDFS_FBK':
                    pdfs_list.append(fname)
        
        if sample_rate < 1.0:
            sample_num = int(sample_rate*(len(pd_list)+len(pdfs_list)))
            num_files = sample_num//2
            pd_list = pd_list[:min(num_files, len(pd_list))]
            pdfs_list = pdfs_list[:min(num_files, len(pdfs_list))]
        self.files = pd_list + pdfs_list
        
    def __len__(self):
        return len(self.files)
    def pre_process(self, kspace):
        shift_k = np.fft.ifftshift(kspace, (-1, -2))
        image = np.fft.fftshift(np.fft.ifft2(shift_k), (-1, -2))

        r = self.resolution
        if r > 0: 
            w, h = image.shape[-2:]
            image = image[..., (w-r)//2 : (w+r)//2, (h-r)//2 : (h+r)//2]
        kspace = np.fft.fft2(image)
        kspace = np.stack((kspace.real, kspace.imag), axis=-1)
        
        return kspace
        
    def __getitem__(self, i):
        fname = self.files[i]
        with h5py.File(fname, 'r') as f:
            kspace = np.array(f['kspace'])
            acq = str(f.attrs['acquisition'])

        kspace = self.pre_process(kspace)
        data = self.transform(kspace, fname.name)
        finfo = dict(
            fname=fname.name,
            acq=acq
        )
        return data, finfo