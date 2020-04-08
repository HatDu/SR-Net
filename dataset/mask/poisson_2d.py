import numpy as np
import torch
from numpy.lib.stride_tricks import as_strided
import sigpy.mri as mri


class MaskFunc:
    def __init__(self, cf, acc, same=False):
        self.acc = acc
        self.cf = cf
        self.same = same
        self.k = 10
        self.crop_corner = True
        self.rng = np.random.RandomState()

    def __call__(self, shape, seed=None, centred=False):
        '''
        shape: (t,w,h)
        '''
        self.rng.seed(seed)
        mask = np.zeros(shape)
        if self.same:
            sed = self.rng.rand()
            mask[:] = mri.poisson(shape[1:], accel=self.acc, K=self.k,
                                  dtype=float, crop_corner=self.crop_corner, seed=seed)
        else:
            for i in range(len(mask)):
                sed = self.rng.rand()
                mask[i] = mri.poisson(shape[1:], accel=self.acc, K=self.k, dtype=float, crop_corner=True, seed=None)
                # print(mask[i].mean())
        
        c = shape[-1]//2
        cl = int(np.sqrt(shape[-1]**2 * self.cf))//2
        mask[:, c-cl:c+cl, c-cl:c+cl] = 1.
        # print(mask.mean())
        if not centred:
            mask = np.fft.ifftshift(mask, axes=(-1, -2))

        pdf = np.sum(mask, axis=0)
        pdf = pdf/np.max(pdf)
        return [mask, pdf]
