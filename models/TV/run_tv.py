# from dataset.calgary_data import MRI_DATA
from dataset.fastmri_data import MRI_DATA
from dataset.transform import DataTransform
from dataset.mask import get_mask_func
from torch.utils.data import DataLoader
from tqdm import tqdm
import h5py
import os
import argparse
from dataset import create_test_loader
import sigpy.mri.app as app
import numpy as np
import torch
import cupy as cp
def TV(undk, pdf, reg=0.01, device=0):
    '''
        undk: undersampled and uncentered kspace data, shape=(1, x, y, 2) 
    '''
    undk_unc = undk.squeeze(0).numpy()
    undk = np.zeros(undk_unc.shape[:-1], dtype='complex64')
    undk.real = undk_unc[..., 0]
    undk.imag = undk_unc[..., 1]
    undk = undk[:,np.newaxis, ...]
    sens_map = np.ones(undk.shape[-3:], dtype='complex64')
    recons = []
    # for i in range(83, 86):
    for i in range(len(undk)):
        s = undk[i]
        print('Slice %d'%i)
        # print(s.shape)
        p = pdf
        TV = app.TotalVariationRecon(s, sens_map, reg, device=1, max_iter=1000, max_power_iter=30, max_cg_iter=10)
        recon = TV.run()
        recons.append(np.fft.ifftshift(cp.asnumpy(recon), (-1, -2)))
    recons = np.array(recons)
    recons_t = torch.zeros((*recons.shape, 2))
    recons_t[..., 0] = torch.tensor(recons.real)
    recons_t[..., 1] = torch.tensor(recons.imag)

    return recons_t

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out-dir', type=str, default='../data/infer/')
    parser.add_argument('-data-path', type=str, default='../data/CC359/')
    parser.add_argument('-mask-style', default='cartesian_1d', type=str)
    parser.add_argument('-gap', default=1, type=int, help='sequence sample interval')
    parser.add_argument('-same', action='store_true', help='all mask will be same in a sequence')
    parser.add_argument('-cf', default=0.08, type=float, help='center fraction')
    parser.add_argument('-acc', default=4, type=float, help='accerlation')
    
    parser.add_argument('-dset', default='calgary')
    # for fastmri dataset
    parser.add_argument('-resolution', default=320, type=int)
    parser.add_argument('-acquisition', nargs='+', default=['CORPD_FBK', 'CORPDFS_FBK'])
    
    args = parser.parse_args()
    return args

def run(args):
    data_loader = create_test_loader(args)
    out_dir = args.out_dir
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    for i, batch in enumerate(data_loader):
        data, finfo = batch
        im_und, k_und, image, mask, kspace, pdf = data
        print('run %s'%finfo['fname'])
        # if i<5 : continue
        fname = finfo['fname'][0]
        output = TV(k_und, pdf)
        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('recon', data=output) 
            f.create_dataset('target', data=image[0]) 


if __name__ == "__main__":
    args = parse_args()
    run(args)
    