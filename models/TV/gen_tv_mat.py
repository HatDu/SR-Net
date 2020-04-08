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
import scipy.io as io
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-out-dir', type=str, default='../data/infer/')
    parser.add_argument('-data-path', type=str, default='../data/CC359/')
    parser.add_argument('-mask-style', default='cartesian_1d', type=str)
    parser.add_argument('-gap', default=1, type=int, help='sequence sample interval')
    parser.add_argument('-same', action='store_true', help='all mask will be same in a sequence')
    parser.add_argument('-cf', default=0.08, type=float, help='center fraction')
    parser.add_argument('-acc', default=4, type=int, help='accerlation')
    
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
    for batch in tqdm(data_loader):
        data, finfo = batch
        im_und, k_und, image, mask, kspace, pdf = data
        fname = finfo['fname'][0]
        output = im_und
        out_path = os.path.join(out_dir, fname)
        # process mask
        mask = mask.squeeze(0).squeeze(-1).numpy()
        shape = list(mask.shape)
        shape[-2] = shape[-1]
        mask_2d = np.ones(shape)
        mask_2d = mask_2d*mask
        mask_2d = np.fft.fftshift(mask_2d, (-1, -2))
        mask_2d = mask_2d.astype('bool')

        # process image
        image = image.squeeze(0).squeeze(-1).numpy()
        image = image.astype('float64')
        image_complex = np.ones(image.shape[:-1], dtype='complex128')
        image_complex.real = image[...,0]
        image_complex.imag = image[...,1]

        # pdf
        pdf = pdf.squeeze(0).numpy()
        io.savemat(os.path.join(out_dir, '%s.mat'%fname), {'data':image_complex, 'mask':mask_2d, 'pdf': pdf})



if __name__ == "__main__":
    args = parse_args()
    run(args)
