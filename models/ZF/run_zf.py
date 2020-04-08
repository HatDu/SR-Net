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
    for batch in tqdm(data_loader):
        data, finfo = batch
        im_und, k_und, image, mask, kspace, pdf = data
        fname = finfo['fname'][0]
        output = im_und
        out_path = os.path.join(out_dir, fname)
        with h5py.File(out_path, 'w') as f:
            f.create_dataset('recon', data=output[0]) 
            f.create_dataset('target', data=image[0]) 
            f.create_dataset('k_und', data=k_und[0])


if __name__ == "__main__":
    args = parse_args()
    run(args)
    