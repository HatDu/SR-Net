"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from tqdm import tqdm
from argparse import ArgumentParser

import h5py
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from utils.show_recon import save_recon_example
import scipy.io as io

def nmse(gt, pred):
    """ Compute Normalized Mean Squared Error (NMSE) """
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def psnr(gt, pred):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return compare_psnr(gt, pred, data_range=gt.max())


def ssim(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    return compare_ssim(
        gt, pred, data_range=gt.max()
    )

def load_data(data_path, k='normal'):
    if k == 'normal':
        with h5py.File(data_path, 'r') as reconstructions:
            recon = reconstructions['recon'][()]
            target = reconstructions['target'][()]
            assert recon.shape[-1] == 2 and target.shape[-1] == 2

            targets = np.sqrt((target**2).sum(-1))
            recons = np.sqrt((recon**2).sum(-1))
        
    elif k=='TV':
        data = io.loadmat(data_path)
        recon = data['recon']
        target = data['images']
        time = data['recon_time']
        # print(time)
        targets = np.abs(target)
        recons = np.abs(recon)
        recons = np.transpose(recons, (2, 0, 1))
        targets = np.transpose(targets, (2, 0, 1))
    elif k=='BM4D':
        data = io.loadmat(data_path)
        recons = data['y_tilde_k']
        targets = data['y']
        recons = np.transpose(recons, (2, 0, 1))
        targets = np.transpose(targets, (2, 0, 1))
        
    return recons, targets


def evaluate(args):
    rst_dict = dict(NMSE=[], PSNR=[], SSIM=[], nameList=[])
    for fname in tqdm(sorted(os.listdir(args.pred_dir))):
        data_path = os.path.join(args.pred_dir, fname)
        with h5py.File(data_path, 'r') as reconstructions:
            recon = reconstructions['recon'][()]
            target = reconstructions['target'][()]
            assert recon.shape[-1] == 2 and target.shape[-1] == 2

            targets = np.sqrt((target**2).sum(-1))
            recons = np.sqrt((recon**2).sum(-1))
        # recons, targets = load_data(data_path, 'TV')
        # recons, targets = load_data(data_path, 'BM4D')
        nmse_list = []
        psnr_list = []
        ssim_list = []
        for target, recon in zip(targets, recons):
            
            nmse_list.append(nmse(target, recon))
            psnr_list.append(psnr(target, recon))
            ssim_list.append(ssim(target, recon))

        rst_dict['NMSE'].append(nmse_list)
        rst_dict['PSNR'].append(psnr_list)
        rst_dict['SSIM'].append(ssim_list)
        rst_dict['nameList'].append(fname)

    for key in rst_dict.keys():
        if key == 'nameList':
            continue
        data_list = rst_dict[key]
        rst_csv = open(os.path.join(args.out_path, '%s.csv'%key), 'w')
        str_to_write = ''
        for name in rst_dict['nameList']:
            str_to_write += '%s,'%name
        rst_csv.write(str_to_write+'\n')
        for row in range(len(data_list[0])):
            str_to_write = ''
            for col in range(len(data_list)):
                str_to_write += '%f,'%data_list[col][row]
            rst_csv.write(str_to_write+'\n')
        rst_csv.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='../data/infer/')
    parser.add_argument('--out-path', type=str, default='./')
    args = parser.parse_args()

    save_recon_example(args.pred_dir, args.out_path)
    evaluate(args)
