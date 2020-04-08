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
        gt.transpose(1, 2, 0), pred.transpose(1, 2, 0), multichannel=True, data_range=gt.max()
    )


def evaluate(args):
    rst_dict = dict(NMSE=[], PSNR=[], SSIM=[], nameList=[])
    for fname in tqdm(sorted(os.listdir(args.pred_dir))):
        data_path = os.path.join(args.pred_dir, fname)
        data = io.loadmat(data_path)
        recon = data['y_hat_k']
        # recon = data['y_tilde_k']
        target = data['y']

        rst_dict['NMSE'].append(nmse(target, recon))
        rst_dict['PSNR'].append(psnr(target, recon))
        rst_dict['SSIM'].append(ssim(target, recon))
        rst_dict['nameList'].append(fname)

    str1 = '||'
    for key in rst_dict.keys():
        rst_dict[key] = np.array(rst_dict[key])
        str1 += '%s|' % key
    str1 += '\n|:-:|:-:|:-:|:-:|'
    str1 += '\n|mean|%.6f|%.4f|%.4f|' % (rst_dict['NMSE'].mean(),
                                     rst_dict['PSNR'].mean(), rst_dict['SSIM'].mean())
    str1 += '\n|std|%.6f|%.4f|%.4f|' % (rst_dict['NMSE'].std(),
                                     rst_dict['PSNR'].std(), rst_dict['SSIM'].std())
    print(str1)
    return rst_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='../data/infer/')
    parser.add_argument('--out-path', type=str, default='./rst.csv')
    args = parser.parse_args()

    rst_dict = evaluate(args)
    rst_csv_path = os.path.join(args.out_path, 'result.csv')
    rst_csv = open(rst_csv_path, 'w')
    str_to_write='fname, NMSE, PSNR, SSIM \n'
    rst_csv.write(str_to_write)
    # save
    for i, fname in enumerate(rst_dict['nameList']):
        str_to_write = '%s,%f,%f,%f \n'%(fname, rst_dict['NMSE'][i], rst_dict['PSNR'][i], rst_dict['SSIM'][i])
        rst_csv.write(str_to_write)
    
    rst_csv.close()
# python .\utils\eval_mat.py --pred-dir C:\Users\dnm\Desktop\images\BM4D\Recon\Cartesian_X4\ --out-path C:\Users\dnm\Desktop\images\BM4D\Recon\Cartesian_X4\
