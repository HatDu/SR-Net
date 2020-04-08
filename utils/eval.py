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
from eval_2d import load_data


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
        # if fname == 'e14195s3_P03584.7.npy.mat':
        #     continue
        data_path = os.path.join(args.pred_dir, fname)
        recon, target = load_data(data_path, 'normal')
        # recon, target = load_data(data_path, 'TV')
        # recon = recon[:-2]
        rst_dict['NMSE'].append(nmse(target, recon))
        # print(fname, nmse(target, recon))
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
    parser.add_argument('-pred-dir', type=str, default='../data/infer/')
    parser.add_argument('-out-dir', type=str, default='../data/infer/')
    args = parser.parse_args()

    rst_dict = evaluate(args)
    out_path = os.path.join(args.out_dir, 'metricx.csv')
    rst_csv = open(out_path, 'w')
    str_to_write = 'fname, NMSE, PSNR, SSIM \n'
    rst_csv.write(str_to_write)
    # save
    for i, fname in enumerate(rst_dict['nameList']):
        str_to_write = '%s,%f,%f,%f \n' % (
            fname, rst_dict['NMSE'][i], rst_dict['PSNR'][i], rst_dict['SSIM'][i])
        rst_csv.write(str_to_write)
    str_to_write = '%s,%f,%f,%f \n' % (fname, rst_dict['NMSE'].mean(
    ), rst_dict['PSNR'].mean(), rst_dict['SSIM'].mean())
    rst_csv.write(str_to_write)
    rst_csv.close()
