import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
from scipy import io
def save_with_colorbar(image, path):
    plt.close()
    plt.clf()
    plt.figure()
    plt.axis('off')
    ax = plt.gca()
    im = ax.imshow(image)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)
    # , orientation='vertical'
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal') 
    cbar.ax.tick_params(labelsize=16)
    plt.savefig(path)
def load_TV_mat(data_path):
    data = io.loadmat(data_path)
    recon_ = data['recon']
    target_ = data['images']
    recon_ = np.transpose(recon_, (2, 0, 1))
    target_ = np.transpose(target_, (2, 0, 1))

    shape = list(target_.shape)
    shape.append(2)
    recon = np.zeros(shape)
    target = np.zeros(shape)
    recon[..., 0] = recon_.real
    recon[..., 1] = recon_.imag
    target[..., 0] = target_.real
    target[..., 1] = target_.imag
    return target, recon

def load_BM3D_mat(data_path):
    data = io.loadmat(data_path)
    recon_ = data['y_tilde_k']
    target_ = data['y']
    recon_ = np.transpose(recon_, (2, 0, 1))
    target_ = np.transpose(target_, (2, 0, 1))
    shape = list(recon_.shape)
    shape.append(2)
    recons = np.zeros(shape)
    recons[..., 0] = recon_
    targets = np.zeros(shape)
    targets[..., 0] = target_
    return targets, recons


def save_recon_example(infer_dir, out_dir):
    file_list = sorted(os.listdir(infer_dir))
    ref_sample = file_list[0]
    fname = ref_sample.split('.')[0]
    ## make dir
    out_dir = os.path.join(out_dir, fname)
        
    
    ref_path = os.path.join(infer_dir, file_list[0])
    out_recon = os.path.join(out_dir, 'recon')
    out_target = os.path.join(out_dir, 'target')
    out_err = os.path.join(out_dir, 'err')

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        os.mkdir(out_recon)
        os.mkdir(out_target)
        os.mkdir(out_err)
    # open data
    with h5py.File(ref_path, 'r') as reconstructions:
        recon = np.array(reconstructions['recon'][()])
        target = np.array(reconstructions['target'][()])
        ## k_und = np.array(reconstructions['k_und'][()])
    ###### TV ####
    # target, recon = load_TV_mat(ref_path)
    ###### BM3D ####
    # target, recon = load_BM3D_mat(ref_path)

    err = target - recon
    print(err.shape)
    recon_img = np.sqrt((recon**2).sum(-1))
    target_img = np.sqrt((target**2).sum(-1))
    # err_img = np.sqrt((err**2).sum(-1))
    err_img = abs(target_img-recon_img)
    print(err_img.shape)
    # k_und = np.sqrt((k_und**2).sum(-1))
    # k_und = np.log(k_und + 1e-9)
    # k_und = np.fft.fftshift(k_und, (-1, -2))

    # save slices
    start = 60
    step = 1
    show_num = 20

    plt.figure(figsize=(18, 10))
    # plt.style.use('dark_background')
    for i in tqdm(range(show_num)):
        slice_no = start + step*i
        save_with_colorbar(target_img[slice_no], os.path.join(out_target, 'slice_%d.png'%slice_no))
        plt.imsave(os.path.join(out_target, 'no_slice_%d.png'%slice_no), target_img[slice_no])
        save_with_colorbar(recon_img[slice_no], os.path.join(out_recon, 'slice_%d.png'%slice_no))
        plt.imsave(os.path.join(out_recon, 'no_slice_%d.png'%slice_no), recon_img[slice_no])
        save_with_colorbar(err_img[slice_no], os.path.join(out_err, 'slice_%d.png'%slice_no))
        # save_with_colorbar(k_und[slice_no], os.path.join(out_err, 'undk_%d.png'%slice_no))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-infer-dir', type=str, default='../data/infer/')
    parser.add_argument('-out-dir', type=str, default='../data/infer/')
    args = parser.parse_args()
    save_recon_example(args.infer_dir, args.out_dir)
    pass