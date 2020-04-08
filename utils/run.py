import torch
import h5py
from tqdm import tqdm
import os
import numpy as np
def run_net(args, model, data_loader):
    param_num = sum(param.numel() for param in model.parameters())
    # print(str(model))
    print('model params num %d'%param_num)
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            data, f_info = batch
            im_und, k_und, image, mask, kspace, pdf = data

            im_und = im_und.to(args.device).squeeze(0)
            k_und = k_und.to(args.device).squeeze(0)
            mask = mask.to(args.device).squeeze(0)
            image = image.squeeze(0)

            output = model(im_und, k_und, mask)
            output = output.cpu()
            fname = f_info['fname'][0]
            # np.save('iter_zf.npy', im_und.permute(0, 3, 1, 2).cpu().numpy())
            # np.save('iter_gt.npy', image.permute(0, 3, 1, 2).numpy())
            with h5py.File(os.path.join(args.out_dir, fname), 'w') as f:
                f.create_dataset('recon', data=output) 
                f.create_dataset('target', data=image) 

            # break
    print('Done!')