import torch
import h5py
from tqdm import tqdm
import os
import numpy as np
def run_net(args, model, data_loader):
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    param_num = sum(param.numel() for param in model.parameters())
    # print(str(model))
    print('model params num %d'%param_num)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            data, f_info = batch
            im_und, k_und, image, mask, kspace, pdf = data
            outs = torch.ones_like(image)
            im_und = im_und.squeeze(0)
            k_und = k_und.squeeze(0)
            mask = mask.squeeze(0)
            image = image.squeeze(0)
            length = im_und.size(0)

            # ol = 0 # overlap
            # out1 = model(im_und[0: 64+ol].to(args.device), k_und[0: 64+ol].to(args.device), mask[0: 64+ol].to(args.device))
            # out2 = model(im_und[64-ol: 128+ol].to(args.device), k_und[64-ol: 128+ol].to(args.device), mask[64-ol: 128+ol].to(args.device))
            # out3 = model(im_und[128-ol: 170].to(args.device), k_und[128-ol: 170].to(args.device), mask[128-ol: 170].to(args.device))
            # # output = torch.cat((out1[:64], out2[ol:-ol], out3[ol:]))
            # output = torch.cat((out1, out2, out3))
            # output = output.cpu()
            
            start = 0
            # step = 32
            step = 30
            # step = 16
            ol = 3 # overlap

            outs = []
            leg = len(image)
            for start in range(0, leg, step):
                s = max(start-ol, 0)
                e = min(start+step+ol, leg)
                im = im_und[s:e].to(args.device)
                ku = k_und[s:e].to(args.device)
                ms = mask[s:e].to(args.device)
                ou = model(im, ku, ms).cpu()
                if s==0:
                    ou = ou[s:-ol]
                elif e==leg:
                    ou = ou[ol: ]
                else:
                    ou = ou[ol: -ol]
                # print(ou.size())
                outs.append(ou)
            output = torch.cat(outs)
            
            fname = f_info['fname'][0]
            with h5py.File(os.path.join(args.out_dir, fname), 'w') as f:
                f.create_dataset('recon', data=output) 
                f.create_dataset('target', data=image) 

            # break
    print('Done!')