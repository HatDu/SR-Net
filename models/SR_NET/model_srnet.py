import torch
import torch.nn as nn
import numpy as np
from models.SR_NET.dcn.deform_conv import ModulatedDeformConvPack as DCN 
from torch.utils.checkpoint import checkpoint

# time offset fusion
class TOF_CRNNcell(nn.Module):
    def __init__(self, in_chans, out_chans, ksize):
        super().__init__()
        self.conv_offset1 = nn.Sequential(
            nn.Conv2d(in_chans*2, in_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=1e-1, inplace=True)
        )
        self.conv_offset2 = nn.Sequential(
            nn.Conv2d(in_chans*2, in_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=1e-1, inplace=True)
        )
        self.conv_offset3 = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=1e-1, inplace=True)
        )
        self.dcnpack = DCN(in_chans, in_chans, 3, stride=1, padding=1, dilation=1, deformable_groups=1,
                              extra_offset_mask=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, 3, padding=1),
            nn.ReLU()
        )
    def forward(self, x, x_neigh, offset_t):
        offset_1 = self.conv_offset1(torch.cat([x, x_neigh], 1))
        offset_2 = self.conv_offset2(torch.cat([offset_1, offset_t], 1))
        offset = self.conv_offset3(offset_2)
        align_feats = self.dcnpack([x, offset])
        align_feats = self.out_conv(align_feats)
        return align_feats, offset

class TOF_BCRNNlayer(nn.Module):
    def __init__(self, in_chans, out_chans, inter_chans, ksize=3):
        super().__init__()
        self.out_chans = out_chans
        self.inter_chans = inter_chans
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, inter_chans, ksize, padding=ksize//2),
            nn.ReLU(inplace=True)
        )
        self.tof_cell = TOF_CRNNcell(inter_chans, inter_chans, ksize)
        self.out_conv = nn.Sequential(
            nn.Conv2d(inter_chans, out_chans, ksize, padding=ksize//2),
        )
    def forward(self, x_seq):
        nt, nc, nx, ny = x_seq.size()
        size_h = [1, self.inter_chans, nx, ny]
        hidden_offset_init = torch.zeros(size_h).cuda()
        hidden_init = torch.zeros(size_h).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hidden_init
        hidden_offset = hidden_offset_init
        for i in range(nt):
            x_t = self.conv(x_seq[i:i+1])
            align_feats, hidden_offset = self.tof_cell(hidden, x_t, hidden_offset)
            hidden = x_t + align_feats
            output_f.append(self.out_conv(hidden))

        output_f = torch.cat(output_f)

        # backward
        hidden = hidden_init
        hidden_offset = hidden_offset_init
        for i in range(nt):
            x_t = self.conv(x_seq[nt - i - 1: nt - i])
            align_feats, hidden_offset = self.tof_cell(hidden, x_t, hidden_offset)
            hidden = x_t + align_feats
            output_b.append(self.out_conv(hidden))
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b
        
        return output

class ConvBlock(torch.nn.Module):
    def __init__(self, n_convs=5, n_filters=16, in_chans=2, out_chans=2):
        super(ConvBlock, self).__init__()
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.in_chans = in_chans
        self.out_chans = out_chans

        first_conv = nn.Sequential(nn.Conv2d(in_chans, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(1e-2))
        simple_convs = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(1e-2))
            for i in range(n_convs - 2)
        ])
        last_conv = nn.Conv2d(n_filters, out_chans, kernel_size=3, padding=1, bias=True)
        self.overall_convs = nn.Sequential(first_conv, simple_convs, last_conv)

    def forward(self, x):
        y = self.overall_convs(x)
        return y
    
class TOF_NET(nn.Module):
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        super().__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.bcrnn = TOF_BCRNNlayer(n_ch, n_ch, nf, ks)
        self.crnn = ConvBlock(n_convs=nd, n_filters=nf, in_chans=2, out_chans=2)
        self.i = 0
        
    def forward(self, x, k, m, ckpt=True):
        x = x.permute(0, 3, 1, 2)
        def iter_c(x, k, m):
            out = self.bcrnn(x)
            x = out + x

            # dc
            k_n = x.permute(0, 2, 3, 1).fft(2, True) 
            k_n = k + (1-m)*k_n
            x = k_n.ifft(2, True).permute(0, 3, 1, 2)
            np.save('iter_%d.npy'%(self.i+1), x.cpu().numpy())

            out = self.crnn(x)
            x = out + x
            # dc
            k_n = x.permute(0, 2, 3, 1).fft(2, True) 
            k_n = k + (1-m)*k_n
            x = k_n.ifft(2, True).permute(0, 3, 1, 2)
            np.save('iter_%d.npy'%(self.i+2), x.cpu().numpy())
            self.i += 2
            return x
        
        for i in range(self.nc):
            if x.requires_grad == True:
                x = checkpoint(iter_c, x, k, m)
            else:
                x = iter_c(x, k, m)
        x = x.permute(0, 2, 3, 1)
        return x
    