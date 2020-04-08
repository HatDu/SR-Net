import torch
import torch.nn as nn
import numpy as np

def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)


def conv_block(n_ch, nd, nf=32, ks=3, dilation=1, bn=False, nl='lrelu', conv_dim=2, n_out=None):

    # convolution dimension (2D or 3D)
    if conv_dim == 2:
        conv = nn.Conv2d
    else:
        conv = nn.Conv3d

    # output dim: If None, it is assumed to be the same as n_ch
    if not n_out:
        n_out = n_ch

    # dilated convolution
    pad_conv = 1
    if dilation > 1:
        # in = floor(in + 2*pad - dilation * (ks-1) - 1)/stride + 1)
        # pad = dilation
        pad_dilconv = dilation
    else:
        pad_dilconv = pad_conv

    def conv_i():
        return conv(nf,   nf, ks, stride=1, padding=pad_dilconv, dilation=dilation, bias=True)

    conv_1 = conv(n_ch, nf, ks, stride=1, padding=pad_conv, bias=True)
    conv_n = conv(nf, n_out, ks, stride=1, padding=pad_conv, bias=True)

    # relu
    nll = relu if nl == 'relu' else lrelu

    layers = [conv_1, nll()]
    for i in range(nd-2):
        if bn:
            layers.append(nn.BatchNorm2d(nf))
        layers += [conv_i(), nll()]

    layers += [conv_n]

    return nn.Sequential(*layers)

class DnCn3D(nn.Module):
    def __init__(self, nc=5, nd=5, nf=32):
        super(DnCn3D, self).__init__()
        self.nc = nc
        self.nd = nd
        print('Creating D{}C{} (3D)'.format(nd, nc))
        conv_blocks = []

        for i in range(nc):
            conv_blocks.append(conv_block(2, nd, nf, conv_dim=3))
            # conv_blocks.append(conv_block(2, nd, nf, conv_dim=2))

        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(self, x, k, m):
        '''
        x, k, m: (n, nx, ny, 2)
        '''
        def dc(x, k, m):
            kspace = x.permute(0, 2, 3, 4, 1).fft(2, True)
            kspace = kspace*(1-m) + k
            x = kspace.ifft(2, True).permute(0, 4, 1, 2, 3)
            return x

        x = x.permute(3, 0, 1, 2).unsqueeze(0)
        k = k.unsqueeze(0)
        m = m.unsqueeze(0)

        for i in range(0, self.nc, 2):
            x_cnn = self.conv_blocks[i](x)
            x = x + x_cnn
            x = dc(x, k, m)
            kspace = x.permute(0, 2, 3, 4, 1).fft(2, True)
            kspace = kspace*(1-m) + k
            x = kspace.ifft(2, True).permute(0, 4, 1, 2, 3)

            # x = x.squeeze(0).permute(1, 0, 2, 3)
            # x_cnn2d = self.conv_blocks[i](x)
            # x = x + x_cnn2d
            # x = dc(x.permute(1, 0, 2, 3).unsqueeze(0), k, m)
        x = x.squeeze(0).permute(1, 2, 3, 0)
        return x