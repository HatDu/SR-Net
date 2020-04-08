import torch
from torch import nn
class ConvBlock(torch.nn.Module):
    def __init__(self, n_convs=5, n_filters=16, in_chans=2, out_chans=2):
        super(ConvBlock, self).__init__()
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.in_chans = in_chans
        self.out_chans = out_chans

        first_conv = nn.Sequential(nn.Conv2d(in_chans, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(1e-1, inplace=True))
        simple_convs = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU(1e-1, inplace=True))
            for i in range(n_convs - 2)
        ])
        last_conv = nn.Conv2d(n_filters, out_chans, kernel_size=3, padding=1, bias=True)
        self.overall_convs = nn.Sequential(first_conv, simple_convs, last_conv)

    def forward(self, x):
        y = self.overall_convs(x)
        return y

class Hybird_Net(nn.Module):
    def __init__(self, depth_str='ikikii', nd=5, nf=32):
        super().__init__()
        self.depth_str = depth_str
        self.n_cascade = len(depth_str)
        self.conv_layers = nn.ModuleList([
            ConvBlock(nd, nf) for _ in range(self.n_cascade)
        ])
    def forward(self, x, k0, mask):
        image = x.permute(0, 3, 1, 2)
        for ii, conv_layer in zip(self.depth_str, self.conv_layers):
            if ii == 'i':
                res_image = image
                res_image = conv_layer(res_image)
                image = image + res_image
                del res_image

                # data consistency layer in image space
                cnn_fft = image.permute(0, 2, 3, 1).fft(2, True)
                kspace = cnn_fft*(1-mask) + k0
                del cnn_fft
                image = kspace.ifft(2, True).permute(0, 3, 1, 2)
            else:
                kspace = image.permute(0, 2, 3, 1).fft(2, True).permute(0, 3, 1, 2)
                res_kspace = kspace
                kspace = conv_layer(kspace)
                kspace = kspace + res_kspace
                del res_kspace

                # data consistency layer in kspace
                kspace = kspace.permute(0, 2, 3, 1)
                kspace = kspace*(1-mask) + k0
                image = kspace.ifft(2, True).permute(0, 3, 1, 2)
        image = image.permute(0, 2, 3, 1)
        return image