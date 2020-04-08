import torch
from torch import nn
class ConvBlock(torch.nn.Module):
    def __init__(self, n_convs=5, n_filters=48, in_chans=2, out_chans=2):
        super(ConvBlock, self).__init__()
        self.n_convs = n_convs
        self.n_filters = n_filters
        self.in_chans = in_chans
        self.out_chans = out_chans

        first_conv = nn.Sequential(nn.Conv2d(in_chans, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU())
        simple_convs = nn.Sequential(*[
            nn.Sequential(nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1, bias=True), nn.LeakyReLU())
            for i in range(n_convs - 2)
        ])
        last_conv = nn.Conv2d(n_filters, out_chans, kernel_size=3, padding=1, bias=True)
        self.overall_convs = nn.Sequential(first_conv, simple_convs, last_conv)

    def forward(self, x):
        y = self.overall_convs(x)
        return y


class KikiNet(torch.nn.Module):
    def __init__(self, n_cascade=5, n_convs=5, n_filters=16):
        super(KikiNet, self).__init__()

        self.n_cascade = n_cascade
        self.n_convs = n_convs
        self.n_filters = n_filters

        self.i_conv_layers = nn.ModuleList([ConvBlock(n_convs, n_filters) for _ in range(n_cascade)])
        self.k_conv_layers = nn.ModuleList([ConvBlock(n_convs, n_filters) for _ in range(n_cascade)])

    def forward(self, x, k0, mask):
        k_i = x.fft(2, True)
        x_i = x.permute(0, 3, 1, 2)
        # this because pytorch doesn't support NHWC
        for i, (i_conv_layer, k_conv_layer) in enumerate(zip(self.i_conv_layers, self.k_conv_layers)):
            # KConv
            k_i = k_i.permute(0, 3, 1, 2)
            k_i = k_conv_layer(k_i)
            k_i = k_i.permute(0, 2, 3, 1)
            # skip connection
            im = k_i.ifft(2, True)
            im = im.permute(0, 3, 1, 2)
            x_i = x_i + im

            # ICNN
            x_i = i_conv_layer(x_i)
            # IDC
            x_ik = x_i.permute(0, 2, 3, 1).fft(2, True)
            x_ik = (1-mask)*x_ik + k0
            k_i = x_ik
            x_i = k_i.ifft(2, True).permute(0, 3, 1, 2)
        return x_i.permute(0, 2, 3, 1)