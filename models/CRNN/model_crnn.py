import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np
from torch.utils.checkpoint import checkpoint

def data_consistency(k, k0, mask, noise_lvl=None):
    """
    k    - input in k-space
    k0   - initially sampled elements in k-space
    mask - corresponding nonzero location
    """
    v = noise_lvl
    if v:  # noisy case
        out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
    else:  # noiseless case
        out = (1 - mask) * k + mask * k0
    return out


class DataConsistencyInKspace(nn.Module):
    """ Create data consistency operator
    Warning: note that FFT2 (by the default of torch.fft) is applied to the last 2 axes of the input.
    This method detects if the input tensor is 4-dim (2D data) or 5-dim (3D data)
    and applies FFT2 to the (nx, ny) axis.
    """

    def __init__(self, noise_lvl=None, norm='ortho'):
        super(DataConsistencyInKspace, self).__init__()
        self.normalized = norm == 'ortho'
        self.noise_lvl = noise_lvl

    def forward(self, *input, **kwargs):
        return self.perform(*input)

    def perform(self, x, k0, mask):
        """
        x    - input in image domain, of shape (n, 2, nx, ny[, nt])
        k0   - initially sampled elements in k-space
        mask - corresponding nonzero location
        """

        if x.dim() == 4: # input is 2D
            x    = x.permute(0, 2, 3, 1)
            k0   = k0.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1)
        elif x.dim() == 5: # input is 3D
            x    = x.permute(0, 4, 2, 3, 1)
            k0   = k0.permute(0, 4, 2, 3, 1)
            mask = mask.permute(0, 4, 2, 3, 1)

        k = torch.fft(x, 2, normalized=self.normalized)
        out = data_consistency(k, k0, mask, self.noise_lvl)
        x_res = torch.ifft(out, 2, normalized=self.normalized)

        if x.dim() == 4:
            x_res = x_res.permute(0, 3, 1, 2)
        elif x.dim() == 5:
            x_res = x_res.permute(0, 4, 2, 3, 1)

        return x_res

class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations
    Parameters
    -----------------
    input: 4d tensor, shape (batch_size, channel, width, height)
    hidden: hidden states in temporal dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    hidden_iteration: hidden states in iteration dimension, 4d tensor, shape (batch_size, hidden_size, width, height)
    Returns
    -----------------
    output: 4d tensor, shape (batch_size, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(CRNNcell, self).__init__()
        self.kernel_size = kernel_size
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        # add iteration hidden connection
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=self.kernel_size // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, hidden_iteration, hidden):
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        ih_to_ih = self.ih2ih(hidden_iteration)

        hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)

        return hidden


class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    Parameters
    --------------------
    incomings: input: 5d tensor, [input_image] with shape (num_seqs, batch_size, channel, width, height)
               input_iteration: 5d tensor, [hidden states from previous iteration] with shape (n_seq, n_batch, hidden_size, width, height)
               test: True if in test mode, False if in train mode
    Returns
    --------------------
    output: 5d tensor, shape (n_seq, n_batch, hidden_size, width, height)
    """
    def __init__(self, input_size, hidden_size, kernel_size):
        super(BCRNNlayer, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.CRNN_model = CRNNcell(self.input_size, self.hidden_size, self.kernel_size)

    def forward(self, input, input_iteration, test=False):
        nt, nb, nc, nx, ny = input.shape
        size_h = [nb, self.hidden_size, nx, ny]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        output_f = []
        output_b = []
        # forward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[i], input_iteration[i], hidden)
            output_f.append(hidden)

        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(nt):
            hidden = self.CRNN_model(input[nt - i - 1], input_iteration[nt - i -1], hidden)

            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])

        output = output_f + output_b

        if nb == 1:
            output = output.view(nt, 1, self.hidden_size, nx, ny)

        return output


class CRNN_MRI(nn.Module):
    """
    Model for Dynamic MRI Reconstruction using Convolutional Neural Networks
    Parameters
    -----------------------
    incomings: three 5d tensors, [input_image, kspace_data, mask], each of shape (batch_size, 2, width, height, n_seq)
    Returns
    ------------------------------
    output: 5d tensor, [output_image] with shape (batch_size, 2, width, height, n_seq)
    """
    def __init__(self, n_ch=2, nf=64, ks=3, nc=5, nd=5):
        """
        :param n_ch: number of channels
        :param nf: number of filters
        :param ks: kernel size
        :param nc: number of iterations
        :param nd: number of CRNN/BCRNN/CNN layers in each iteration
        """
        super(CRNN_MRI, self).__init__()
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.ks = ks

        self.bcrnn = BCRNNlayer(n_ch, nf, ks)
        self.conv1_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv1_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv2_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_x = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3_h = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4_x = nn.Conv2d(nf, n_ch, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

        dcs = []
        for i in range(nc):
            dcs.append(DataConsistencyInKspace(norm='ortho'))
        self.dcs = dcs

    def forward(self, x, k, m, test=False):
        """
        x   - input in image domain, of shape (n, 2, nx, ny, n_seq)
        k   - initially sampled elements in k-space
        m   - corresponding nonzero location
        test - True: the model is in test mode, False: train mode
        """
        # (n_seq, nx, ny, 2) -> (1, 2, nx, ny, n_seq)
        x = x.permute(3, 1, 2, 0).unsqueeze(0)
        k = k.permute(3, 1, 2, 0).unsqueeze(0)
        m = m.permute(3, 1, 2, 0).unsqueeze(0)

        net = {}
        n_batch, n_ch, width, height, n_seq = x.size()
        size_h = [n_seq*n_batch, self.nf, width, height]
        if test:
            with torch.no_grad():
                hid_init = Variable(torch.zeros(size_h)).cuda()
        else:
            hid_init = Variable(torch.zeros(size_h)).cuda()

        for j in range(self.nd-1):
            net['t0_x%d'%j]=hid_init
        for i in range(1,self.nc+1):

            x = x.permute(4,0,1,2,3)
            x = x.contiguous()
            net['t%d_x0' % (i - 1)] = net['t%d_x0' % (i - 1)].view(n_seq, n_batch,self.nf,width, height)
            if x.requires_grad:
                net['t%d_x0'%i] = checkpoint(self.bcrnn, x, net['t%d_x0'%(i-1)])
            else:
                net['t%d_x0'%i] = self.bcrnn(x, net['t%d_x0'%(i-1)], test)
                
            net['t%d_x0'%i] = net['t%d_x0'%i].view(-1,self.nf,width, height)

            net['t%d_x1'%i] = self.conv1_x(net['t%d_x0'%i])
            net['t%d_h1'%i] = self.conv1_h(net['t%d_x1'%(i-1)])
            net['t%d_x1'%i] = self.relu(net['t%d_h1'%i]+net['t%d_x1'%i])

            net['t%d_x2'%i] = self.conv2_x(net['t%d_x1'%i])
            net['t%d_h2'%i] = self.conv2_h(net['t%d_x2'%(i-1)])
            net['t%d_x2'%i] = self.relu(net['t%d_h2'%i]+net['t%d_x2'%i])

            net['t%d_x3'%i] = self.conv3_x(net['t%d_x2'%i])
            net['t%d_h3'%i] = self.conv3_h(net['t%d_x3'%(i-1)])
            net['t%d_x3'%i] = self.relu(net['t%d_h3'%i]+net['t%d_x3'%i])

            net['t%d_x4'%i] = self.conv4_x(net['t%d_x3'%i])

            x = x.view(-1,n_ch,width, height)
            net['t%d_out'%i] = x + net['t%d_x4'%i]

            net['t%d_out'%i] = net['t%d_out'%i].view(-1,n_batch, n_ch, width, height)
            net['t%d_out'%i] = net['t%d_out'%i].permute(1,2,3,4,0)
            net['t%d_out'%i].contiguous()
            net['t%d_out'%i] = self.dcs[i-1].perform(net['t%d_out'%i], k, m)
            x = net['t%d_out'%i]

            # clean up i-1
            if not x.requires_grad:
                to_delete = [ key for key in net if ('t%d'%(i-1)) in key ]

                for elt in to_delete:
                    del net[elt]

                torch.cuda.empty_cache()
        out = net['t%d_out'%i].squeeze(0).permute(3, 1, 2, 0)
        return out