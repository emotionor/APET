import torch
import torch.nn as nn
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    #emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return sin_inp #torch.flatten(emb, -2, -1)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 12) * 2)
        if channels % 4:
            channels += channels % 4
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 1).float() / channels))
        #inv_freq = 1.0 / (2 ** (torch.arange(0, channels, 1).float() ))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor, pos):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """

        # if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
        #     return self.cached_penc

        self.cached_penc = None
        batch_size, N, orig_ch = tensor.shape

        pos_x = pos[:,:,0]*2*np.pi
        pos_y = pos[:,:,1]*2*np.pi
        pos_z = pos[:,:,2]*2*np.pi
        #print(pos_x.size())
        #lenp = len(pos_x)
        pos_x = torch.rand(pos_x.size()).to(tensor.device)
        pos_y = torch.rand(pos_y.size()).to(tensor.device)
        pos_z = torch.rand(pos_z.size()).to(tensor.device)


        sin_inp_x_1 = torch.einsum("ij,k->ijk", pos_x, self.inv_freq).sin()
        sin_inp_x_2 = torch.einsum("ij,k->ijk", pos_x, self.inv_freq).cos()
        sin_inp_y_1 = torch.einsum("ij,k->ijk", pos_y, self.inv_freq).sin()
        sin_inp_y_2 = torch.einsum("ij,k->ijk", pos_y, self.inv_freq).cos()
        sin_inp_z_1 = torch.einsum("ij,k->ijk", pos_z, self.inv_freq).sin()
        sin_inp_z_2 = torch.einsum("ij,k->ijk", pos_z, self.inv_freq).cos()


        emb_x_1 = get_emb(sin_inp_x_1)
        emb_y_1 = get_emb(sin_inp_y_1)
        emb_z_1 = get_emb(sin_inp_z_1)

        emb_x_2 = get_emb(sin_inp_x_2)
        emb_y_2 = get_emb(sin_inp_y_2)
        emb_z_2 = get_emb(sin_inp_z_2)

        emb = torch.zeros((batch_size, N, self.channels * 6), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x_1
        emb[:, :, self.channels : 2 * self.channels] = emb_x_2
        emb[:, :, 2 * self.channels : 3 * self.channels] = emb_y_1
        emb[:, :, 3 * self.channels : 4 * self.channels] = emb_y_2
        emb[:, :, 4 * self.channels : 5 * self.channels] = emb_z_1
        emb[:, :, 5 * self.channels :] = emb_z_2



        res = tensor + emb
        return res






    