
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import hparams as hp
import numpy as np
import math

from model.SubLayers import MultiHeadAttention, PositionwiseFeedForward



class Linear(nn.Module):
    """
    Linear Module
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init='linear'):
        """
        :param in_dim: dimension of input
        :param out_dim: dimension of output
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Linear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        return self.linear_layer(x)


class PreNet(nn.Module):
    """
    Pre Net before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, p=0.5):
        """
        :param input_size: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(PreNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', Linear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p)),
            ('fc2', Linear(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p)),
        ]))

    def forward(self, input_):

        out = self.layer(input_)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.conv(x)
        return x


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class GLUBlock(torch.nn.Module):
    """GLU Block"""
    def __init__(self,
                 in_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 bias=True,):

        super(GLUBlock, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels,
                                    in_channels*2,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=1,
                                    dilation=dilation,
                                    bias=bias)

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        conv_x = self.conv(x)
        filter, gate = torch.chunk(conv_x, 2, dim=1)  # torch.cat的逆操作
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = x.permute(0, 2, 1).contiguous()

        return x


# 只有1个head
class AttentionBlock(nn.Module):

    def __init__(self, d_model, d_k, d_v, temperature, attn_dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k)
        self.w_ks = nn.Linear(d_model, d_k)
        self.w_vs = nn.Linear(d_model, d_v)

        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.fc = Linear(d_v, d_model)

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

        # 目前是所有head 公用一个sigma 当前设置只有1个head
        self.tau = nn.Parameter(torch.tensor(30.0))

        # (1, max_len, max_len)
        self.temp_n=0

    def forward(self, q, k, v, g_bias, mask=None):
        # (batch, input_len, hidden)
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        q = self.w_qs(q)
        k = self.w_ks(k)
        v = self.w_vs(v)

        attn = torch.bmm(q, k.transpose(1, 2))
        # (input_len_q, input_len_k)
        attn = attn / self.temperature

        sigma = self.tau
        divisor = 2 * sigma**2
        #gaussian_bias = attn.new(np.fromfunction(bias_f, attn.shape))
        gaussian_bias = g_bias[:attn.shape[0], :attn.shape[1], :attn.shape[1]]
        gaussian_bias = gaussian_bias / divisor.expand_as(attn)

        attn_gaussian = attn + gaussian_bias

        if mask is not None:
            attn_gaussian = attn_gaussian.masked_fill(mask, -np.inf)

        attn_softmax = self.softmax(attn_gaussian)
        attn_dropped = self.dropout(attn_softmax)
        output = torch.bmm(attn_dropped, v)

        output = self.fc(output)

        # test
        # if self.temp_n % 10000 == 0:
        #     atten_np = attn.detach().cpu().numpy()
        #     gaussian_np = gaussian_bias.detach().cpu().numpy()
        #     attn_gaussian_np = attn_gaussian.detach().cpu().numpy()
        #     attn_softmax_np = attn_softmax.detach().cpu().numpy()
        #
        #     # imshow shape(col row)
        #     # row: q  col:k
        #     plt.imshow(np.transpose(atten_np[0]), aspect='auto', origin='bottom',  interpolation='none')
        #     plt.show()
        #     plt.imshow(np.transpose(attn_softmax_np[0]), aspect='auto', origin='bottom', interpolation='none')
        #     plt.show()

            #imgPath = '/home/miku/Data/fast_cnnpss/img/'
            #mpimg.imsave(imgPath+str(self.temp_n)+"_attn_"+str(time.time())+'.jpg', np.transpose(atten_np[0]))
            #mpimg.imsave(imgPath + str(self.temp_n) + "_attnsoft_" + str(time.time()) + '.jpg', np.transpose(attn_softmax_np[0]))


        #self.temp_n = 1 + self.temp_n
        # test end


        return output


class AttSubLayerBlock(nn.Module):
    def __init__(self, d_model,
                     d_k,
                     d_v,
                     dropout=0.1):
        super(AttSubLayerBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(d_model)

        self.atten = AttentionBlock(d_model, d_k, d_v, temperature=np.power(d_k, 0.5))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, g_bias, non_pad_mask=None):
        residual = x
        x = self.layer_norm(x)
        # no mask cause feedforward
        atten = self.atten(x, x, x, g_bias)
        output = self.dropout(atten)
        output = output + residual
        output = output * math.sqrt(0.5)

        output *= non_pad_mask

        return output


class GLUSubLayerBlock(nn.Module):

    def __init__(self,  in_channels, dropout=0.1):
        super(GLUSubLayerBlock, self).__init__()

        self.layer_norm = nn.LayerNorm(in_channels)
        self.glu = GLUBlock(in_channels=in_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        glu = self.glu(x)
        output = self.dropout(glu)
        output = output + residual
        output = output * math.sqrt(0.5)

        return output


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 dilation=1,
                 bias=True,
                 w_init_gain='linear'):
        super(ConvNorm, self).__init__()

        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self,
                 n_mel_channels=hp.output_channel,
                 postnet_embedding_dim=512,
                 postnet_kernel_size=5,
                 postnet_n_convolutions=5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'),

                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size,
                             stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'),

                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim,
                         n_mel_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'),

                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)

        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(
                self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
