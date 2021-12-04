import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

        # 目前是所有head 公用一个sigma 当前设置只有1个head
        self.tau = nn.Parameter(torch.rand(1))

        # (1, max_len, max_len)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        sigma = self.tau**2
        divisor = 2 * sigma**2
        gaussian_bias = attn.new(np.fromfunction(bias_f, attn.shape))
        gaussian_bias = gaussian_bias / divisor.expand_as(attn)
        attn = attn + gaussian_bias

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


# batch row col
def bias_f(b, i, j):
    temp = -(i-j)**2
    return temp


# some test code
# ____________________
# attn_np = attn.detach().cpu().numpy()
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('TkAgg')
# plt.imshow(attn_np[0], aspect='auto', origin='bottom', interpolation='none')
# plt.show()
# gau = np.fromfunction(bias_f, attn.shape)
# plt.imshow(gau[0], aspect='auto', origin='bottom', interpolation='none')
# plt.show()