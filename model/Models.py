import torch
import torch.nn as nn
import numpy as np

import model.Constants as Constants
from model.Layers import GLUBlock, Linear, AttSubLayerBlock, GLUSubLayerBlock
import hparams as hp
import math


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


# 一般的transformer postion encoding
def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    ''' Encoder '''

    def __init__(self,
                 input_channel=hp.input_channel,
                 d_word_vec=hp.word_vec_dim,
                 glu_channel=hp.glu_channel,
                 n_layers=hp.encoder_glu_layer):

        super(Encoder, self).__init__()

        self.src_word_emb = nn.Linear(input_channel, d_word_vec, bias=False)

        self.start_fc = Linear(d_word_vec, glu_channel)

        self.glu_stack = nn.ModuleList()

        self.n_layers = n_layers
        for b in range(n_layers):
            self.glu_stack.append(GLUBlock(glu_channel))

        self.end_fc = Linear(glu_channel, d_word_vec)

    def forward(self, src_seq):

        emb = self.src_word_emb(src_seq)
        glu = self.start_fc(emb)

        for i in range(self.n_layers):
            glu = self.glu_stack[i](glu)

        output = self.end_fc(glu)

        output = output + emb
        output = output * math.sqrt(0.5)

        return output


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self,
                 len_max_seq=hp.max_sep_len,
                 d_word_vec=hp.word_vec_dim,
                 n_layers=hp.decoder_n_layer,
                 d_k=hp.word_vec_dim,
                 d_v=hp.word_vec_dim,
                 d_model=hp.word_vec_dim,
                 d_input_f0=hp.input_f0_channel,
                 output_channel=hp.output_channel,
                 dropout=hp.dropout):

        super(Decoder, self).__init__()

        n_position = len_max_seq + 1

        self.fc = Linear(d_input_f0, d_word_vec)

        self.pos_fc = Linear(4, d_word_vec)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, 4, padding_idx=0),
            freeze=True)

        self.n_layers = n_layers
        self.atten_stack = nn.ModuleList()
        self.glu_stack = nn.ModuleList()

        for b in range(n_layers):
            self.atten_stack.append(AttSubLayerBlock(d_model, d_k, d_v, dropout))
            self.glu_stack.append(GLUSubLayerBlock(d_model, dropout))

        self.end_fc = Linear(d_model, output_channel)

    def forward(self, enc_seq, enc_pos, f0, for_mask, g_bias, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(for_mask)

        # -- Forward
        f0_condi = self.fc(f0)

        pos_condi = self.pos_fc(enc_pos)

        # # test
        # pos_input_np = enc_pos.numpy()
        # import matplotlib.pyplot as plt
        # # imshow shape(col row)
        #
        # # row: q  col:k
        # plt.imshow(pos_input_np[0], aspect='auto', origin='bottom', interpolation='none')
        # plt.show()
        # # test end

        dec_output = enc_seq + pos_condi + f0_condi

        for i in range(self.n_layers):
            dec_output = self.atten_stack[i](dec_output, g_bias, non_pad_mask)
            dec_output = self.glu_stack[i](dec_output)

        output = self.end_fc(dec_output)

        return output
