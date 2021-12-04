import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from model.Models import Encoder, Decoder
from model.Layers import PostNet
import model.util as util
import hparams as hp
from data_util.sp_code import bin_energy_enc


class FastNPSS(nn.Module):
    """ FastSpeech """

    def __init__(self):
        super(FastNPSS, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.energy_embeding = nn.Linear(hp.energy_output_channel, hp.word_vec_dim)
        self.energy_predictor = Decoder(output_channel=hp.energy_output_channel)

    def forward(self, src_seq, phn_count, dis_pos, f0, for_mask, g_bias, energy=None):

        encoder_output = self.encoder(src_seq)

        # length regular
        r_output = list()
        for batch, p_count in zip(encoder_output, phn_count):
            r_output.append(self.expand(batch, p_count))

        r_output = util.pad(r_output, dis_pos.shape[1])
        # temp0 = encoder_output.detach().numpy()
        # temp = r_output.detach().numpy()
        # temp1 = batch_pos_table.detach().numpy()

        energy_predicted = self.energy_predictor(r_output, dis_pos, f0, for_mask, g_bias)

        if energy is None:
            energy_src = bin_energy_enc(energy_predicted.detach().clone())
            # temp = energy_src.cpu().detach().numpy()
            energy_emdbeded = self.energy_embeding(energy_src)
        else:
            energy = bin_energy_enc(energy.detach().clone())
            energy_emdbeded = self.energy_embeding(energy)

        decoder_input = r_output + energy_emdbeded
        decoder_output = self.decoder(decoder_input, dis_pos, f0, for_mask, g_bias)

        return decoder_output, energy_predicted

    def expand(self, batch, p_count):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = p_count[i].item()
            expanded = vec.expand(int(expand_size), -1)
            out.append(expanded)

        out = torch.cat(out, 0)

        return out