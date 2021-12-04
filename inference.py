import sys
sys.path.append("..")
import matplotlib
#matplotlib.use('Agg')
from model.FastNPSS import FastNPSS
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from data_util.sp_code import decode_harmonic
import pyworld as pw
from config import *
import soundfile as sf
import model.util as util
from train_util import ScheduledOptim, FastnpssLoss
from dataset import FastnpssDataset, collate_fn
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import hparams as hp
import torch.nn as nn
import librosa
from model.Modules import bias_f
from pylab import rcParams
rcParams['figure.figsize'] = 15, 5
from torch.distributions.normal import Normal


class ModelEvaluater():

    def __init__(self, checkpoint_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_count = torch.cuda.device_count()
        # Define model
        self.model = FastNPSS().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

        self.fastnpssLoss = FastnpssLoss()
        # Get dataset
        # dataset = FastnpssDataset(DATA_ROOT_PATH, train=False)

        max_bias_shape = (hp.batch_size, hp.max_sep_len, hp.max_sep_len)
        self.G_BIAS = torch.Tensor(np.fromfunction(bias_f, max_bias_shape)).to(self.device)

        # self.test_loader = DataLoader(dataset,
        #                                   batch_size=hp.batch_size,
        #                                   shuffle=True,
        #                                   collate_fn=collate_fn,
        #                                   drop_last=True,
        #                                   num_workers=cpu_count())

    def evaluate(self):

        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)
            print('multiple device using: ', self.device_count)

        step = 0

        epoch_loss = 0
        epoch_step = 0

        for (src, target) in iter(self.test_loader):
            phn, phn_count, f0 = src
            target, dis_pos = target

            phn = torch.Tensor(phn).to(self.device)
            phn_count = torch.Tensor(phn_count).to(self.device)
            f0 = torch.Tensor(f0).to(self.device)

            dis_pos = torch.Tensor(dis_pos).long().to(self.device)
            target = torch.Tensor(target).to(self.device)

            timbre = self.model(phn, phn_count, dis_pos, f0)

            timbre_loss = self.fastnpssLoss.forward(timbre, target)
            loss = timbre_loss.item()
            epoch_loss += loss
            epoch_step += 1
            # print('loss: ', loss)
            step += 1

            print('loss: ', loss)
            # time step duration:

        print("average loss: " + str(epoch_loss / epoch_step))

    def generate_wav(self, phn_condi, frame_condi, f0_np, name, energy_condi=None):
        print('generate ', name, ' len of this file: ', len(f0_np))

        if len(frame_condi) > 2048:
            return

        phn = phn_condi[:, 1:]
        phn_count = phn_condi[:, :1]


        # origin_f0 = f0_np
        # f0_oh = self.make_f0_condi(origin_f0)
        # f0 = torch.Tensor(f0_oh).to(self.device).unsqueeze(0)

        formask_np = np.array([(i+1) for i in range(len(frame_condi))])

        phn = torch.Tensor(phn).to(self.device).unsqueeze(0)
        phn_count = torch.Tensor(phn_count).to(self.device).unsqueeze(0)

        formask = torch.Tensor(formask_np).long().to(self.device).unsqueeze(0)

        if energy_condi is not None:
            energy_condi = torch.Tensor(energy_condi).to(self.device).unsqueeze(-1)

        f0 = frame_condi[:, 2:]
        f0 = torch.Tensor(f0).to(self.device).unsqueeze(0)

        pos_in_note = frame_condi[:, :1].squeeze()
        pos_in_note_sinusoid = util.get_s2s_singing_sinusoid_pos_in_x(pos_in_note)
        pos_in_note_sinusoid = torch.Tensor(pos_in_note_sinusoid).to(self.device).unsqueeze(0)

        # (1, len, 64)
        timbre, energy_pred = self.model(phn, phn_count, pos_in_note_sinusoid, f0, formask, self.G_BIAS, energy_condi)
        timbre_postnet = timbre.squeeze().cpu().detach()

        energy_predicted = energy_pred.detach().cpu().squeeze()

        # energy_predicted *= 64
        # plt.plot(energy_predicted, color='red')
        # if energy_condi is not None:
        #     energy_condi = energy_condi.squeeze()
        #     energy_condi *= 64
        #     plt.plot(energy_condi.cpu(), color='green')
        # plt.show()

        decode_sp, decode_ap = self.decode_timbre(timbre_postnet.numpy())
        # origin_timbre = np.concatenate((ap, sp), axis=1)
        #
        # self.plot_data([(np.transpose(timbre_postnet), None, None, energy_predicted),
        #            (np.transpose(origin_timbre), None, None, energy_condi.cpu())],
        #           ['Synthetized Spectrogram', 'Ground-Truth Spectrogram'],
        #           filename=os.path.join(GEN_PATH, 'step_%d_%s.png' % (999, name)))

        # plt.imshow(np.transpose(timbre_postnet), aspect='auto', origin='bottom', interpolation='none')
        # plt.show()
        #
        # plt.imshow(np.transpose(origin_timbre), aspect='auto', origin='bottom', interpolation='none')
        # plt.show()
        #
        # plt.plot(f0_np)
        # plt.show()

        synthesized = pw.synthesize(f0_np, decode_sp, decode_ap, sample_rate, pw.default_frame_period*2)

        return synthesized

    def make_f0_condi(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        # f0_mel[f0_mel == 0] = 0
        # 大于0的分为256个箱
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
                             (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

        f0_mel[f0_mel < 0] = 0
        f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

        f0_coarse = np.rint(f0_mel).astype(np.int)
        print('Max f0: ', np.max(f0_coarse), ' ||Min f0: ', np.min(f0_coarse))
        assert (np.max(f0_coarse) <= f0_bin and np.min(f0_coarse) >= 0)

        oh_list =[]
        for i in range(len(f0)):
            f0_coarse_oh = np.zeros(f0_bin)
            f0_coarse_oh[f0_coarse[i]] = 1
            oh_list.append(f0_coarse_oh)

        return np.stack(oh_list)

    @staticmethod
    def decode_timbre(coded_timbre):

        coded_sp = coded_timbre[:, 4:]
        coded_ap = coded_timbre[:, :4]

        [sp_min, sp_max, ap_min, ap_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'))

        coded_sp = (coded_sp + 0.5) * (sp_max - sp_min) + sp_min
        coded_ap = (coded_ap + 0.5) * (ap_max - ap_min) + ap_min

        fft_size = 2048
        decode_sp = decode_harmonic(coded_sp, fft_size)
        decode_ap = pw.decode_aperiodicity(coded_ap.astype(np.double), sample_rate, fft_size)

        return decode_sp, decode_ap

    @staticmethod
    def decode_ap(coded_ap):
        [sp_min, sp_max, ap_min, ap_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'))

        coded_ap = (coded_ap + 0.5) * (ap_max - ap_min) + ap_min

        fft_size = 2048

        decode_ap = pw.decode_aperiodicity(coded_ap.astype(np.double), sample_rate, fft_size)

        return decode_ap

    @staticmethod
    def decode_sp(coded_sp):

        [sp_min, sp_max, ap_min, ap_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'))

        coded_sp = (coded_sp + 0.5) * (sp_max - sp_min) + sp_min

        decode_sp = decode_harmonic(coded_sp, fft_size)

        return decode_sp

    def plot_data(self, data, titles=None, filename=None):
        [energy_min, energy_max] = np.load(os.path.join(DATA_ROOT_PATH, 'min_max_energy.npy'))

        fig, axes = plt.subplots(len(data), 1, squeeze=False, figsize=(15, 4))
        if titles is None:
            titles = [None for i in range(len(data))]

        def add_axis(fig, old_ax, offset=0):
            ax = fig.add_axes(old_ax.get_position(), anchor='W')
            ax.set_facecolor("None")
            return ax

        for i in range(len(data)):
            spectrogram, pitch, pitch_norm, energy = data[i]
            #         spectrogram=np.swapaxes(spectrogram,0,1)
            axes[i][0].imshow(spectrogram, origin='lower')
            axes[i][0].set_aspect('auto', adjustable='box')
            axes[i][0].set_ylim(0, 64)
            axes[i][0].set_title(titles[i], fontsize='medium')
            axes[i][0].tick_params(labelsize='x-small', left=False, labelleft=False)
            axes[i][0].set_anchor('W')

            ax1 = add_axis(fig, axes[i][0])
            if pitch_norm is not None:
                ax1.plot(pitch_norm, color='red', alpha=0.5, lw=1.0)
            if pitch is not None:
                ax1.plot(pitch, color='tomato', lw=1.0)
            ax1.set_xlim(0, spectrogram.shape[1])
            ax1.set_ylim(f0_min, f0_max)
            ax1.set_ylabel('F0', color='tomato')
            ax1.tick_params(labelsize='x-small', colors='tomato', bottom=False, labelbottom=False)

            ax2 = add_axis(fig, axes[i][0], 1.2)
            ax2.plot(energy, color='darkviolet', lw=1.0)
            ax2.set_xlim(0, spectrogram.shape[1])
            ax2.set_ylim(0, 1)
            ax2.set_ylabel('Energy', color='darkviolet')
            ax2.yaxis.set_label_position('right')
            ax2.tick_params(labelsize='x-small', colors='darkviolet', bottom=False, labelbottom=False, left=False,
                            labelleft=False, right=True, labelright=True)

        plt.savefig(filename, dpi=200)
        plt.close()





if __name__ == '__main__':
    model = ModelEvaluater('snapshots/default_1699_2021-11-23_14-29-26')

    # model.evaluate()


    #dist_name = '33_niyaodequannazou'

    data_folder = os.path.join(DATA_ROOT_PATH, 'test')
    f0_folder = os.path.join(data_folder, 'f0')
    condi_folder = os.path.join(data_folder, 'condition')
    dirlist = os.listdir(f0_folder)

    sp_folder = os.path.join(data_folder, 'sp')
    ap_folder = os.path.join(data_folder, 'ap')

    for item in dirlist:
        if 'npy' not in item:
            continue

        name = item.replace('_f0.npy', '')
        # if dist_name not in name:
        #    continue
        print(name)

        f0 = np.load(os.path.join(f0_folder, name + '_f0.npy'))

        phn_condi = np.load(os.path.join(condi_folder, name + '_phn_condi.npy'))
        frame_condi = np.load(os.path.join(condi_folder, name + '_frame_condi.npy'))

        energy_condi = np.load(os.path.join(condi_folder, name + '_energy_condi.npy'))

        ap = np.load(os.path.join(ap_folder, name + '_ap.npy'))
        sp = np.load(os.path.join(sp_folder, name + '_sp.npy'))

        synthesized = model.generate_wav(phn_condi, frame_condi, f0, name, energy_condi)
        synthesized_norm = librosa.util.normalize(np.array(synthesized))
        if not os.path.exists(GEN_PATH):
            os.mkdir(GEN_PATH)
        sf.write(os.path.join(GEN_PATH, name + '.wav'), synthesized_norm, sample_rate)

    # data_folder = os.path.join(DATA_ROOT_PATH, 'train')
    # f0_folder = os.path.join(data_folder, 'f0')
    # condi_folder = os.path.join(data_folder, 'condition')
    # dirlist = os.listdir(f0_folder)
    #
    # for item in dirlist:
    #     name = item.replace('_f0.npy', '')
    #     if dist_name not in name:
    #         continue
    #     print(name)
    #     f0 = np.load(os.path.join(f0_folder, name + '_f0.npy'))
    #     condition = np.load(os.path.join(condi_folder, name + '_condi.npy'))
    #     model.generate_wav(condition, f0, name)


# 699 0.0005879239282122155
# 799 0.0005541354433417593
# 1399
