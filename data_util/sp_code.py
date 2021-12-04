import scipy.fftpack as fftpack
import librosa
import pyworld as pw
import numpy as np
import os
import soundfile as sf
import fnmatch
import matplotlib.pyplot as plt
import pysptk
from librosa.display import specshow
import copy
from config import f0_max, f0_min
import torch

gamma = 0
mcepInput = 3  # 0 for dB, 3 for magnitude
alpha = 0.45
en_floor = 10 ** (-80 / 20)


def code_harmonic(sp, order):

    #get mcep
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    #do fft and take real
    scale_mceps = copy.copy(mceps)
    scale_mceps[:, 0] *= 2
    scale_mceps[:, -1] *= 2
    mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    mfsc = np.fft.rfft(mirror).real

    return mfsc


def decode_harmonic(mfsc, fftlen):
    # get mcep back
    mceps_mirror = np.fft.irfft(mfsc)
    mceps_back = mceps_mirror[:, :60]
    mceps_back[:, 0] /= 2
    mceps_back[:, -1] /= 2

    #get sp
    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, mceps_back, alpha, gamma, fftlen=fftlen).real)

    return spSm


def one_hot_energy_enc(energy, min_e=0, max_e=1.0, e_bin=256):

    energy[energy > 0] = (energy[energy > 0] - min_e) * \
                         (e_bin - 2) / (max_e - min_e) + 1

    energy[energy < 0] = 0
    energy[energy > e_bin - 1] = e_bin - 1

    # f0_coarse (T_frame, f0_bin)
    e_coarse = np.rint(energy).astype(np.int)
    #print('Max energy: ', np.max(e_coarse), ' ||Min energy: ', np.min(e_coarse))
    assert (np.max(e_coarse) <= e_bin and np.min(e_coarse) >= 0)

    oh_list = []
    for i in range(len(e_coarse)):
        e_coarse_oh = np.zeros(e_bin)
        e_coarse_oh[e_coarse[i]] = 1
        oh_list.append(e_coarse_oh)

    code_e = np.stack(oh_list)

    # plt.imshow(np.transpose(code_e), aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    return code_e


def bin_energy_enc(energy, min_e=0, max_e=1.0, e_bin=256):

    energy[energy > 0] = (energy[energy > 0] - min_e) * \
                         (e_bin - 2) / (max_e - min_e) + 1

    energy[energy < 0] = 0
    energy[energy > e_bin - 1] = e_bin - 1

    # f0_coarse (T_frame, f0_bin)
    if torch.is_tensor(energy):
        e_coarse = torch.round(energy)
    else:
        e_coarse = np.rint(energy).astype(np.int)
        assert (np.max(e_coarse) < e_bin and np.min(e_coarse) >= 0)
    #print('Max energy: ', np.max(e_coarse), ' ||Min energy: ', np.min(e_coarse))

    return e_coarse


def one_hot_f0_enc(f0_mel, f0_mel_min, f0_mel_max, f0_bin):
    # f0_mel[f0_mel == 0] = 0
    # 大于0的分为f0_bin个箱

    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * \
                         (f0_bin - 2) / (f0_mel_max - f0_mel_min) + 1

    f0_mel[f0_mel < 0] = 0
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1

    # f0_coarse (T_frame, f0_bin)
    f0_coarse = np.rint(f0_mel).astype(np.int)
    print('Max f0: ', np.max(f0_coarse), ' ||Min f0: ', np.min(f0_coarse))
    assert (np.max(f0_coarse) <= f0_bin and np.min(f0_coarse) >= 0)

    oh_list = []
    for i in range(len(f0_coarse)):
        f0_coarse_oh = np.zeros(f0_bin)
        f0_coarse_oh[f0_coarse[i]] = 1
        oh_list.append(f0_coarse_oh)

    code_f0 = np.stack(oh_list)

    # plt.imshow(np.transpose(code_f0), aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    return code_f0


def trian_basis_f0_enc(f0, floor, ceil, num_basis=4):
    """
        Triangular basises f0 encodeing algorithm
            Parameters
            ----------
            f0 : ndarray
                Input F0 contour.
            floor: float
                Lower f0 limit in Hz.
            ceil: float
                Higher f0 limit in Hz.
            num_basis: int
                The number of triangular basises used to encoder f0
                Default: 4

            Returns
            ----------
            code_f0: ndarray
                The encoded result, shape=(len(f0), num_basis)
      """

    # n个basis,就会有n-1个overlap，每个overlap长度为1/2个basis_wide。因此，basis_wide符合下面的方程
    # num_basis*basis_wide-(ceil-floor) = (num_basis-1)*0.5*basis_wide，解方程得：
    basis_wide  = (ceil-floor)/(0.5*num_basis+0.5)

    code_f0 = np.zeros((len(f0), num_basis))
    for i, f0_value in enumerate(f0):
        for which in range(num_basis):
            code_f0[i,which] = trian_window_f(f0_value, which, basis_wide)

    return code_f0


def trian_basis_f0_dec(code_f0, floor, ceil, num_basis=4):##^#^

    assert code_f0.shape[-1] == num_basis, r'The depth of code_f0 doesn\'t match num_basis!'

    basis_wide  = (ceil-floor)/(0.5*num_basis+0.5)

    f0 = np.zeros(code_f0.shape[0])
    for i, code_f0_vec in enumerate(code_f0):

        nonzero_i = [i_b for i_b, e in enumerate(code_f0_vec) if e>0]

        if len(nonzero_i) == 0:
            f0_value = 0.0
        elif len(nonzero_i) == 1:
            which = nonzero_i[0]
            enc_f0_v = code_f0_vec[which]
            if nonzero_i[0] == 0:
                f0_value = trian_window_f_inverse(enc_f0_v, which, basis_wide)[0]
            else:
                f0_value = trian_window_f_inverse(enc_f0_v, which, basis_wide)[-1]
        elif len(nonzero_i) == 2:
            f0_possible_value1 = np.round(
                trian_window_f_inverse(code_f0_vec[nonzero_i[0]], nonzero_i[0], basis_wide),
                5
            )
            f0_possible_value2 = np.round(
                trian_window_f_inverse(code_f0_vec[nonzero_i[1]], nonzero_i[1], basis_wide),
                5
            )
            f0_value = [a for a in f0_possible_value1 if a in f0_possible_value2][0]

        else:
            raise ValueError("the num of none zero value of code_f0_vec not more than 2")

        f0[i] = f0_value

    return f0


def trian_window_f(f0_value, which, basis_wide):
    # 给定一个f0_value（float scaler），给定当前三角窗为第几个三角，给定三角窗的宽度
    # 计算出该三角窗应用在该f0 value后的值
    basis_min = 0.5*(basis_wide)*which
    basis_center = basis_min + 0.5*basis_wide
    basis_max = basis_center + 0.5*basis_wide

    if basis_min<=f0_value<basis_center:
        enc_f0_v = (f0_value-basis_min)*((1.0-0.0)/(0.5*basis_wide))
    elif basis_center<=f0_value<basis_max:
        enc_f0_v = (f0_value-basis_center)*(-(1.0-0.0)/(0.5*basis_wide))+1.0
    else:
        enc_f0_v = 0.0

    return enc_f0_v


def trian_window_f_inverse(enc_f0_v, which, basis_wide):
    # 给定一个enc f0 value(float scaler），给定当前三角窗为第几个三角，给定三角窗的宽度
    # 计算出应用该三角窗之前的f0原始值
    basis_min = 0.5*(basis_wide)*which
    basis_center = basis_min + 0.5*basis_wide
    basis_max = basis_center + 0.5*basis_wide

    f0_possible_value_0 = (enc_f0_v * (0.5*basis_wide))/(1.0-0.0) + basis_min
    f0_possible_value_1 = (enc_f0_v-1.0)* (0.5*basis_wide)/(-(1.0-0.0)) + basis_center

    return [f0_possible_value_0, f0_possible_value_1]


if __name__ == '__main__':
    # y, osr = sf.read('raw/nitech_jp_song070_f001_040_1.raw', subtype='PCM_16', channels=1, samplerate=48000,
    #                  endian='LITTLE')  # , start=56640, stop=262560)
    # D = np.abs(librosa.stft(y, hop_length=160)) ** 2
    # #D_db = librosa.power_to_db(D, ref=np.max)
    # S = librosa.feature.melspectrogram(S=D)
    # ptd_S = librosa.power_to_db(S)
    # mfcc = librosa.feature.mfcc(S=ptd_S, n_mfcc=60)
    #
    #
    # 使用DIO算法计算音频的基频F0

    #############测试f0_mel被triangular basis code以及decode之后是否与原来相等#############
    for r, _, files in os.walk('/Users/kissshot/pyProj/SVS/Data/xiaolongnv70_norm_with_br/raw_piece'):

        for f in files:
            if f.endswith('.wav'):
                wav_path = os.path.join(r, f)
                y, osr = sf.read(wav_path)
                sr = 32000
                if osr != sr:
                    y = librosa.resample(y, osr, sr)


                _f0, t = pw.dio(y, sr, f0_floor=50.0, f0_ceil=800.0, channels_in_octave=2, frame_period=pw.default_frame_period)
                print(_f0.shape)

                code_f0 = trian_basis_f0_enc(_f0, floor=f0_min, ceil=f0_max, num_basis=4)
                dec_f0 = trian_basis_f0_dec(code_f0,floor=f0_min, ceil=f0_max, num_basis=4)

                abs_dif = sum(abs(dec_f0 - _f0))
                abs_abs_dif = sum(abs(dec_f0) - abs(_f0))

                assert abs_dif<0.01
                assert abs_abs_dif <0.01

                dif_array = []
                for i in range(len(_f0)):
                    if dec_f0[i] != _f0[i]:
                        dif_array.append([_f0[i], dec_f0[i], i])
                dif_array = np.array(dif_array)
                match_array = np.concatenate(([_f0], [dec_f0], [_f0==dec_f0],[abs(dec_f0 - _f0)]),axis=0).T


                f0_mel = 1127*np.log(1+_f0/700)
                f0_mel_min = 1127 * np.log(1 + f0_min / 700)
                f0_mel_max = 1127 * np.log(1 + f0_max / 700)

                code_f0_mel = trian_basis_f0_enc(f0_mel, floor=f0_mel_min, ceil=f0_mel_max, num_basis=4)
                dec_f0_mel = trian_basis_f0_dec(code_f0_mel, floor=f0_mel_min, ceil=f0_mel_max, num_basis=4)

                abs_dif2 = sum(abs(dec_f0_mel - f0_mel))
                abs_abs_dif2 = sum(abs(dec_f0_mel) - abs(f0_mel))

                assert abs_dif2<0.01
                assert abs_abs_dif2<0.01

                dif_array2 = []
                for i in range(len(f0_mel)):
                    if dec_f0_mel[i] != f0_mel[i]:
                        dif_array.append([f0_mel[i], dec_f0_mel[i], i])
                dif_array2 = np.array(dif_array2)
                match_array2 = np.concatenate(([f0_mel], [dec_f0_mel], [f0_mel==dec_f0_mel],[abs(dec_f0_mel - f0_mel)]),axis=0).T

                print(1)
    #############################################################################


    # 使用CheapTrick算法计算音频的频谱包络
    sp = pw.cheaptrick(y, _f0, t, sr)

    _ap = pw.d4c(y, _f0, t, sr)


    specshow(10 * np.log10(sp.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('sp envelope spectrogram')
    plt.tight_layout()
    plt.show()



    mfsc = code_harmonic(sp, 59)



    specshow(mfsc.T, sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('mfsc')
    plt.tight_layout()
    plt.show()


    sp1 = decode_harmonic(mfsc, 2048)



    specshow(10 * np.log10(sp1.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('sp1 envelope spectrogram')
    plt.tight_layout()
    plt.show()

    synthesized = pw.synthesize(_f0, sp1, _ap, 32000, pw.default_frame_period)
    # 1.输出原始语音
    sf.write('../temp_test/world3.wav', synthesized, 32000)


    # ptd_S = librosa.power_to_db(np.transpose(_sp))
    # tran_ptd_S = (ptd_S - 0.45)/(1 - 0.45*ptd_S)
    # mfcc = librosa.feature.mfcc(S=tran_ptd_S, n_mfcc=60)
    #
    # _sp_min = np.min(mfcc)
    # _sp_max = np.max(mfcc)
    # mfcc = (mfcc - _sp_min)/(_sp_max - _sp_min)
    #
    # code_sp = pw.code_spectral_envelope(_sp, sr, 60)
    # t_code_sp = np.transpose(code_sp)
    #
    # _sp_min = np.min(t_code_sp)
    # _sp_max = np.max(t_code_sp)
    # t_code_sp = (t_code_sp - _sp_min) / (_sp_max - _sp_min)
    #
    # plt.imshow(mfcc, aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    # plt.imshow(t_code_sp, aspect='auto', origin='bottom', interpolation='none')
    # plt.show()
    #
    # decode_sp = pw.decode_spectral_envelope(code_sp, 32000, 2048)
    # x = code_harmonic(_sp)
    order = 60
    gamma = 0
    mcepInput = 3  # 0 for dB, 3 for magnitude
    alpha = 0.35
    fftlen = (sp.shape[1] - 1) * 2
    en_floor = 10 ** (-80 / 20)

    # Reduction and Interpolation
    mceps = np.apply_along_axis(pysptk.mcep, 1, sp, order - 1, alpha, itype=mcepInput, threshold=en_floor)

    # scale_mceps = copy.copy(mceps)
    # scale_mceps[:, 0] *=2
    # scale_mceps[:, -1] *=2
    # mirror = np.hstack([scale_mceps[:, :-1], scale_mceps[:, -1:0:-1]])
    # mfsc = np.fft.rfft2(mirror).real
    mfsc = fftpack.dct(mceps, norm='ortho')

    specshow(mfsc.T, sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('MCEPS')
    plt.tight_layout()
    plt.show()

    # itest = np.fft.ifft2(mfsc).real
    itest = fftpack.idct(mfsc, norm='ortho')

    specshow(itest.T, sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('itest')
    plt.tight_layout()
    plt.show()

    spSm = np.exp(np.apply_along_axis(pysptk.mgc2sp, 1, itest, alpha, gamma, fftlen=fftlen).real)

    specshow(10 * np.log10(sp.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('Original envelope spectrogram')
    plt.tight_layout()
    plt.show()

    specshow(10 * np.log10(spSm.T), sr=sr, hop_length=80, x_axis='time')
    plt.colorbar()
    plt.title('Smooth envelope spectrogram')
    plt.tight_layout()
    plt.show()

    synthesized = pw.synthesize(_f0, spSm, _ap, 32000, pw.default_frame_period)
    # 1.输出原始语音
    sf.write('../temp_test/dct.wav', synthesized, 32000)

    # mgc = pysptk.mcep(np.sqrt(fft), 59, 0.35, itype=3)
    # mfsc = np.exp(pysptk.mgc2sp(mgc, 0.35, fftlen=2048).real)
    # pysptk.mgc2sp
    # pass