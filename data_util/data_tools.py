import librosa
import numpy as np
from data_util.pinyin_phone import pp_dict_reverse, get_all_phon, get_shengmu
import soundfile as sf

import data_util.textgrid as textgrid
from data_util.sp_code import code_harmonic, one_hot_f0_enc, trian_basis_f0_enc, one_hot_energy_enc
from config import *

import matplotlib.pyplot as plt
import torch


all_phn = get_all_phon()
sheng_mu = get_shengmu()


def process_wav(wav_path, f0_code_mode=None):
    y, osr = sf.read(wav_path)
    if len(y.shape) > 1: #对可能是双声道数据的处理
        y = np.ascontiguousarray((y[:, 0]+y[:, 1])/2)
    sr = sample_rate
    if osr != sr: #对可能不符合模型所需采样率的数据，进行resample处理
        y = librosa.resample(y, osr, sr)

    #sf.write(wav_path, y, sample_rate)

    # 使用dio算法计算音频的基频F0
    _f0, t = pw.dio(y, sr, f0_floor=f0_min, f0_ceil=f0_max,
                        frame_period=(1000*hop/sample_rate))

    # plt.plot(_f0)
    # plt.title(wav_path.split('/')[-1])
    # plt.show()
    #_f0 = pw.stonemask(y, _f0, t, sr)
    # 如果该数据的f0数值中，有大于等于f0_max的情况则丢弃该数据，这个函数直接返回一个错误信息供后续做丢弃处理
    # （之所以是大于等于，是为了后续triangular basis encode的时候没有歧义）
    if len(_f0[_f0 >= f0_max]):
        return 'wrong_f0','wrong_f0','wrong_f0','wrong_f0','wrong_f0','wrong_f0'
    print('f0 shape: ',_f0.shape)

    # 将f0进行mel scale处理
    f0_mel = 1127*np.log(1+_f0/700)
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)

    # 将f0进行离散分箱，分成256维: f0 (T_frame, ) -> f0_coarse (T_frame, f0_bin)
    if not f0_code_mode:
        code_f0 = one_hot_f0_enc(f0_mel, f0_mel_min, f0_mel_max, f0_bin)
    # 将f0进行triangular basis code， f0 (T_frame, ) -> code_f0 (T_frame, num_basis i.e. 4)
    elif f0_code_mode == 'triangular_basis':
        code_f0 = trian_basis_f0_enc(f0_mel, floor=f0_mel_min, ceil=f0_mel_max, num_basis=4)
        print('code_f0 shape: ', code_f0.shape)
    else:
        raise ValueError('wrong f0 code mode!!')

    # 使用CheapTrick算法计算音频的频谱包络
    _sp = pw.cheaptrick(y, _f0, t, sr)
    energy = calculate_energy(y)
    print('energy shape: ', energy.shape)
    #code_energy = one_hot_energy_enc(energy)
    #print('code_energy shape: ', code_energy.shape)
    # plt.imshow(np.transpose(np.log(_sp)), aspect='auto', origin='bottom', interpolation='none')
    # plt.plot(energy*1024)
    # plt.plot(_f0)
    # plt.show()

    code_sp = code_harmonic(_sp, 60)
    print('sp and code_sp shape: ', _sp.shape, code_sp.shape)
    # 计算aperiodic参数
    _ap = pw.d4c(y, _f0, t, sr)

    code_ap = pw.code_aperiodicity(_ap, sr)
    print('ap and code_ap shape: ', _ap.shape, code_ap.shape)

    return _f0, code_f0, _sp, code_sp, _ap, code_ap, energy


def calculate_energy(y):

    # for test
    # sample_len = len(y)
    # energy_len = int(sample_len/hop) + 1
    #
    # energy1 = []
    # for i in range(energy_len-2):
    #     test = y[hop * i:hop * i + fft_size]
    #     x = np.linalg.norm(test)
    #     energy1.append(x)

    energy = librosa.feature.rms(y, frame_length=fft_size, hop_length=hop)

    return energy[0]


def process_phon_label(label_path):

    py_grid = textgrid.TextGrid.fromFile(label_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':  # 人工标注的
            source_tier = tier
            break
        elif tier.name == 'phones':  # Montreal-Forced-Aligner 标注
            source_tier = tier
            break

    assert source_tier != None

    time_phon_list = []
    phon_list = []
    for i, interval in enumerate(source_tier):
        phn = interval.mark.strip()

        tup = (int(round(interval.minTime*sample_rate/hop)),
               int(round(interval.maxTime*sample_rate/hop)), phn)

        assert (phn in all_phn)

        time_phon_list.append(tup)
        if phn not in phon_list:
            phon_list.append(phn)

    return time_phon_list, phon_list


# 名字长吧哈哈哈
def time_phon_list_to_time_pinyin_list(time_phon_list):
    time_pinyin_list = []
    temp_list = []
    temp_start = 0
    for time_phon in time_phon_list:
        start = time_phon[0]
        end = time_phon[1]
        phn = time_phon[2].strip()

        if phn in sheng_mu:
            temp_list.append(phn)
            temp_start = start
            continue
        elif len(temp_list) > 0:
            temp_list.append(phn)
            pinyin = pp_dict_reverse[str(temp_list)]
            temp_list.clear()
            time_pinyin_list.append((temp_start, end, pinyin))
        else:
            if str([phn]) in pp_dict_reverse:
                pinyin = pp_dict_reverse[str([phn])]
            else:
                pinyin = phn

            time_pinyin_list.append((start, end, pinyin))

    for i in time_pinyin_list:
        print(i)
    return time_pinyin_list


def add_syllable_pos(time_phon_list):
    # time_phon_list: [(strat_time, end_time, phoneme),...,]  len=T_phn

    # 给每个time phon加上一个syllable position(phn相对于syllable的地位)：
    # 0：代表phn是音节的开头，
    # 1：代表音节的结尾

    time_phon_list_with_syllable_pos = []

    syllable_pos = 0
    for time_phon in time_phon_list:
        start = time_phon[0]
        end = time_phon[1]
        phn = time_phon[2].strip() # ' str '.strip() > 'str'去掉string前后的空格

        time_phon_list_with_syllable_pos.append((start, end, phn, syllable_pos))

        syllable_pos += 1

        if phn not in sheng_mu:
            syllable_pos = 0

    # time_phon_list_with_syllable_pos: [(strat_time, end_time, phoneme, pos1),...,]  len=T_phn
    return time_phon_list_with_syllable_pos


def add_syllable_pos2(time_phon_list):

    # time_phon_list: [(strat_time, end_time, phoneme, *pos1),...,] len=T_phn
    # *pos1代表有可能有pos1，有可能没有。因为add_syllable_pos这个方法有可能会在使用该方法之前调用

    # 给每个phn打上一个标签，标签分为以下四种
    # 0: 开头声母；
    # 1：结尾韵母；
    # 2：单韵母（一个韵母就是一个音节的情况）；
    # 3：空。
    time_phon_list_with_syllable_pos = []

    for i, time_phon in enumerate(time_phon_list):
        before_phn = "None" if i==0 else time_phon_list[i-1][2].strip()
        phn = time_phon[2].strip()

        if phn in sheng_mu:
            syllable_pos = 0
        elif phn not in ['sil','pau','br']:
            #上个phn是韵母 or 空 or 边界线，则这是一个单韵母音节
            if before_phn in sheng_mu:
                syllable_pos = 1
            else:
                syllable_pos = 2
        else:
            syllable_pos = 3

        time_phon += (syllable_pos, )
        time_phon_list_with_syllable_pos.append(time_phon)

    # time_phon_list_with_syllable_pos: [(strat_time, end_time, phoneme, *pos1, *pos2),...,]
    return time_phon_list_with_syllable_pos


def get_frame_pos_within_phn_by_idx(time_phon_list, i):

    for j, time_phon in enumerate(time_phon_list):
        begin = time_phon[0]
        end = time_phon[1]
        width = end - begin  # [begin, end)

        if begin <= i < end:
            if width == 1:
                pos_in_phon = 0
            else:
                pos_in_phon = (i-begin)/(width-1)

    return pos_in_phon


def get_frame_pos_within_note_by_idx(time_phon_list, i):

    split_idices = [0]
    for j, time_phon in enumerate(time_phon_list):
        phn = time_phon[2].strip()

        if phn not in sheng_mu:
            if j not in split_idices:
                split_idices.append(j)
    split_idices.append(len(time_phon_list))

    time_note_list = []
    for ii in range(len(split_idices)-1):

        split_start = split_idices[ii]
        note_start_fr = time_phon_list[split_start][0]

        if ii+1 == len(split_idices)-1:
            note_end_fr = time_phon_list[split_start][1]
        else:
            split_end = split_idices[ii+1]
            note_end_fr = time_phon_list[split_end][0]

        time_note_list.append((note_start_fr, note_end_fr))

        if note_start_fr<=i<note_end_fr:
            pos_within_note_fr = i-note_start_fr
            note_len = note_end_fr-note_start_fr
            pos_within_note = pos_within_note_fr/(note_len-1)

    return pos_within_note, time_note_list


def get_frame_pos_within_pinyin_by_idx(pinyin_list, i):

    for pinyin in pinyin_list:

        note_start_fr = pinyin[0]
        note_end_fr = pinyin[1]

        if note_start_fr <= i < note_end_fr:
            pos_within_note_fr = i-note_start_fr
            note_len = note_end_fr-note_start_fr
            if note_len > 1:
                pos_within_note = pos_within_note_fr/(note_len-1)
            else:
                pos_within_note = 0

    return pos_within_note


def get_phon_condi_by_time_phon(time_phon):

    cur_phn = all_phn.index(time_phon[2])
    cur_phn_syllable_pos = time_phon[3]
    cur_phn_syllable_pos2 = time_phon[4]
    phn_count = time_phon[1] - time_phon[0]

    # if phn_count < 0:
    #     pass
    # assert(phn_count >= 0)

    return cur_phn, cur_phn_syllable_pos, cur_phn_syllable_pos2, phn_count


def make_frame_condition(time_phon_list, f0):

    frame_condi_list = []
    debug_list =[]

    pinyin_list = time_phon_list_to_time_pinyin_list(time_phon_list)

    for i in range(len(f0)):

        #pos_in_note, _ = get_frame_pos_within_note_by_idx(time_phon_list, i)
        pos_in_note = get_frame_pos_within_pinyin_by_idx(pinyin_list, i)

        pos_in_phn = get_frame_pos_within_phn_by_idx(time_phon_list, i)
        frame_condi_list.append(
            np.concatenate(([pos_in_note], [pos_in_phn], f0[i])).astype(np.float32) # 1+1+4
        )
        debug_list.append((pos_in_note, pos_in_phn))

    return np.stack(frame_condi_list)


def make_phn_condition(time_phon_list, f0):
    #time_phon_list: [(start_frame, end_frame, phn, *pos1, *pos2),...,] len=T_phn
    #f0: (T_frame, code_depth) 当输入code_f0时的shape，此时code_mode应为"triangular"，code_depth一般为4


    # 实际all_phon中有两个我们不会用到的phn，所以后续考虑优化
    all_phon = get_all_phon() # ['none', 'pau', 'br', 'sil', 'a', 'ch','iii',...] len=88

    # 验证最后一个phn的结束帧数与f0 curve的总帧数相等。
    # 否则，强行将最后一个phn的end frame改成f0长度
    if time_phon_list[-1][1] != len(f0):
        tup = time_phon_list[-1]
        tup_list = list(tup)
        tup_list[1] = len(f0)
        time_phon_list[-1] = tuple(tup_list)
    assert(time_phon_list[-1][1] == len(f0))

    # debug下验证结果需要用到label_list
    # label_list = []
    # cur_phn_oh_lists = []
    phn_condi_list = []
    for j, time_phon in enumerate(time_phon_list):

        cur_phn, cur_phn_syllable_pos, cur_phn_syllable_pos2, phn_count = \
            get_phon_condi_by_time_phon(time_phon)

        if phn_count <= 0:
            continue
        # debug下验证结果需要用到label_list
        # label_list.append([cur_phn, cur_phn_syllable_pos, f0_coarse[i]])

        # onehot
        cur_phn_oh = np.zeros(len(all_phon))
        cur_phn_syllable_pos2_oh = np.zeros(4)
        # f0_coarse_oh = np.zeros(f0_bin)

        cur_phn_oh[cur_phn] = 1
        # 将该标签进行one-hot处理
        cur_phn_syllable_pos2_oh[cur_phn_syllable_pos2] = 1

        phn_condi_list.append(
            np.concatenate(([phn_count],[cur_phn_syllable_pos],cur_phn_syllable_pos2_oh,      # 1+1+4+88
                            cur_phn_oh))) ######！！！！！！

        if j == len(time_phon_list) - 1:
            print('phn condition:', len(phn_condi_list[-1]), ' ', np.sum(phn_condi_list[-1]))

    return np.stack(phn_condi_list)
