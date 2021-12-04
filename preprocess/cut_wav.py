import os
import fnmatch
import soundfile as sf
import pretty_midi
import numpy as np

import sys
sys.path.append('..')

from config import *
import data_util.textgrid as textgrid
from data_util.pinyin_phone import get_all_phon

all_phn = get_all_phon()


def get_segs_from_txtgrd(raw_path):

    txtgrd_path = raw_path.replace('.wav', '.TextGrid')
    py_grid = textgrid.TextGrid.fromFile(txtgrd_path)

    source_tier = py_grid.tiers[0]
    assert source_tier

    source_intervals_li = list(source_tier)
    print(source_intervals_li[0].minTime)

    start_idices = []
    end_idices = []
    for i, interval in enumerate(source_intervals_li):

        before = None if i == 0 else source_intervals_li[i-1]
        after = None if i == len(source_intervals_li)-1 else source_intervals_li[i+1]

        if interval.mark not in ['pau', 'sil']:

            if interval.mark in ['br']:

                # after是真实音素时，加个start idx
                if after !=None and after.mark not in ['pau', 'sil', 'br']:
                    start_idices.append(i)
                # before是真实音素时，加个end idx
                if before != None and before.mark not in ['pau', 'sil', 'br']:
                    end_idices.append(i)
            else:

                if after == None or after.mark in ['pau', 'sil']:
                    end_idices.append(i)
                if before == None or before.mark in ['pau', 'sil']:
                    start_idices.append(i)

    # try to cut a longer piece and shorter than a fixed time
    shorter_than_second = 19

    piece_indices = []
    last_end_index = 0
    for start_index in start_idices:
        if start_index >= last_end_index:
            start_time = source_intervals_li[start_index].minTime
            for end_index in end_idices:
                end_time = source_intervals_li[end_index].maxTime
                if end_time > start_time:
                    if end_time - start_time < shorter_than_second:
                        last_end_index = end_index
                    else:
                        break

            piece_indices.append((start_index, last_end_index))

    segs = []
    for (s_id, e_id) in piece_indices:
        segs.append(source_intervals_li[s_id:e_id+1])

    return segs


def write_seg_txtgrd(segs, file_name, dist_folder, pad=None):
    for seg_i, seg in enumerate(segs):

        offset = seg[0].minTime
        seg_tier = textgrid.IntervalTier('phoneme')


        if pad:
            pad_interval_lenTime = pad
            seg_tier.add(0.0, pad_interval_lenTime, 'sil')
            offset -= pad_interval_lenTime


        for interval in seg:
            minTime = round(interval.minTime-offset, 5)
            maxTime = round(interval.maxTime-offset, 5)
            phn = interval.mark
            seg_tier.add(minTime, maxTime, phn)

        if pad:
            seg_tier.add(maxTime, maxTime+pad_interval_lenTime, 'sil')
        seg_grid = textgrid.TextGrid()
        seg_grid.append(seg_tier)
        write_path = os.path.join(dist_folder, file_name+str(seg_i)+'.TextGrid')
        seg_grid.write(write_path)


def write_seg_wav(segs, raw_path, file_name, dist_folder, pad=None):

    wav_path = raw_path

    y, sr = sf.read(wav_path)
    for seg_i, seg in enumerate(segs):
    # 写WAV
        start_time = seg[0].minTime
        stop_time = seg[-1].maxTime
        for interval in seg:
            if interval.mark in ['sil', 'pau']:
                s = int(sr*interval.minTime)
                e = int(sr*interval.maxTime)
                y[s:e] = 0.0

        start = int(start_time * sr)
        stop = int(stop_time * sr)

        new_path = os.path.join(dist_folder, file_name+str(seg_i)+'.wav')
        y_dist = y[start:stop]
        if pad:
            pad_array = np.zeros(int(pad*sr))
            y_dist = np.concatenate((pad_array, y_dist, pad_array))

        sf.write(new_path, y_dist, sr)


if __name__ == '__main__':
    # 功能:将原始的音频以及标注切割成一句一句的音频以及标注,并且保留音频与标注的对应关系

    # 切割音频以及标注后存放位置
    dist_folder = RAW_DATA_PATH
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    # 原始音频以及标注存放位置
    uncut_folder = os.path.join(ROOT_PATH,  'raw')

    pad = 0.5 # second
    supportedExtensions = '*.wav'
    for dirpath, dirs, files in os.walk(uncut_folder):
        for file in fnmatch.filter(files, supportedExtensions):

            file_name = file.replace('.wav', '')

            print('processing '+file_name)
            raw_path = os.path.join(dirpath, file)
            # midi_path = raw_path.replace('.wav', '.mid')
            segs = get_segs_from_txtgrd(raw_path)
            write_seg_txtgrd(segs, file_name, dist_folder, pad)
            write_seg_wav(segs, raw_path, file_name, dist_folder, pad)


    # debug用代码
    # path = os.path.join(ROOT_PATH, 'raw/49fushengweixie.wav')
    # segs = get_segs_from_txtgrd(path)
    # write_seg_txtgrd(segs, '49fushengweixie', os.path.join(ROOT_PATH, 'temp_test'), pad)
    # write_seg_wav(segs, path,'49fushengweixie', os.path.join(ROOT_PATH, 'temp_test'), pad )