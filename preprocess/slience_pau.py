import fnmatch
import os
import data_util.textgrid as textgrid
from data_util.pinyin_phone import get_all_phon
import soundfile as sf


def slience_pau(wav_path, txt_path):

    py_grid = textgrid.TextGrid.fromFile(txt_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':  # 人工标注的
            source_tier = tier
            break
        elif tier.name == 'phones':  # Montreal-Forced-Aligner 标注
            source_tier = tier
            break

    y, sr = sf.read(wav_path)

    for interval in source_tier:
        if interval.mark not in get_all_phon():
            print(interval.mark, "||", wav_path)
    # 将pau 和 br都静音
        if interval.mark in ['br', 'pau']:
            i_start = int(interval.minTime*sr)
            i_end = int(interval.maxTime*sr)
            y[i_start:i_end] = 0

    sf.write(wav_path, y, sr)


if __name__ == '__main__':
    # 功能:将原始的音频中标注为br和pau的部分静音

    uncut_folder = './data/raw'

    supportedExtensions = '*.wav'
    for dirpath, dirs, files in os.walk(uncut_folder):
        for file in fnmatch.filter(files, supportedExtensions):

            file_name = file.replace('.wav', '')

            #print('processing '+file_name)
            wav_path = os.path.join(dirpath, file)
            txt_path = wav_path.replace('.wav', '-phn.TextGrid')

            slience_pau(wav_path, txt_path)
