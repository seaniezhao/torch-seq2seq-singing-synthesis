import data_util.textgrid as textgrid
from config import *
from data_util.pinyin_phone import get_all_phon, get_voice_plosives, get_pause
import fnmatch
import pickle

cdir = os.path.dirname(__file__)
pkl_file = open(os.path.join(cdir, 'average_phn_dur_dict.pkl'), 'rb')
average_phn_dur_dict = pickle.load(pkl_file)

a = sorted(average_phn_dur_dict.items(), key=lambda x: x[1])


def get_phn_dur_in_note(c_phn, v_phn, note_dur):
    c_avg_dur = 10

    v_proportion = 0.7
    if c_phn in get_voice_plosives():
        v_proportion = 0.8

    # 如果另一个是pau或者sil等不发音的,则辅音可以全部占满
    if v_phn in get_pause():
        v_proportion = 0


    if c_phn in average_phn_dur_dict.keys():
        c_avg_dur = average_phn_dur_dict[c_phn]

    temp = (note_dur-v_proportion*note_dur)/c_avg_dur
    rc = min([temp, 1])

    c_dur = max([1, int(round(rc*c_avg_dur))])

    return c_dur


def get_phn_dur(label_path):

    py_grid = textgrid.TextGrid.fromFile(label_path)

    source_tier = None
    for tier in py_grid.tiers:
        if tier.name == 'phoneme':  # 人工标注的
            source_tier = tier
            break
        elif tier.name == 'phones':  # Montreal-Forced-Aligner 标注
            source_tier = tier
            break

    assert source_tier

    phn_dur_list = []

    for i, interval in enumerate(source_tier):
        phn = interval.mark.strip()

        dur = int(round(interval.maxTime*sample_rate/hop)) - int(round(interval.minTime*sample_rate/hop))
        if dur <= 0:
            dur = 1

        assert (phn in get_all_phon())

        phn_dur_list.append((phn, dur))

    return phn_dur_list


def calculate_average_phn_dur():
    supported_extensions = '*.wav'
    wav_files = fnmatch.filter(os.listdir(RAW_DATA_PATH), supported_extensions)

    phn_dur_dict = {}
    for file in wav_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        print('processing: ', file_name)

        # 默认是 phn.TextGrid
        txt_path = os.path.join(RAW_DATA_PATH, file_name + '-phn.TextGrid')
        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH, file_name + '.TextGrid')
        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH, file_name + '_pinyin.TextGrid')
        if not os.path.isfile(txt_path):
            print("[Warning]   no found the TextGrid of " + file_name)
            continue

        phn_dur = get_phn_dur(txt_path)
        for item in phn_dur:
            phn = item[0]
            dur = item[1]

            if phn in phn_dur_dict.keys():
                phn_dur_dict[phn][0] += dur
                phn_dur_dict[phn][1] += 1
            else:
                phn_dur_dict[phn] = [dur, 1]

    average_phn_dur_dict = {}
    for key in phn_dur_dict.keys():
        average_phn_dur_dict[key] = phn_dur_dict[key][0]/phn_dur_dict[key][1]

    return average_phn_dur_dict


if __name__ == '__main__':
    average_phn_dur_dict = calculate_average_phn_dur()
    with open('average_phn_dur_dict.pkl', 'wb') as handle:
        pickle.dump(average_phn_dur_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    pass
