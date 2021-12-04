import pretty_midi
from config import *
import random
import numpy as np
from data_util.pinyin_phone import get_phoneme


# 训练数据可以直接使用这个获取notelist,因为训练数据的midi提前处理过
def get_midi_notes(path):
    # Load MIDI file into PrettyMIDI object
    midi_data = pretty_midi.PrettyMIDI(path)

    note_list = []
    assert len(midi_data.instruments) >= 1
    # 若多于一轨只取一轨
    instrument = midi_data.instruments[0]
    # print(instrument.name)
    for note in instrument.notes:
        # print(note.start, note.end, note.pitch, (note.end - note.start))
        # print(int(round(note.start*sample_rate/hop)), int(round(note.end*sample_rate/hop)), note.pitch)
        int_start = int(round(note.start*sample_rate/hop))
        int_end = int(round(note.end*sample_rate/hop))
        int_dur = int_end - int_start

        note_list.append((int_start, int_end, note.pitch, int_dur))

    return note_list


# 新来一个midi文件用来生成f0,需要用这个函数处理一下,因为一般的midi文件音与音之间可能重叠或者有空隙,还要增加开头结尾
def preprocess_midi(midi_path):
    # 先把没有音的空当用0填上
    note_list = get_midi_notes(midi_path)
    note_num = len(note_list)

    for i in range(note_num):
        if i < note_num-1:
            current_end = note_list[i][1]
            next_start = note_list[i+1][0]
            # 有空隙
            if current_end < next_start:
                note_list.append((current_end, next_start, 0, next_start-current_end))
            # 有重叠
            elif current_end > next_start:
                current_note = list(note_list[i])
                current_note[1] = next_start
                note_list[i] = tuple(current_note)

    # 增加开头结尾
    add_frame = 100
    for i, note in enumerate(note_list):
        l_note = list(note)
        l_note[0] += add_frame
        l_note[1] += add_frame
        note_list[i] = tuple(l_note)
    note_list.append((0, 100, 0, 100))
    # 排序
    note_list.sort(key=lambda note: note[0])
    last = note_list[-1][1]
    note_list.append((last, last+add_frame, 0, add_frame))

    return note_list


# 通过midi 填词
def make_phn_from_midi(note_list, pinyins):
    # 每个音一个词不够补"啦"
    res = len(note_list) - len(pinyins)
    if res > 0:
        for x in range(res):
            pinyins.append('la')

    time_phon_list = []

    pinyin_index = 0
    for note in note_list:
        note_start = note[0]
        note_end = note[1]
        note_pitch = note[2]
        note_dur = note[3]

        if note_pitch == 0:
            phones = ['pau']
        else:
            note_pinyin = pinyins[pinyin_index]
            pinyin_index += 1
            phones = get_phoneme(note_pinyin)

        if len(phones) == 1:
            time_phon_list.append((note_start, note_end, phones[0]))
        else:
            assert len(phones) == 2
            p_dur = 0
            if note_dur < 30:
                rand = random.uniform(0.15, 0.4)
                p_dur = int(note_dur * rand)
            else:
                rand = random.gauss(15, 3)
                p_dur = np.clip(int(rand), 10, 30)
            time_phon_list.append((note_start, note_start + p_dur, phones[0]))
            time_phon_list.append((note_start + p_dur, note_end, phones[1]))

    return time_phon_list
