from config import *
import pretty_midi
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pylab import rcParams
rcParams['figure.figsize'] = 20, 5


def f0_limiter(input, length=9, threshold=0.05, ratio=0.15):
    if len(input) < length + 1:
        return input

    extend = input[0] / input[length] - 1

    if extend < threshold and extend > -threshold:
        return input

    if extend > threshold:
        for i in range(length):
            # input[i] = input[i] * (1 - ratio)
            input[i] = input[length] + (input[i] - input[length]) * ratio
    if extend < -threshold:
        for i in range(length):
            # input[i] = input[i] * (1 + ratio)
            input[i] = input[length] + (input[i] - input[length]) * ratio
    return input


def low_pass_FIR(data):
    b = signal.firwin(15, 0.2)    # need more orders than IIR
    low_data = signal.lfilter(b, 1, data)
    return low_data


def low_pass_IIR(data):
    b, a = signal.butter(5, 0.15, btype='low', output='ba')
    low_data = signal.lfilter(b, a, data)
    return low_data


def add_vibrato(f0_input, long_note_positions):

    mod_freq = random.uniform(3.0, 6.0)
    f0_modulated = lfo(f0_input, mod_freq, depth=0.002)

    for long_note in long_note_positions:
        note_start = long_note[0]
        note_end = long_note[1]
        start, end = random_interval(note_start, note_end, len_factor=0.6)
        mod_freq = random.uniform(4.2, 6.3)
        mod_depth = random.uniform(0.011, 0.015)
        f0_modulated_long = vibrato(f0_input[start: end], mod_freq, mod_depth)
        f0_modulated[start: end] = f0_modulated_long

    return f0_modulated


def get_long_note_positions(time_phn_list, threshold=0.5):
    # threshold by sec
    long_note_positions = []
    long_note_threshold = threshold * sample_rate / hop
    for item in time_phn_list:
        start = item[0]
        end = item[1]
        dur = end - start
        if dur > long_note_threshold:
            long_note_positions.append([start, end])
    return long_note_positions


def fit_overshoot_and_preparation_curve(f0_input, window_len=11, order=3):
    if len(f0_input) < window_len:
        return f0_input
    f0_output = signal.savgol_filter(f0_input, window_len, order)
    return f0_output


def make_f0_whole(note_list):
    list_end_time = note_list[-1]['end_time']

    piece_pitches = []
    long_notes_position = []
    step_up_position = []
    step_down_position = []

    for i, note in enumerate(note_list):
        note_pitch_list = []

        start_time = note['start_time']
        end_time = note['end_time']
        note_pitch = note['pitch']

        if note_pitch == 0 and i < len(note_list) - 1:
            note_pitch = note_list[i+1]['pitch']

        if i > 0:
            if note_pitch > note_list[i - 1]['pitch']:
                step_up_position.append(start_time)
            if note_pitch < note_list[i - 1]['pitch']:
                step_down_position.append(start_time)

        note_pitch_list.append((start_time, note_pitch))

        pitch_bends = note['pitchBends']
        abs_pitch = note_pitch
        for item in pitch_bends:
            bend_time = item['time']
            bend_pitch = item['pitch']
            abs_pitch = note_pitch + bend_pitch
            act_time = bend_time

            if act_time >= start_time:
                note_pitch_list.append((act_time, abs_pitch))
            else:
                pass

        # add last, 因为每个piece内的note都是收首尾相接的,前一个end_time等于后一个start_time所以最后一个note再添加end_time处的pitch
        # if i == len(ace_piece) - 1:
        note_pitch_list.append((end_time, abs_pitch))
        piece_pitches.extend(note_pitch_list)

    f0_pitch = []
    f0_interval = hop_sec
    f0_len = int(round(list_end_time * sample_rate / hop))

    # 先做简单差值吧
    for i in range(f0_len):
        cur_time = i * f0_interval

        start_idx = 0
        for j, item in enumerate(piece_pitches):
            p_time = item[0]
            if j == 0 and p_time > cur_time:
                break
            if cur_time >= p_time:
                start_idx = j

        if start_idx < len(piece_pitches) - 2:
            end_idx = start_idx + 1

            pos = cur_time - piece_pitches[start_idx][0]

            if pos < 0:
                f0_pitch.append(piece_pitches[start_idx][1])
            else:
                time_band = piece_pitches[end_idx][0] - piece_pitches[start_idx][0]
                pitch_band = piece_pitches[end_idx][1] - piece_pitches[start_idx][1]

                pitch = pos / time_band * pitch_band + piece_pitches[start_idx][1]
                f0_pitch.append(pitch)
        else:
            f0_pitch.append(piece_pitches[start_idx][1])

    f0 = []
    for item in f0_pitch:
        f0_item = pretty_midi.note_number_to_hz(item - 12)
        f0.append(f0_item)

    # plt.plot(f0)
    # plt.show()

    note_freq = np.array(f0)
    # f0 = second_order_damping(note_freq, 0.0348, 0.5422)

    # the over-shoot extent is larger than that of preparation,
    # and the overshoot duration is shorter than that of preparation.
    for time_point in step_up_position:
        frame = round(time_point * sample_rate / hop)
        step_up_process(f0, frame, pre_step=3, pre_depth=0.01, over_step=2, over_depth=0.02)

    for time_point in step_down_position:
        frame = round(time_point * sample_rate / hop)
        step_down_process(f0, frame, pre_step=2, pre_depth=0.01, over_step=1, over_depth=0.02)

    return note_freq, np.array(f0)


def step_up_process(f0, frame, pre_step=3, pre_depth=0.03, over_step=2, over_depth=0.05):
    if frame - pre_step <= 0 or frame + over_step >= len(f0):
        return
    # preparation
    pre_depth_step = pre_depth / pre_step
    for i in range(pre_step):
        f0[frame - 1 - i] *= (1 - (i + 1) * pre_depth_step)
    # overshoot
    over_depth_step = over_depth / over_step
    for i in range(over_step):
        f0[frame + i] *= (1 + i * over_depth_step)


def step_down_process(f0, frame, pre_step=3, pre_depth=0.03, over_step=2, over_depth=0.05):
    if frame - pre_step <= 0 or frame + over_step >= len(f0):
        return
    # preparation
    pre_depth_step = pre_depth / pre_step
    for i in range(pre_step):
        f0[frame - 1 - i] *= (1 + (i + 1) * pre_depth_step)
    # overshoot
    over_depth_step = over_depth / over_step
    for i in range(over_step):
        f0[frame + i] *= (1 - i * over_depth_step)


def random_interval(start, end, len_factor=0.5):
    dur = end - start
    # start_new = round(start + random.uniform(0.4, 1.0) * dur * (1 - len_factor))
    start_new = round(start + random.random() * dur * (1 - len_factor) / 2)
    end_new = round(end - random.random() * dur * (1 - len_factor) / 2)
    if end_new == start_new:
        end_new = end_new + 1
    assert start_new >= start and end_new <= end
    return start_new, end_new


def vibrato(input, freq, depth_max=0.01):
    # period是vibrato的周期，单位s，数值越大颤音速度越慢
    # depth位vibrato的强度，0.02以下比较正常，可尝试
    phase = 0.0
    output = np.zeros(len(input))
    x = np.linspace(0, len(input), len(input))
    envelope = np.sin(np.pi / len(input) * x)
    depth = envelope * depth_max
    assert len(depth) == len(input)
    # freq = 1.0 / period
    for i in range(len(input)):
        if phase > 1:
            phase -= 1
        mod = np.sin(2 * np.pi * phase)
        output[i] = input[i] * (1 + mod * depth[i])
        # ( -1 , 1 )
        rand = (random.random() * 2 - 1) / 22
        phase += freq / float(sample_rate) * hop * (1 + rand)
    # plt.plot(envelope)
    # plt.show()
    # plt.plot(input)
    # plt.plot(output)
    # plt.show()
    return output


def lfo(input, freq, depth=0.01):
    # period是vibrato的周期，单位s，数值越大颤音速度越慢
    # depth位vibrato的强度，0.02以下比较正常，可尝试
    phase = 0.0
    output = np.zeros(len(input))
    # freq = 1.0 / period
    for i in range(len(input)):
        if phase > 1:
            phase -= 1
        mod = np.sin(2 * np.pi * phase)
        output[i] = input[i] * (1 + mod * depth)
        # ( -1 , 1 )
        rand = (random.random() * 2 - 1) / 22
        phase += freq / float(sample_rate) * hop * (1 + rand)
    return output


def wgn(x, snr=42):
    # snr为加入随机抖动的大小，数值越大随机性越小
    Ps = np.sum(abs(x) ** 2) / len(x)
    Pn = Ps / (10 ** ((snr / 10)))
    noise = np.random.randn(len(x)) * np.sqrt(Pn)
    signal_add_noise = x + noise
    return signal_add_noise
