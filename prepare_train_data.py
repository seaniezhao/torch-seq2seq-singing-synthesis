import random

from config import *
import fnmatch
import sys
from tqdm import tqdm
import numpy as np
from data_util.data_tools import process_wav, make_frame_condition, process_phon_label, add_syllable_pos, \
    add_syllable_pos2, make_phn_condition



def prepare_directory():
    to_prepares = [TRAIN_SP_PATH, TRAIN_AP_PATH, TRAIN_CONDITION_PATH, TRAIN_F0_PATH,
                   TEST_SP_PATH, TEST_AP_PATH, TEST_CONDITION_PATH, TEST_F0_PATH]

    for p in to_prepares:
        if not os.path.exists(p):
            os.makedirs(p)


def main():
    supported_extensions = '*.wav'
    wav_files = fnmatch.filter(os.listdir(RAW_DATA_PATH), supported_extensions)

    sp_min, sp_max = sys.maxsize, (-sys.maxsize - 1)
    ap_min, ap_max = sys.maxsize, (-sys.maxsize - 1)

    energy_min, energy_max = sys.maxsize, (-sys.maxsize - 1)
    # 为了获取sp ap最大值最小值, 先暂存然后在处理成条件

    data_to_save = []

    wrong_f0_num = 0

    f0_total_count = 0
    f0_total = 0
    for file in wav_files:
        file_name = os.path.splitext(os.path.basename(file))[0]
        print('processing: ', file_name)

        wav_path = os.path.join(RAW_DATA_PATH,  file)
        f0, code_f0, _sp, code_sp, _ap, code_ap, energy = process_wav(wav_path, f0_code_mode='triangular_basis')
        # 如果以下函数返回"wrong_f0"，说明其中有f0超出阈值，这时丢弃这条数据
        if f0 == 'wrong_f0':
            wrong_f0_num += 1
            continue
        v_uv = f0 > 0

        sum_f0 = np.sum(f0)
        count_f0 = np.count_nonzero(f0)
        f0_total += sum_f0
        f0_total_count += count_f0

        # 默认是 phn.TextGrid
        txt_path = os.path.join(RAW_DATA_PATH,   file_name + '-phn.TextGrid')
        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH,   file_name + '.TextGrid')
        if not os.path.isfile(txt_path):
            txt_path = os.path.join(RAW_DATA_PATH,   file_name + '_pinyin.TextGrid')
        if not os.path.isfile(txt_path):
            print("[Warning]   no found the TextGrid of " + wav_path)
            continue

        time_phon_list, _ = process_phon_label(txt_path)
        time_phon_list = add_syllable_pos(time_phon_list)
        time_phon_list = add_syllable_pos2(time_phon_list)

        for item in time_phon_list:
            if item[1] - item[0] <0:
                print("ERROR!! time_phon_list for: " + file_name)

        # maybe label error
        if time_phon_list[-1][1] - len(f0) >1:
            print("ERROR!! time_phon_list for: " + file_name)
            continue

        data_to_save.append(
            (file_name, time_phon_list, code_f0, code_sp, code_ap, f0, energy))

        _sp_min = np.min(code_sp)
        _sp_max = np.max(code_sp)

        sp_min = _sp_min if _sp_min < sp_min else sp_min
        sp_max = _sp_max if _sp_max > sp_max else sp_max

        _ap_min = np.min(code_ap)
        _ap_max = np.max(code_ap)

        ap_min = _ap_min if _ap_min < ap_min else ap_min
        ap_max = _ap_max if _ap_max > ap_max else ap_max

        _energy_min = np.min(energy)
        _energy_max = np.max(energy)

        energy_min = _energy_min if _energy_min < energy_min else energy_min
        energy_max = _energy_max if _energy_max > energy_max else energy_max
        #
        # 为了debug时不让程序把全部数据处理完而使用
        # if len(data_to_save) >= 32:
        #     break

    print('num of pieces with wrong f0: ', wrong_f0_num)

    np.save(os.path.join(DATA_ROOT_PATH, 'min_max_record.npy'),
            [sp_min, sp_max, ap_min, ap_max])

    average_f0 = f0_total/f0_total_count
    np.save(os.path.join(DATA_ROOT_PATH, 'average_f0.npy'),
            average_f0)

    np.save(os.path.join(DATA_ROOT_PATH, 'min_max_energy.npy'),
            [energy_min, energy_max])

    # 为了更好的优化模型, 还是手动控制测试集比较好
    test_names = [item[0] for item in random.choices(data_to_save, k=20)]

    total_count = 0
    test_count = 0
    for (file_name, time_phon_list, code_f0, code_sp, code_ap, f0, energy) in tqdm(data_to_save):
        total_count += 1

        try:
            phn_condi  = make_phn_condition(time_phon_list, code_f0)
            frame_condi = make_frame_condition(time_phon_list, code_f0)
        except:
            print("error")
            continue


        code_sp = (code_sp - sp_min) / (sp_max - sp_min) - 0.5
        code_ap = (code_ap - ap_min) / (ap_max - ap_min) - 0.5

        energy = (energy - energy_min) / (energy_max - energy_min)  # 0~1

        is_test = False
        for name in test_names:
            if name in file_name:
                is_test = True
                break

        if is_test:
            test_count += 1
            np.save(TEST_CONDITION_PATH + '/' + file_name + '_phn_condi.npy', phn_condi)
            np.save(TEST_CONDITION_PATH + '/' + file_name + '_frame_condi.npy', frame_condi)
            np.save(TEST_CONDITION_PATH + '/' + file_name + '_energy_condi.npy', energy)

            np.save(TEST_SP_PATH + '/' + file_name + '_sp.npy', code_sp)
            np.save(TEST_AP_PATH + '/' + file_name + '_ap.npy', code_ap)
            np.save(TEST_F0_PATH + '/' + file_name + '_f0.npy', f0)
        else:
            np.save(TRAIN_CONDITION_PATH + '/' + file_name + '_phn_condi.npy', phn_condi)
            np.save(TRAIN_CONDITION_PATH + '/' + file_name + '_frame_condi.npy', frame_condi)
            np.save(TRAIN_CONDITION_PATH + '/' + file_name + '_energy_condi.npy', energy)

            np.save(TRAIN_SP_PATH + '/' + file_name + '_sp.npy', code_sp)
            np.save(TRAIN_AP_PATH + '/' + file_name + '_ap.npy', code_ap)
            np.save(TRAIN_F0_PATH + '/' + file_name + '_f0.npy', f0)

    print('total count: ', total_count, ', test count: ', test_count)


if __name__ == '__main__':
   prepare_directory()
   main()
