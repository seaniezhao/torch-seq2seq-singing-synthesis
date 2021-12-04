import pandas as pd
import numpy as np
import pickle
import json
import os
from pypinyin import lazy_pinyin

cdir = os.path.dirname(__file__)
# with open(os.path.join(cdir, 'pinyin.json'), "r") as f:
#     pp_dict = json.loads(f.read())

# todo delete key "PIY"
pkl_file_name = 'pinyin-phoneme.pkl'

with open(os.path.join(cdir, pkl_file_name), "rb") as f:
    pp_dict = pickle.load(f)

pp_dict_reverse = {}
for pinyin in pp_dict:
    pp_dict_reverse[str(pp_dict[pinyin])] = pinyin


# with open("pinyin.json", "r") as f:
#     sy_obj = json.loads(f.read())
#
# for item in pp_dict:
#     if item not in sy_obj:
#         print("====="+item+"||"+str(pp_dict[item]))
#     elif pp_dict[item] != sy_obj[item]:
#         print(item+'  '+str(pp_dict[item])+'||'+str(sy_obj[item]))


def get_phoneme(pinyin):

    if pinyin in pp_dict:
        phns = []
        ph = pp_dict[pinyin]
        for p in ph:
            phns.append(p)
        return phns
    else:
        return [pinyin]


def get_yun_mu(pinyin):
    """
    获取韵母,没韵母返回拼音
    :param pinyin:
    :return:
    """
    try:
        syllable = pp_dict[pinyin]
        # 只需要韵母
        return syllable[-1]
    except KeyError:
        print("对照表中没有该拼音，需要拓展拼音韵母对照表")
        return pinyin


def get_pinyins(characters):
    return_pinyins = []

    try:
        pinyins = lazy_pinyin(characters)
        for pinyin in pinyins:
            if pinyin in pp_dict:
                return_pinyins.append(pinyin)
    except Exception as e:
        print(e)

    return return_pinyins


def Is_pinyin(pinyin):
    if pinyin in pp_dict:
        return True
    return False


# 不发声,元音,辅音
def get_phn_class(phn):
    if phn.strip() in ['pau', 'sli', 'br', 'sp', '#']:
        return 0
    elif phn in ['k', 'p', 's', 'h', 't', 'j', 'c', 'b', 'z', 'm', 'g', 'l',
                 'd', 'ch', 'zh', 'x', 'q', 'sh', 'f', 'n', 'r', 'pl']:
        return 0
    else:
        return 2


def get_all_phon():
    all_phon = ['none', 'pau', 'br', 'sil', 'sp', 'uar']
    for k, v in pp_dict.items():
        for phn in v:
            if phn not in all_phon:
                all_phon.append(phn)

    return all_phon


def get_shengmu():
    sheng_mu=[]
    for pinyin in pp_dict:
        if len(pp_dict[pinyin]) > 1:
            sheng_mu.append(pp_dict[pinyin][0])

    sheng_mu = set(sheng_mu)

    return sheng_mu


def get_voice_plosives():
    return ['b', 'd', 'g']


def get_plosives():
    return ['b', 'd', 'g', 'p', 't', 'k']


def get_sonorants():
    # r added according to label
    return ['m', 'n', 'l', 'r']


def get_pause():
    return ['none', 'pau', 'sil', 'sp']


def add_pinyin_phn(pinyin, phn_list):
    if pinyin not in pp_dict:
        pp_dict[pinyin] = phn_list
        print("pinyin added for "+pinyin+': '+str(phn_list))
    else:
        pp_dict[pinyin] = phn_list
        print("pinyin modified for " + pinyin + ': ' + str(phn_list))

    with open(os.path.join(cdir, pkl_file_name), "wb") as f:
        pickle.dump(pp_dict, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    #add_pinyin_phn('kei', ['k', 'ei'])
    #add_pinyin_phn('yo', ['io'])
    print(get_phoneme('yo'))

