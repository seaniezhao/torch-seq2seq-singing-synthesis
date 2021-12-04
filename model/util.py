import torch
import torch.nn.functional as F
import numpy as np


def pad(input_ele, max_length=None):
    if max_length:
        out_list = list()
        max_len = max_length
        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded
    else:
        out_list = list()
        max_len = max([input_ele[i].size(0)for i in range(len(input_ele))])

        for i, batch in enumerate(input_ele):
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
            out_list.append(one_batch_padded)
        out_padded = torch.stack(out_list)
        return out_padded


# 将一个矩阵中重复的vec去掉，返回新的矩阵以及 矩阵中每个vec 出现的次数
def squash(input_matrix):

    origin_len = len(input_matrix)

    vec_arr = []
    vec_arr_count = []
    temp_vec = np.array([])
    for vec in input_matrix:
        if not np.array_equal(temp_vec, vec):
            temp_vec = vec
            vec_arr.append(vec)
            vec_arr_count.append(1)
        else:
            vec_arr_count[-1] += 1

    vec_arr = np.array(vec_arr)
    vec_arr_count = np.array(vec_arr_count)
    vec_arr_count = np.expand_dims(vec_arr_count, axis=1)
    sum_len = np.sum(vec_arr_count)

    assert origin_len == sum_len

    return vec_arr, vec_arr_count


def get_s2s_singing_sinusoid_pos_in_x(pos_in_x):
    d_hid = 4

    def cal_angle(position, hid_idx):
        return 2 * np.pi * (position - (hid_idx - 1) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i)
                               for pos_i in pos_in_x])

    sinusoid_table = np.cos(sinusoid_table) / 2 + 0.5  # d

    return sinusoid_table
