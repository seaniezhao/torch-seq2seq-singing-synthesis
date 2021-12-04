import copy
import json

import requests

from data_util.pinyin_phone import get_shengmu, get_phoneme
from config import *


def extract_notes_by_load_ace_v1(pau_mark, path=None, file_url=None,
                                 max_piece_time=None, pau_time=None, pau_min=0.05):#郭靖改：将pau_min改为0.05，原因是标注文档中的标准为，0.05以下sp会被音素瓜分
    """
    1.先将工程文件字处理成note
        1.1每个note由本个字的元音和下个字的辅音组成
        1.2停顿部分用pau表示
    :param pau_mark:
    :param path: 资源文件路径
    :param file_url: 资源地址路径
    :param max_piece_time: 最大片段时长
    :param pau_time: 初始位置暂停时间
    :param pau_min: 认为是暂停最小间隔
    :return:
    """

    if not pau_time:
        pau_time = 1

    if file_url:
        load_dict = requests.get(file_url).json()
        # 由于ios上传文件后缀为.json做下处理
        file_name = file_url.split("/")[-1].replace('json', 'ace')
    else:
        file_name = path.split("/")[-1]
        with open(path, 'r') as load_f:
            load_dict = json.load(load_f)

    pau_template = {
        "br": False,
        "end_time": 1,
        "key": "B",
        "pinyin": pau_mark,
        "phoneme": [pau_mark],
        "pitch": 0,
        "pitchBends": [],
        "scale": [],
        "start_time": 0,
        "word": "空"
    }

    notes = load_dict["notes"]
    role_id = load_dict["role_info"]["role_id"]
    if load_dict.get("song_info"):
        notes_duration = load_dict["song_info"]["duration"]
        bpm = load_dict["song_info"]["bpm"]
    else:
        notes_duration = 120
        bpm = 120
    # notes_duration = load_dict["song_info"]["duration"]
    # bpm = load_dict["song_info"]["bpm"]
    note_list = []
    pau_note = copy.deepcopy(pau_template)
    notes.insert(0, pau_note)

    # 先对不正规的ace文件做处理
    # 1.处理ace文件中end_time为0的情况
    for n, note in enumerate(notes):
        if note["end_time"] == 0:
            if n == len(notes) - 1:
                note["end_time"] = min(notes_duration, note["start_time"] + 60 / bpm)
            else:
                note["end_time"] = notes[n + 1]["start_time"]

    notes_new = []
    for n, note in enumerate(notes):
        # if n == 112:
        #     print(1)
        note = notes[n]
        start_time = note['start_time']
        end_time = note['end_time']
        start_frame = int(round(start_time * sample_rate / hop))
        dur_time = end_time - start_time
        if n == 0:
            notes_new.append(note)
        else:
            last_note = notes_new[-1]
            last_note_start_time = last_note['start_time']
            last_note_end_time = last_note['end_time']
            last_note_end_frame = int(round(last_note_end_time*sample_rate/hop))
            last_note_dur_time = last_note_end_time - last_note_start_time

            # 与上一个有间隔且上一个过小，与上一个没间隔且自己过小，与上一个没间隔且上一个过小，都不是
            if last_note_end_frame != start_frame:
                if last_note_dur_time < 3*hop_sec:
                    notes_new = notes_new[:-1]
                    notes_new.append(note)
                else:
                    notes_new.append(note)
            else:
                if dur_time < 3*hop_sec:
                    last_note['end_time'] = note['end_time']
                elif last_note_dur_time < 3*hop_sec:
                    note['start_time'] = last_note['start_time']
                    notes_new[-1] = note
                else:
                    notes_new.append(note)
    last_one_note = notes_new[-1]
    last_one_note_start_time = last_one_note['start_time']
    last_one_note_end_time = last_one_note['end_time']
    last_dur = last_one_note_end_time - last_one_note_start_time
    if last_dur < 3*hop_sec:
        notes_new = notes_new[:-1]


    for n, note in enumerate(notes_new):
        if n == 112:
            print(1)
        phoneme = get_phoneme(note["pinyin"])
        # 解决ace文件end_time为0的情况
        if n == 0:
            # 判断第一个note是否有辅音
            phoneme = get_phoneme(notes_new[1]["pinyin"])
            if len(phoneme) == 1:
                # 第一个note无辅音
                note["start_time"] = notes_new[1]["start_time"] - pau_time
                note["end_time"] = notes_new[1]["start_time"]

            else:
                # 第一个note有辅音
                consonant_time = 0
                # 0310改：删除下面这句
                # note["phoneme"] = phoneme
                # phoneme = phoneme[:1]
                note["phoneme"] = [pau_mark, phoneme[0]]
                note["start_time"] = notes_new[1]["start_time"] - pau_time
                # 如果辅音时间>起止时间 取起止时间的一半
                duration = notes_new[1]["end_time"] - notes_new[1]["start_time"]

                # 0319
                # if consonant_time >= duration:
                #     note["end_time"] = notes[1]["start_time"] + duration / 2
                # else:
                #     note["end_time"] = notes[1]["start_time"] + consonant_time
                if duration - consonant_time < 1.01 * hop_sec:
                    consonant_time = duration - 1.01 * hop_sec
                note["end_time"] = notes_new[1]["start_time"] + consonant_time


            ## 0310改：不再丢弃开始时间为负的note
            # # 判断如果加的pau开始时间<0则丢弃
            # if note["start_time"] < 0:
            #     note["start_time"] = 0
            # else:
            #   note_list.append(note)
            note_list.append(note)

        # 判断如果是最后一个
        elif len(notes_new) == n + 1:
            # 判断当前note是否有辅音
            note["phoneme"] = phoneme[-1:]
            consonant_time = 0
            if len(phoneme) != 1:
                # 如果有辅音
                # 如果辅音时间>起止时间 取起止时间的一半
                duration = note["end_time"] - note["start_time"]

                # 0319
                # if consonant_time >= duration:
                #     note["start_time"] = note["start_time"] + duration / 2
                # else:
                #     note["start_time"] = note["start_time"] + consonant_time
                if duration - consonant_time < 1.01 * hop_sec:
                    consonant_time = duration - 1.01 * hop_sec
                note["start_time"] = note["start_time"] + consonant_time

            note_list.append(note)

        else:
            phoneme_next = get_phoneme(notes_new[n + 1]["pinyin"])
            # 判断两个note是否有间隔如果有间隔插入pau
            if notes_new[n + 1]["start_time"] - note["end_time"] > pau_min:
                # 如果有间隔
                # 先处理当前note
                # 判断当前note是否有辅音
                if len(phoneme) == 1:
                    # 如果没有辅音
                    note["phoneme"] = phoneme
                    note_list.append(note)
                else:

                    # 如果有辅音
                    consonant_time = 0
                    # 如果辅音时间>起止时间 取起止时间的一半
                    duration = note["end_time"] - note["start_time"]

                    # 0319
                    # if consonant_time >= duration:
                    #     note["start_time"] = note["start_time"] + duration / 2
                    # else:
                    #     note["start_time"] = note["start_time"] + consonant_time
                    if duration - consonant_time < 1.01 * hop_sec:
                        consonant_time = duration - 1.01 * hop_sec
                    note["start_time"] = note["start_time"] + consonant_time

                    note["phoneme"] = phoneme[-1:]
                    note_list.append(note)

                # 再补充空隙
                # 判断下一个note是否有辅音
                if len(phoneme_next) == 1:
                    # 下一个note无辅音
                    pau_note = copy.deepcopy(pau_template)
                    pau_note["start_time"] = note["end_time"]
                    pau_note["end_time"] = notes_new[n + 1]["start_time"]
                    pau_note["phoneme"] = [pau_mark]
                    note_list.append(pau_note)

                else:
                    pau_note = copy.deepcopy(pau_template)
                    # 下一个note有辅音
                    consonant_time = 0
                    # 如果辅音时间>起止时间 取起止时间的一半
                    duration = notes_new[n + 1]["end_time"] - notes_new[n + 1]["start_time"]
                    pau_note["start_time"] = note["end_time"]

                    #0319
                    # if consonant_time >= duration:
                    #     pau_note["end_time"] = notes[n + 1]["start_time"] + duration / 2
                    # else:
                    #     pau_note["end_time"] = notes[n + 1]["start_time"] + consonant_time
                    if duration - consonant_time < 1.01 * hop_sec:
                        consonant_time = duration - 1.01 * hop_sec
                    pau_note["end_time"] = notes_new[n + 1]["start_time"] + consonant_time

                    pau_note["phoneme"] = [pau_mark, phoneme_next[0]]
                    note_list.append(pau_note)

            else:
                # 如果没有间隔
                # 当前note是否有辅音
                if len(phoneme) == 1:
                    # 如果当前note无辅音
                    # 判断下一个note是否有辅音
                    if len(phoneme_next) == 1:
                        # 如果下一个note无辅音
                        note["phoneme"] = [phoneme[-1]]
                        ## 郭靖改：这里应该添加以下代码，否则间隔小于阈值且下一个note无辅音时，间隔没有被处理（给到上一个note）
                        note['end_time'] = notes_new[n + 1]['start_time']
                    else:
                        # 如果下一个note有辅音
                        note["phoneme"] = [phoneme[-1], phoneme_next[0]]
                        consonant_time = 0
                        # 如果辅音时间>起止时间 取起止时间的一半
                        duration = notes_new[n + 1]["end_time"] - notes_new[n + 1]["start_time"]

                        #0319
                        # if consonant_time >= duration:
                        #     note["end_time"] = notes[n + 1]["start_time"] + duration / 2
                        # else:
                        #     note["end_time"] = notes[n + 1]["start_time"] + consonant_time
                        if duration - consonant_time < 1.01 * hop_sec:
                            consonant_time = duration - 1.01 * hop_sec
                        note["end_time"] = notes_new[n + 1]["start_time"] + consonant_time

                else:
                    # 如果当前note有辅音
                    consonant_time = 0
                    # 如果辅音时间>起止时间 取起止时间的一半
                    duration = note["end_time"] - note["start_time"]

                    #0319
                    # if consonant_time >= duration:
                    #     note["start_time"] = note["start_time"] + duration / 2
                    # else:
                    #     note["start_time"] = note["start_time"] + consonant_time
                    if duration - consonant_time < 1.01 * hop_sec:
                        consonant_time = duration - 1.01 * hop_sec
                    note["start_time"] = note["start_time"] + consonant_time

                    # 判断下一个note是否有辅音
                    if len(phoneme_next) == 1:
                        # 如果下一个note无辅音
                        note["phoneme"] = [phoneme[-1]]
                        ## 郭靖改：这里应该添加以下代码，否则间隔小于阈值且下一个note无辅音时，间隔没有被处理（给到上一个note）
                        note['end_time'] = notes_new[n+1]['start_time']

                    else:
                        # 如果下一个note有辅音
                        note["phoneme"] = [phoneme[-1], phoneme_next[0]]
                        consonant_time = 0
                        # 如果辅音时间>起止时间 取起止时间的一半
                        duration = notes_new[n + 1]["end_time"] - notes_new[n + 1]["start_time"]

                        # 0319
                        # if consonant_time >= duration:
                        #     note["end_time"] = notes[n + 1]["start_time"] + duration / 2
                        # else:
                        #     note["end_time"] = notes[n + 1]["start_time"] + consonant_time
                        if duration - consonant_time < 1.01 * hop_sec:
                            consonant_time = duration - 1.01 * hop_sec
                        note["end_time"] = notes_new[n + 1]["start_time"] + consonant_time

                note_list.append(note)

    # 0311
    # 把过小的note整合在一起
    # notes_list_new = []
    # n=0
    # while n < len(note_list):
    #     note = note_list[n]
    #     start_time = note['start_time']
    #     end_time = note['end_time']
    #     start_frame = int(round(start_time*sample_rate/hop))
    #     end_frame = int(round(end_time*sample_rate/hop))
    #     dur_frame = end_frame - start_frame
    #     if n == 0:
    #         notes_list_new.append(note)
    #     else:
    #         last_note = notes_list_new[-1]
    #         last_note_start_time = last_note['start_time']
    #         last_note_end_time = last_note['end_time']
    #         last_note_start_frame = int(round(last_note_start_time*sample_rate/hop))
    #         last_note_end_frame = int(round(last_note_end_time*sample_rate/hop))
    #         last_note_dur_frame = last_note_end_frame - last_note_start_frame
    #         if dur_frame<2:
    #             last_note['end_time'] = note['end_time']
    #         elif dur_frame >= 2 and last_note_dur_frame < 2:
    #             note['start_time'] = last_note['start_time']
    #             notes_list_new[-1] = note
    #         else:
    #             notes_list_new.append(note)
    #     n += 1


    return note_list, file_name, notes_duration


def cut_notes_v1(notes, pau_mark="pau", frame_limit=1000):
    # print("""
    # 每片帧数间隔不能超过设置的阈值
    # 切片规则 pau>br>声母>韵母
    # 单个帧超过阈值切成多片
    # :param notes: [(2981, 2995, 'br'), (2995, 3095, 'pau'),...]
    # :param pau_mark:
    # :param frame_limit: 每片帧数限制
    # :return: [[(2981, 2995, 'br'), (2995, 3095, 'pau')],[...]...]
    # """)
    sm = get_shengmu()
    first_start_frame = notes[0][0]
    piece = []
    pieces = []
    for n, note in enumerate(notes):
        start_frame = note[0]
        end_frame = note[1]
        # 如果加上该note 片段没超过最大片段限定帧数加上该note
        if note[1] - first_start_frame <= frame_limit:
            # 处理是最后一个note的情况
            if n == len(notes) - 1:
                piece.append(note)
                pieces.append(piece.copy())
                break
            else:
                piece.append(note)
        # 如果加上该note 片段超过限定帧数做以下处理
        else:
            # 如果加上该note 片段超过了最大片段限定帧数 单个note超过限定帧数从该处进行切割
            if note[1] - note[0] > frame_limit:
                if note[2] != pau_mark:
                    end_time = first_start_frame + frame_limit
                    if n == 0:
                        piece.append([first_start_frame, end_time, note[2]])
                    else:
                        piece.append([notes[n - 1][1], end_time, note[2]])
                    first_start_frame = end_time
                    pieces.append(piece.copy())
                    piece.clear()
                    # 处理是最后一个note情况
                    if n != len(notes) - 1:
                        first_start_frame = notes[n + 1][0]
                    # 处理超出情况
                    temp = [end_time, note[1], note[2]]
                    for num in range(int((note[1] - end_time) // frame_limit)):
                        temp[0] = end_time + num * frame_limit
                        temp[1] = end_time + (num + 1) * frame_limit
                        piece.append(tuple(temp.copy()))
                        pieces.append(piece.copy())
                        piece.clear()
                    # 处理不能被整除部分放到下一个片段
                    dua = (note[1] - end_time) % frame_limit
                    if dua != 0:
                        temp[0] = end_frame - dua
                        temp[1] = end_frame
                        piece.append(tuple(temp.copy()))
                        # 处理是最后一个note情况
                        if n == len(notes) - 1:
                            pieces.append(piece)
                            break
                        else:
                            first_start_frame = end_frame - dua

                else:
                    # 如果暂停时间超过限定长度
                    pieces.append(piece.copy())
                    piece.clear()
                    piece.append(note)
                    first_start_frame = note[0]

            # 如果加上该note片段超过最大片段限定帧数 但单个note未超最大片段限定帧数 先结束片段
            else:
                # 按照规则优先找切割位置
                # todo 处理是最后一个note的情况
                if n == len(notes) - 1:
                    # 如果前个note是声母 把该note放到最后-片
                    if piece[-1][2] in sm:
                        # 如果前个声母和最后的韵母加起来超过片段限定帧数 用补齐的方式将最后的韵母切割两部分补齐上个片段余者放到下个片段
                        if note[1] - piece[-1][0] > frame_limit:
                            split_note_none = (notes[-2][1], first_start_frame + frame_limit, note[2])

                            piece.append(split_note_none)
                            pieces.append(piece.copy())
                            piece.clear()
                            split_note_two = (first_start_frame + frame_limit, note[1], note[2])
                            piece.append(split_note_two)
                            pieces.append(piece.copy())
                            print("韵母补齐声母,剩余韵母放到最后片段")
                        else:
                            pieces.append(piece.copy()[:-1])
                            piece.clear()
                            piece.append(notes[-2])
                            piece.append(note)
                            pieces.append(piece.copy())
                            print("声母和韵母一起放到最后片段")
                    else:
                        pieces.append(piece.copy())
                        piece.clear()
                        piece.append(note)
                        pieces.append(piece.copy())
                        print("韵母放到最后片段")
                    break

                # 倒叙找到pau位置做切割
                p = copy.deepcopy(piece)
                p.reverse()
                piece.reverse()
                mark = True
                for x, x_note in enumerate(p):
                    if x == len(piece) - 1:
                        # piece.reverse()
                        # pieces.append(piece.copy())
                        # piece.clear()
                        break

                    if piece[x][2] == pau_mark:
                        pieces.append(piece.copy()[x + 1:][::-1])
                        x_note_list = piece.copy()[:x + 1][::-1]
                        piece.clear()
                        piece = x_note_list
                        mark = False
                        break
                if mark:
                    for x, x_note in enumerate(p):
                        if x == len(piece) - 1:
                            # piece.reverse()
                            # pieces.append(piece.copy())
                            # piece.clear()
                            break
                        if piece[x][2] == "br":
                            pieces.append(piece.copy()[x + 1:][::-1])
                            x_note_list = piece.copy()[:x + 1][::-1]
                            piece.clear()
                            piece = x_note_list
                            mark = False
                            break
                if mark:
                    for x, x_note in enumerate(p):
                        if x == len(piece) - 1:
                            piece.reverse()
                            pieces.append(piece.copy())
                            piece.clear()
                            break
                        if piece[x][2] in sm:
                            pieces.append(piece.copy()[x + 1:][::-1])
                            x_note_list = piece.copy()[:x + 1][::-1]
                            piece.clear()
                            piece = x_note_list
                            mark = False
                            break

                if len(piece) > 0:
                    first_start_frame = piece[0][0]
                else:
                    first_start_frame = note[0]
                # 解决切一刀之后加note超出的情况
                if note[1] - first_start_frame > frame_limit:
                    # pieces.append(piece.copy())
                    # piece.clear()
                    p1 = copy.deepcopy(piece)
                    p1.reverse()
                    piece.reverse()
                    for x, x_note in enumerate(p1):
                        if x == len(piece) - 1:
                            piece.reverse()
                            pieces.append(piece.copy())
                            piece.clear()
                            break
                        if piece[x][2] in sm:
                            pieces.append(piece.copy()[x + 1:][::-1])
                            x_note_list = piece.copy()[:x + 1][::-1]
                            piece.clear()
                            piece = x_note_list
                            break
                    if len(piece) > 0:
                        first_start_frame = piece[0][0]
                    else:
                        first_start_frame = note[0]
                piece.append(note)
                
                

    # 剔除只含pau的片
    pieces_deal = copy.deepcopy(pieces)
    for piece in pieces:
        n = 0
        for note in piece:
            if note[2] == pau_mark:
                n += 1
        if len(piece) == n:
            pieces_deal.remove(piece)
                    
    return pieces_deal


if __name__ == '__main__':
    print()