import os
import pyworld as pw

hop_sec = 2 * pw.default_frame_period / 1000
sample_rate = 32000
hop = int(hop_sec * sample_rate)
fft_size = 2048
f0_bin = 256
f0_max = 1100.0
f0_min = 125.0


ROOT_PATH = '/Users/zhaowenxiao/PythonProj/torch-seq2seq-singing-synthesis/data_root'


# ------------path config------------------
RAW_DATA_PATH = os.path.join(ROOT_PATH, 'raw_piece')
DATA_ROOT_PATH = os.path.join(ROOT_PATH, 'dataset')
GEN_PATH = os.path.join(ROOT_PATH, 'gen')


# 测试数据集的目录
TEST_ROOT_PATH = os.path.join(
    DATA_ROOT_PATH, 'test')  # 测试数据的根目录

TEST_SP_PATH = os.path.join(TEST_ROOT_PATH,  'sp')  # sp测试数据据集
TEST_AP_PATH = os.path.join(TEST_ROOT_PATH,  'ap')  # ap测试数据集
TEST_VUV_PATH = os.path.join(TEST_ROOT_PATH,  'vuv')  # vuv测试数据集
TEST_CONDITION_PATH = os.path.join(
    TEST_ROOT_PATH,  'condition')  # condition数据集
TEST_F0_PATH = os.path.join(TEST_ROOT_PATH,  'f0')  # f0数据集
TEST_F0_CONDITION_PATH = os.path.join(TEST_ROOT_PATH,  'f0_condition')  # f0条件
TEST_phn_PATH = os.path.join(TEST_ROOT_PATH,  'phn')  # phn onehot 序列

# 训练集的数据目录
TRAIN_ROOT_PATH = os.path.join(
    DATA_ROOT_PATH, 'train')  # 训练数据的根目录

TRAIN_SP_PATH = os.path.join(TRAIN_ROOT_PATH,  'sp')  # sp测试数据据集
TRAIN_AP_PATH = os.path.join(TRAIN_ROOT_PATH,  'ap')  # ap测试数据集
TRAIN_VUV_PATH = os.path.join(TRAIN_ROOT_PATH,  'vuv')  # vuv测试数据集
TRAIN_CONDITION_PATH = os.path.join(
    TRAIN_ROOT_PATH,  'condition')  # condition数据集
TRAIN_F0_PATH = os.path.join(TRAIN_ROOT_PATH,  'f0')  # f0数据集
TRAIN_F0_CONDITION_PATH = os.path.join(TRAIN_ROOT_PATH,  'f0_condition')  # f0条件
TRAIN_phn_PATH = os.path.join(TRAIN_ROOT_PATH,  'phn')  # phn onehot 序列

SNAOSHOTS_ROOT_PATH = os.path.join('.', 'snapshots')

# ------------path config------------------