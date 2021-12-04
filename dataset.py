import numpy as np
from torch.utils.data import Dataset
import os
import model.util as util
import hparams as hp

class FastnpssDataset(Dataset):

    # 为加快速度先全部放到内存
    def __init__(self, data_folder, train=True):

        if train:
            data_folder = os.path.join(data_folder, 'train')
        else:
            data_folder = os.path.join(data_folder, 'test')

        self.sp_folder = os.path.join(data_folder, 'sp')
        self.ap_folder = os.path.join(data_folder, 'ap')
        self.f0_folder = os.path.join(data_folder, 'f0')
        self.condi_folder = os.path.join(data_folder, 'condition')

        self.dataset_files = []
        dirlist = os.listdir(self.sp_folder)
        for item in dirlist:

            sp = np.load(os.path.join(self.sp_folder, item))
            if len(sp) > hp.max_sep_len:
                continue

            self.dataset_files.append(item)

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):

        item = self.dataset_files[idx]

        name = item.replace('_sp.npy', '')
        sp = np.load(os.path.join(self.sp_folder, item))
        ap = np.load(os.path.join(self.ap_folder, name + '_ap.npy'))

        phn_condi = np.load(os.path.join(self.condi_folder, name + '_phn_condi.npy'))
        frame_condi = np.load(os.path.join(self.condi_folder, name + '_frame_condi.npy'))

        energy_condi = np.load(os.path.join(self.condi_folder, name + '_energy_condi.npy'))
        energy_condi = np.expand_dims(energy_condi, axis=1)

        # curent phn onehot
        phn = phn_condi[:, 1:]
        phn_count = phn_condi[:, :1]
        # f0 onehot
        f0 = frame_condi[:, 2:]
        pos_in_note = frame_condi[:, :1].squeeze()
        pos_in_note_sinusoid = util.get_s2s_singing_sinusoid_pos_in_x(pos_in_note)

        # pos_in_phn = frame_condi[:, 1:2].squeeze()
        # pos_in_phn_sinusoid = get_s2s_singing_sinusoid_pos_in_x(pos_in_phn)
        if not  len(sp) == len(ap) == len(f0) == np.sum(phn_count):
            print(name,'   ',len(sp) , ' ',len(ap) , ' ', len(f0) , ' ', np.sum(phn_count))
        assert len(phn) == len(phn_count)
        assert len(sp) == len(ap) == len(f0) == np.sum(phn_count)



        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(pos_in_note_sinusoid), aspect='auto', origin='bottom', interpolation='none')
        # plt.show()
        # plt.imshow(np.transpose(pos_in_phn_sinusoid), aspect='auto', origin='bottom', interpolation='none')
        # plt.show()
        # plt.imshow(np.transpose(sp), aspect='auto', origin='bottom', interpolation='none')
        # plt.show()

        timbre = np.concatenate((ap, sp), axis=1)

        return timbre, f0, phn, phn_count, pos_in_note_sinusoid, energy_condi


def collate_fn(batch):
    timbres = [b[0] for b in batch]
    f0s = [b[1] for b in batch]
    phns = [b[2] for b in batch]
    phn_count = [b[3] for b in batch]
    pos_in_note_sinusoid = [b[4] for b in batch]
    energy_condis = [b[5] for b in batch]

    phns_padded = pad_data(phns)
    phn_count_padded = pad_data(phn_count)

    f0s_padded = pad_data(f0s)

    pos_padded = pad_data(pos_in_note_sinusoid)
    timbres_padded, for_mask = pad_with_pos(timbres)

    energy_condis_padded = pad_data(energy_condis)

    return (phns_padded, phn_count_padded, f0s_padded, energy_condis_padded), (timbres_padded, pos_padded, for_mask)


def pad_with_pos(inputs):

    def pad_data(x, length):
        pad = 0
        x_padded = np.pad(
            x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values=pad)
        pos_padded = np.pad(np.array([(i+1) for i in range(np.shape(x)[0])]),
                            (0, length - x.shape[0]), mode='constant', constant_values=pad)

        return x_padded, pos_padded

    max_len = max((len(x) for x in inputs))

    condi_padded = np.stack([pad_data(x, max_len)[0] for x in inputs])
    pos_padded = np.stack([pad_data(x, max_len)[1] for x in inputs])

    return condi_padded, pos_padded


def pad_data(inputs):

    def pad(x, max_len):
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x = np.pad(x, (0, max_len - np.shape(x)
                       [0]), mode='constant', constant_values=0)

        return x[:, :s]

    max_len = max(np.shape(x)[0] for x in inputs)
    pad_output = np.stack([pad(x, max_len) for x in inputs])

    return pad_output



#test
if __name__ == '__main__':
    test = FastnpssDataset('data/dataset')
