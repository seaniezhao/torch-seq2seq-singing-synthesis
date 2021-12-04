import sys
sys.path.append("..")
import matplotlib
matplotlib.use('Agg')

import torch
import torch.utils.data
import time
import torch.nn as nn
from model.FastNPSS import FastNPSS
from model.Modules import bias_f
from dataset import FastnpssDataset, collate_fn
from train_util import ScheduledOptim, FastnpssLoss
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import hparams as hp
import numpy as np
import matplotlib.pyplot as plt

from config import *
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class ModelTrainer:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define model
        self.model = FastNPSS().to(self.device)
        self.device_count = torch.cuda.device_count()

        num_param = sum(param.numel() for param in self.model.parameters())

        print('Number of FastNPSS Parameters:', num_param)

        # Get dataset
        dataset = FastnpssDataset(DATA_ROOT_PATH)

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self.model.parameters(), betas=(0.9, 0.98), eps=1e-9)

        self.scheduled_optim = ScheduledOptim(optimizer,
                                         hp.lr,
                                         hp.n_warm_up_step)

        self.fastnpssLoss = FastnpssLoss()

        max_bias_shape = (hp.batch_size, hp.max_sep_len, hp.max_sep_len)
        self.G_BIAS = torch.Tensor(np.fromfunction(bias_f, max_bias_shape)).to(self.device)

        # Get training loader
        print("Get Training Loader")
        self.training_loader = DataLoaderX(dataset,
                                     batch_size=hp.batch_size,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     num_workers=4)

        # testset = FastnpssDataset(DATA_ROOT_PATH, train=False)
        # self.test_loader = DataLoader(testset,
        #                               batch_size=hp.batch_size,
        #                               shuffle=True,
        #                               collate_fn=collate_fn,
        #                               drop_last=True,
        #                               num_workers=cpu_count())

        self.snapshot_path = './snapshots'
        self.snapshot_name = 'default'

    def train(self):
        self.model.train()
        if self.device_count > 1:
            self.model = nn.DataParallel(self.model)
            print('multiple device using: ', self.device_count)



        step = 0
        for current_epoch in range(hp.epoch):
            print("epoch", current_epoch)
            self.epoch = current_epoch

            tic = time.time()

            epoch_loss = 0
            epoch_step = 0
            draw = True
            for (src, target) in iter(self.training_loader):
                phn, phn_count, f0, energy = src
                target, dis_pos, for_mask = target

                phn = torch.Tensor(phn).to(self.device)
                phn_count = torch.Tensor(phn_count).to(self.device)
                f0 = torch.Tensor(f0).to(self.device)

                dis_pos = torch.Tensor(dis_pos).to(self.device)
                target = torch.Tensor(target).to(self.device)

                energy = torch.Tensor(energy).to(self.device)

                for_mask = torch.Tensor(for_mask).to(self.device)

                self.scheduled_optim.zero_grad()
                # Forward

                timbre, energy_pred = self.model(phn, phn_count, dis_pos, f0, for_mask, self.G_BIAS, energy)

                timbre_loss, energy_loss = self.fastnpssLoss.forward(timbre, target, energy_pred, energy)

                total_loss = timbre_loss + energy_loss
                total_loss.backward()

                loss = total_loss.item()
                epoch_loss += loss
                epoch_step += 1

                real_model = self.model
                if self.device_count > 1:
                    real_model = self.model.module
                torch.nn.utils.clip_grad_norm_(real_model.parameters(), 1.0)
                self.scheduled_optim.step_and_update_lr()

                step += 1

                print('total_loss: ', loss, 'timbre_loss: ', timbre_loss.item(), 'energy_loss: ', energy_loss.item())
                # time step duration:
                if step == 100:
                    toc = time.time()
                    # print("one training step does take approximately " + str((toc - tic) * 0.01) + " seconds)")

                if (current_epoch) % 500 == 0 and draw:
                    plt.imshow(np.transpose(target.detach().cpu().numpy()[0]), aspect='auto', origin='bottom',
                               interpolation='none')
                    plt.show()
                    draw = False

            eval_avg_loss = 0 #self.evaluate()
            toc = time.time()
            print("one epoch does take approximately " + str((toc - tic)) +
                  " seconds), average loss: " + str(epoch_loss/epoch_step) + " eval average loss: " + str(eval_avg_loss))

            if (current_epoch + 1) % hp.save_per_epoch == 0:
                self.save_model()
        self.save_model()

    # def evaluate(self):
    #
    #     self.model.eval()
    #     step = 0
    #     epoch_loss = 0
    #     epoch_step = 0
    #     for (src, target) in iter(self.test_loader):
    #         phn, phn_count, f0 = src
    #         target, dis_pos, for_mask = target
    #
    #         phn = torch.Tensor(phn).to(self.device)
    #         phn_count = torch.Tensor(phn_count).to(self.device)
    #         f0 = torch.Tensor(f0).to(self.device)
    #
    #         dis_pos = torch.Tensor(dis_pos).to(self.device)
    #         target = torch.Tensor(target).to(self.device)
    #
    #         for_mask = torch.Tensor(for_mask).to(self.device)
    #
    #         self.scheduled_optim.zero_grad()
    #         # Forward
    #
    #         timbre = self.model(phn, phn_count, dis_pos, f0, for_mask)
    #
    #         timbre_loss = self.fastnpssLoss.forward(timbre, target)
    #         loss = timbre_loss.item()
    #         epoch_loss += loss
    #         epoch_step += 1
    #         # print('loss: ', loss)
    #         step += 1
    #
    #         # print('loss: ', loss)
    #         # time step duration:
    #     avg_loss = epoch_loss / epoch_step
    #     # print("average loss: " + str(avg_loss))
    #     self.model.train()
    #
    #     return avg_loss

    def save_model(self):
        if self.snapshot_path is None:
            return
        time_string = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        if not os.path.exists(self.snapshot_path):
            os.mkdir(self.snapshot_path)

        filename = self.snapshot_path + '/' + self.snapshot_name + '_' + str(self.epoch) + '_' + time_string

        to_save = self.model
        if self.device_count > 1:
            to_save = self.model.module
        torch.save(to_save.state_dict(), filename)
        print('model saved')


if __name__ == '__main__':

    trainer = ModelTrainer()
    trainer.train()
