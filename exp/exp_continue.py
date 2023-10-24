from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, LightTS, FEDformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from utils.augmentations import augmentation
import os
import time
import copy
import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'FEDformer': FEDformer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'LightTS': LightTS,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def regenerate_dataset(self, dataset, nonoverlap_index):
        x_data, y_data, x_time, y_time = dataset.x_data[:nonoverlap_index], dataset.y_data[:nonoverlap_index], dataset.x_time[:nonoverlap_index], dataset.y_time[:nonoverlap_index]
        dataset.reload_data(x_data, y_data, x_time, y_time)
        return dataset

    def div_dataset(self, dataset, div):
        dataset_list = []
        div_length = len(dataset)//div
        for i in range(div):
            dataset_temp = copy.deepcopy(dataset)
            start = i*div_length
            end = (i+1)*div_length if i+1 < div else len(dataset)
            dataset_temp.x_data = dataset_temp.x_data[start: end]
            dataset_temp.y_data = dataset_temp.y_data[start: end]
            dataset_temp.x_time = dataset_temp.x_time[start: end]
            dataset_temp.y_time = dataset_temp.y_time[start: end]
            dataset_list.append(dataset_temp)
        return dataset_list

    def continue_learning(self, setting):
        # We use different dataloader here: train data -> first 90%, valid -> last 10%
        all_train_set, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        # Divide the dataset to several parts, train on 1, test on 2, train on 1 2, test on 3.
        testset_div = self.args.testset_div
        dataset_list = self.div_dataset(all_train_set, testset_div)

        all_loss = []
        current_index = 0
        nonoverlap_index = 0 # Determine data avaliable for training -> not overlap with test part

        for i, test_data in enumerate(dataset_list[:-1]): # the last part has no test set to validate
            current_index += len(test_data)
            nonoverlap_index = current_index - self.args.pred_len

            # Create train set
            if nonoverlap_index > 0:
                train_set = self.regenerate_dataset(all_train_set, nonoverlap_index)
                print("regenerating dataset: %d"%len(train_set))
            else:
                continue
            
            # Train
            train_loader = DataLoader(
                train_set,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                drop_last=True)
            self.model = self._build_model().to(self.device)
            self.train(setting, train_loader)

            # Test
            test_data = dataset_list[i+1]
            test_loader = DataLoader(
                test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                drop_last=False)
            loss = self.vali(test_loader, self._select_criterion(), i, setting)
            all_loss.append(loss)
            print('Part %d/%d'%(i+1,testset_div),'\tTest loss: ', loss)
            
        print('Final Test loss: ', sum(all_loss)/len(all_loss))
        for i, loss  in enumerate(all_loss):
            print('Part %d/%d'%(i + testset_div - len(all_loss),testset_div), '\tTest loss: ', loss)
        

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion, save_iter=0, setting=None):
        total_loss = []
        self.model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if 'former' not in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                preds.append(pred)
                trues.append(true)

                loss = criterion(pred, true)
                total_loss.append(loss)

        total_loss = np.average(total_loss)

        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)
        # np.save(folder_path + 'pred_%d.npy'%save_iter, preds)
        # np.save(folder_path + 'true_%d.npy'%save_iter, trues)

        self.model.train()
        return total_loss

    def train(self, setting, train_loader):
        path = os.path.join(self.args.checkpoints, setting)
        vali_data, vali_loader = self._get_data(flag='val')
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        x_data, y_data, x_time, y_time = train_loader.dataset.x_data, train_loader.dataset.y_data, train_loader.dataset.x_time, train_loader.dataset.y_time

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            train_loader.dataset.reload_data(copy.deepcopy(x_data), copy.deepcopy(y_data), copy.deepcopy(x_time), copy.deepcopy(y_time))
            train_loader.dataset.data_augmentation()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if 'former' not in self.args.model:
                    outputs = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
