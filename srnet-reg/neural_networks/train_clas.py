import json
import os.path
from functools import partial

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data_utils import io, draw
from neural_networks.nn_models import MLP
from neural_networks.utils import Dataset

suffix = '.tsv.gz'
dataset_name, data_path = 'analcatdata_aids', '/home/luoyuanzhen/STORAGE/dataset/pmlb/'

nn_model = partial(MLP, n_hiddens=[1000, 1000, 1000, 1000], regression=False)
n_epoch, lr, optimizer_class = 50000, 0.03, torch.optim.SGD
batch_size = 1


def load_pmlb_dataset():
    dataset_locate = os.path.join(data_path, dataset_name, dataset_name+suffix)
    dataset = pd.read_csv(dataset_locate, sep='\t', compression='gzip')
    return dataset


def train_classifier():
    dataset = load_pmlb_dataset()
    x_train, x_test, y_train, y_test = train_test_split(dataset.drop('target', axis=1).values, dataset['target'].values)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = torch.nn.CrossEntropyLoss()
    n_var, n_output = x_train.shape[1], len(np.unique(y_train))
    classifier_blackbox = nn_model(n_var, n_output).double().to(device)
    optimizer = optimizer_class(classifier_blackbox.parameters(), lr=lr)

    training_set = Dataset(x_train, y_train)
    training_loader = DataLoader(training_set, batch_size=int(batch_size*dataset.shape[0]), shuffle=True)

    test_set = Dataset(x_test, y_test)
    test_loader = DataLoader(test_set, batch_size=int(batch_size*dataset.shape[0]), shuffle=True)

    def cal_loss(x, label):
        x, label = x.to(device), label.to(device)

        y_hat = classifier_blackbox(x)
        if isinstance(y_hat, tuple):
            y_hat = y_hat[-1]

        return y_hat, mse(y_hat, label)

    def cal_accuracy(prediction, label):
        label = label.to(device)
        return (prediction.argmax(dim=1) == label).float().mean().item()

    train_loss_his, test_loss_his = [], []
    print('Begin training classifier ({}, {}) with size of dataset: {}...'.format(n_var, n_output, dataset.shape))
    for epoch in range(n_epoch):
        train_loss_sum, train_acc_sum, batch_count = 0.0, 0.0, 0
        for x_batch, y_batch in training_loader:
            y_hat, loss = cal_loss(x_batch, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            train_acc_sum += cal_accuracy(y_hat, y_batch)
            batch_count += 1
        train_loss_aver = train_loss_sum / batch_count
        train_acc_aver = train_acc_sum / batch_count
        # test
        test_loss_sum, test_acc_sum, test_batch_count = 0.0, 0.0, 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                y_hat, loss = cal_loss(x_batch, y_batch)
                test_loss_sum += loss.cpu().item()
                test_acc_sum += cal_accuracy(y_hat, y_batch)
                test_batch_count += 1

        test_loss_aver = test_loss_sum / test_batch_count
        test_acc_aver = test_acc_sum / batch_count
        print("# Epoch %d: Train Loss: %.6f, Test Loss: %.6f, Train Acc : %.4f%%, Test Acc : %.4f%%" % (
            epoch, train_loss_aver, test_loss_aver, train_acc_aver*100, test_acc_aver*100))
        train_loss_his.append(train_loss_aver)
        test_loss_his.append(test_loss_aver)

    save_path = os.path.join(data_path, dataset_name+'_nn', '')
    io.mkdir(save_path)
    # save blackbox and the splited training set, test set
    model_path = save_path + 'nn_module.pt'
    training_set_path = save_path + 'train'
    test_set_path = save_path + 'test'
    io.save_nn_model(classifier_blackbox, savepath=model_path, save_type='dict')
    io.save_parameters(np.hstack((x_train, y_train.reshape(-1, 1))), training_set_path)
    io.save_parameters(np.hstack((x_test, y_test.reshape(-1, 1))), test_set_path)
    # draw img
    img_path = save_path + 'trend.pdf'
    draw.draw_mul_curves_and_save(list(range(n_epoch)), [train_loss_his, test_loss_his],
                                  savepath=img_path,
                                  title='Train and Test loss trend',
                                  xlabel='epoch',
                                  ylabel='loss',
                                  labels=['Train', 'Test'])
    # Save hyperparameters
    hp_path = save_path + 'settings.json'
    blackbox_outputs = classifier_blackbox(torch.from_numpy(x_train[0:1]).to(device))
    mlp_structure = [x_train.shape[1]] + [h_output.shape[1] for h_output in blackbox_outputs]
    log_dict = {
        'n_epoch': n_epoch,
        'batch_size': int(batch_size*dataset.shape[0]),
        'optimizer': optimizer.__class__.__name__,
        'lr': lr,
        'mlp_model': classifier_blackbox.__class__.__name__,
        'structure': mlp_structure,
        'acc_train': train_acc_aver,
        'acc_test': test_acc_aver
    }
    with open(hp_path, 'w') as f:
        json.dump(log_dict, f, indent=4)


train_classifier()





