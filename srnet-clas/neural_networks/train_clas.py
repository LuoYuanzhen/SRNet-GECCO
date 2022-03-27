import os
import argparse
import sys


from data_utils.io import load_mlp_dataset, load_cnn_dataset

sys.path.append('../')
import json
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils import draw, io
from neural_networks.nn_models import MLP, LeNet


DATASET_PREFIX = '/home/luoyuanzhen/RemoteDisk/STORAGE/dataset/'
parser = argparse.ArgumentParser("training classifier")
parser.add_argument('--dataset_dir', type=str, default='mnist/')
parser.add_argument('--dataname', type=str, default='digit')
parser.add_argument('--model', type=str, default='cnn', choices=['mlp', 'cnn'])
parser.add_argument('--source', type=str, default='mnist')
args = parser.parse_args()

source = args.source
dataset_name, data_path = args.dataname, os.path.join(DATASET_PREFIX, args.dataset_dir, '')

if args.model == 'mlp':
    nn_model = partial(MLP, n_hiddens=[100, 100], regression=False)
    save_path = os.path.join(data_path, dataset_name + '_nn', '')
elif args.model == 'cnn':
    nn_model = LeNet()
    save_path = os.path.join(data_path, dataset_name + '_LeNet', '')
n_epoch, lr, optimizer_class = 1000, 1e-1, torch.optim.SGD
batch_size = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = torch.nn.CrossEntropyLoss()
io.mkdir(save_path)


def cal_loss(x, label, classifier):
    x, label = x.to(device), label.to(device)

    y_hat = classifier(x)
    if isinstance(y_hat, tuple) or isinstance(y_hat, list):
        y_hat = y_hat[-1]

    return y_hat, loss_func(y_hat, label)


def cal_accuracy(prediction, label):
    label = label.to(device)
    return (prediction.argmax(dim=1) == label).float().mean().item()


def train_clf(classifier, training_set, test_set):
    data_shape, n_output = training_set.data.shape[1:], len(np.unique(training_set.targets.numpy()))
    n_train, n_dataset = training_set.data.shape[0], training_set.data.shape[0] + test_set.data.shape[0]

    training_loader = DataLoader(training_set, batch_size=int(batch_size * n_train), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=int(batch_size * n_train), shuffle=True)

    optimizer = optimizer_class(classifier.parameters(), lr=lr)
    train_loss_his, test_loss_his = [], []
    print('Begin training classifier ({}->{}) with size of dataset: {}...'.format(data_shape, n_output, n_dataset))
    for epoch in range(n_epoch):
        train_loss_sum, train_acc_sum, batch_count = 0.0, 0.0, 0
        for x_batch, y_batch in training_loader:
            y_hat, loss = cal_loss(x_batch, y_batch, classifier)

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
                y_hat, loss = cal_loss(x_batch, y_batch, classifier)
                test_loss_sum += loss.cpu().item()
                test_acc_sum += cal_accuracy(y_hat, y_batch)
                test_batch_count += 1

        test_loss_aver = test_loss_sum / test_batch_count
        test_acc_aver = test_acc_sum / test_batch_count
        print("# Epoch %d: Train Loss: %.6f, Test Loss: %.6f, Train Acc : %.4f%%, Test Acc : %.4f%%" % (
            epoch, train_loss_aver, test_loss_aver, train_acc_aver * 100, test_acc_aver * 100))
        train_loss_his.append(train_loss_aver)
        test_loss_his.append(test_loss_aver)

    # save blackbox and the splited training set, test set
    model_path = save_path + 'nn_module.pt'
    torch.save(classifier.state_dict(), model_path)
    # draw img
    img_path = save_path + 'trend.pdf'
    draw.draw_mul_curves_and_save(list(range(n_epoch)), [train_loss_his, test_loss_his],
                                  savepath=img_path,
                                  title='Train and Test loss trend',
                                  xlabel='epoch',
                                  ylabel='loss',
                                  labels=['Train', 'Test'])
    return train_acc_aver, test_acc_aver


def train_cnn_classifier():
    training_set, test_set = load_cnn_dataset(
        data_path, dataset_name, source
    )
    n_train = training_set.data.shape[0]
    classifier_blackbox = nn_model.to(device)

    train_acc_aver, test_acc_aver = train_clf(classifier_blackbox, training_set, test_set)

    # Save hyperparameters
    hp_path = save_path + 'settings.json'
    x_instance, y_instance = training_set[0]
    x_instance = x_instance.unsqueeze(dim=1)
    n_conv = len(classifier_blackbox.n_channels) - 1
    blackbox_outputs = classifier_blackbox(x_instance.to(device))
    conv_input_shapes = [x_instance.shape[2:]] + [conved.shape[2:] for conved in blackbox_outputs[:n_conv-1]]
    conv_output_shapes = classifier_blackbox.conv_output_size_list
    mlp_structure = [classifier_blackbox.n_input] + classifier_blackbox.n_hiddens + [classifier_blackbox.n_output]
    log_dict = {
        'n_epoch': n_epoch,
        'batch_size': int(batch_size * n_train),
        'optimizer': optimizer_class.__name__,
        'lr': lr,
        'mlp_model': classifier_blackbox.__class__.__name__,
        'input_img_shapes': conv_input_shapes,
        'output_img_shapes': conv_output_shapes,
        'mlp_structure': mlp_structure,
        'acc_train': train_acc_aver,
        'acc_test': test_acc_aver
    }
    with open(hp_path, 'w') as f:
        json.dump(log_dict, f, indent=4)


def train_mlp_classifier():
    training_set, test_set = load_mlp_dataset(
        data_path, dataset_name, source
    )

    if source == 'mnist':
        n_var = training_set.data.shape[1] * training_set.data.shape[2]
    else:
        n_var = training_set.data.shape[1]
    n_output = len(np.unique(training_set.targets.numpy()))
    n_train = training_set.data.shape[0]
    classifier_blackbox = nn_model(n_var, n_output).float().to(device)
    optimizer = optimizer_class(classifier_blackbox.parameters(), lr=lr)

    training_loader = DataLoader(training_set, batch_size=int(batch_size*n_train), shuffle=True)
    test_loader = DataLoader(test_set, batch_size=int(batch_size*n_train), shuffle=True)

    train_loss_his, test_loss_his = [], []
    print('Begin training classifier ({}, {}) with size of dataset: {}...'.format(n_var, n_output, n_train))
    for epoch in range(n_epoch):
        train_loss_sum, train_acc_sum, batch_count = 0.0, 0.0, 0
        for x_batch, y_batch in training_loader:
            y_hat, loss = cal_loss(x_batch, y_batch, classifier_blackbox)

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
                y_hat, loss = cal_loss(x_batch, y_batch, classifier_blackbox)
                test_loss_sum += loss.cpu().item()
                test_acc_sum += cal_accuracy(y_hat, y_batch)
                test_batch_count += 1

        test_loss_aver = test_loss_sum / test_batch_count
        test_acc_aver = test_acc_sum / test_batch_count
        print("# Epoch %d: Train Loss: %.6f, Test Loss: %.6f, Train Acc : %.4f%%, Test Acc : %.4f%%" % (
            epoch, train_loss_aver, test_loss_aver, train_acc_aver*100, test_acc_aver*100))
        train_loss_his.append(train_loss_aver)
        test_loss_his.append(test_loss_aver)

    save_path = os.path.join(data_path, dataset_name+'_nn', '')
    io.mkdir(save_path)
    # save blackbox and the splited training set, test set
    model_path = save_path + 'nn_module.pt'
    torch.save(classifier_blackbox.state_dict(), model_path)
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
    x_instance, y_instance = training_set[0]
    blackbox_outputs = classifier_blackbox(x_instance.reshape(1, -1).to(device))
    mlp_structure = [n_var] + [h_output.shape[1] for h_output in blackbox_outputs]
    log_dict = {
        'n_epoch': n_epoch,
        'batch_size': int(batch_size*n_train),
        'optimizer': optimizer.__class__.__name__,
        'lr': lr,
        'mlp_model': classifier_blackbox.__class__.__name__,
        'structure': mlp_structure,
        'acc_train': train_acc_aver,
        'acc_test': test_acc_aver
    }
    with open(hp_path, 'w') as f:
        json.dump(log_dict, f, indent=4)

    # save the train and test data
    train_data = torch.hstack((training_set.data, training_set.targets.reshape(-1, 1)))
    test_data = torch.hstack((test_set.data, test_set.targets.reshape(-1, 1)))
    np.savetxt(os.path.join(save_path, 'train'), train_data.numpy())
    np.savetxt(os.path.join(save_path, 'test'), test_data.numpy())


if args.model == 'mlp':
    train_mlp_classifier()
elif args.model == 'cnn':
    train_cnn_classifier()





