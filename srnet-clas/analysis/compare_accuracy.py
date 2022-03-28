import json
import os.path
import pickle

import numpy as np
import torch

from data_utils import io
from neural_networks.nn_models import NN_MAP

DATASET_PREFIX = 'dataset/'
RESULT_PREFIX = 'result/'
datasets = ['adult', 'analcatdata_aids', 'agaricus_lepiota', 'breast', 'car']
nn_dirs = ['{}_nn'.format(dataset) for dataset in datasets]

for dataset, nn_dir in zip(datasets, nn_dirs):
    blackbox_dir = os.path.join(DATASET_PREFIX, nn_dir)
    blackbox = io.get_blackbox(blackbox_dir, NN_MAP[dataset])
    train, test = io.get_dataset(os.path.join(blackbox_dir, 'train')), io.get_dataset(os.path.join(blackbox_dir, 'test'))
    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    nn_pred_train, nn_pred_test = blackbox(X_train)[-1].detach().argmax(dim=1), blackbox(X_test)[-1].detach().argmax(dim=1)

    path = os.path.join(RESULT_PREFIX, dataset)
    # get the best srnets for each trial
    srnets = []
    for file in os.listdir(path):
        if file.isdigit():
            with open(os.path.join(path, file, 'SRNet_0'), 'rb') as f:
                srnets.append(pickle.load(f))
    srnet_accs = [
        (
            (srnet(X_train)[-1].detach().argmax(dim=1) == nn_pred_train).float().mean(),
            (srnet(X_test)[-1].detach().argmax(dim=1) == nn_pred_test).float().mean()
        )
        for srnet in srnets
    ]

    acc_json = {}
    names = ['train', 'test']
    for accs in srnet_accs:
        for acc, name in zip(accs, names):
            acc_json['{}_min'.format(name)] = torch.min(acc).item()
            acc_json['{}_mean'.format(name)] = torch.mean(acc).item()
            acc_json['{}_max'.format(name)] = torch.max(acc).item()

    with open(os.path.join(path, 'srnet_acc.json'), 'w') as f:
        json.dump(acc_json, f, indent=4)
