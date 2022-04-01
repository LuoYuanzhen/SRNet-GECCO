import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torchvision.datasets
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from neural_networks.utils import Dataset

sources = ['synthesis', 'pmlb', 'mnist']


def load_mlp_dataset(dataset_dir, dataname, source):
    assert source in sources

    if source == 'pmlb':
        dataset_locate = os.path.join(dataset_dir, dataname, dataname + '.tsv.gz')
        dataset = pd.read_csv(dataset_locate, sep='\t', compression='gzip')
        X = dataset.drop('target', axis=1).values
        y = dataset['target'].values
    else:
        dataset_locate = os.path.join(dataset_dir, dataname)
        dataset = np.loadtxt(dataset_locate)
        X, y = dataset[:, :-1], dataset[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y)
    training_set = Dataset(x_train, y_train)
    test_set = Dataset(x_test, y_test)
    return training_set, test_set


def get_blackbox(nn_dir, model_clas):
    with open(os.path.join(nn_dir, 'settings.json'), 'r') as f:
        structure = json.load(f)['structure']
    blackbox_model = model_clas(structure[0], structure[-1], structure[1:-1])
    return load_nn_model(os.path.join(nn_dir, 'nn_module.pt'), blackbox_model)


def get_srnet(srnet_path):
    with open(srnet_path, 'rb') as f:
        srnet = pickle.load(f).cpu()
    return srnet


def get_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def get_dataset(dataset_path, dtype='float'):
    dataset = torch.from_numpy(np.loadtxt(dataset_path))
    if len(dataset.shape) == 1:
        dataset = dataset.reshape(dataset.shape[0], -1)
    if dtype == 'float':
        return dataset.float()
    else:
        return dataset


def load_nn_model(nn_path, model):
    model.load_state_dict(torch.load(nn_path, map_location='cpu'))
    model.eval()
    return model


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
