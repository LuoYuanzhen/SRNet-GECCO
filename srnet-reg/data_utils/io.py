import torch
import pickle
import numpy as np

import os


def _get_datanames(nn_dir):
    """if your datalist is named as like ['input', 'hidden1', 'hidden2', ..., 'output']
    then you can use this method for convenience."""
    n = 0
    for hfile in os.listdir(nn_dir):
        if hfile.startswith('hidden') or hfile == 'input' or hfile == 'output':
            n += 1

    if n < 2:
        raise ValueError("n should >= 2")
    names = []
    for i in range(n):
        if i == 0:
            names.append('input')
        elif i == n - 1:
            names.append('output')
        else:
            names.append(f'hidden{i}')
    return names


# return a tensor dataset
def get_dataset(filepath):
    dataset = np.loadtxt(filepath)
    dataset = torch.from_numpy(dataset).float()
    dataset = dataset.view(dataset.shape[0], -1)
    return dataset


def get_datalist(file_dir, nameList):
    dataList = []
    for i in range(len(nameList)):
        dataList.append(get_dataset(file_dir+nameList[i]))
    return dataList


def get_nn_datalist(nn_dir):
    data_names = _get_datanames(nn_dir)
    return get_datalist(nn_dir, data_names)


# report hyper parameters to file
def save_report(message, filepath, mode='a'):
    with open(filepath, mode) as f:
        f.write(message+'\n')


# save layer's parameters to file
def save_parameters(param, filepath):
    data = param
    if len(param.shape) >= 3:
        data = data.view(data.shape[0], -1)
    np.savetxt(filepath, data)


def save_layers(layers, save_dir):
    for i in range(len(layers)):
        if i == 0:
            name = 'input'
        elif i == len(layers)-1:
            name = 'output'
        else:
            name = 'hidden%d' % (i)
        np.savetxt(save_dir+name, layers[i])


def save_nn_model(nn, savepath, save_type='model'):
    if save_type == 'model':
        torch.save(nn, savepath)
    elif save_type == 'dict':
        torch.save(nn.state_dict(), savepath)
    else:
        raise ValueError(f'karg save_type should be one of model or dict, not {save_type}.')


def load_nn_model(nnpath, load_type='model', nn=None):
    if load_type == 'model':
        nn = torch.load(nnpath)
    elif load_type == 'dict':
        if not nn:
            raise ValueError(f'karg nn should not be None when you specify load_type as dict')
        nn.load_state_dict(torch.load(nnpath))
    else:
        raise ValueError(f'karg load_type should be one of model or dict, not {load_type}.')
    nn.eval()
    return nn


def save_objs(objs, obj_names, savedir):
    for obj, name in zip(objs, obj_names):
        with open(f'{savedir}{name}.p', 'wb') as f:
            pickle.dump(obj, f)


def load_objs(obj_names, dir):
    objs = []
    for name in obj_names:
        with open(f'{dir}{name}.p', 'rb') as f:
            objs.append(pickle.load(f))

    return objs


def mkdir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)