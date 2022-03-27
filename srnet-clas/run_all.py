import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed

from CGPNet.functions import default_functions
from CGPNet.algorithms import evolve_srnet
from data_utils import io
from exp_utils import stat_trials, standard_data, brute_force_class_sample, grid_data
from neural_networks.nn_models import NN_MAP


DATASET_PREFIX = 'dataset/'
RESULT_PREFIX = 'result/'

parser = argparse.ArgumentParser("SRNet for neural network explaining task")
parser.add_argument('--start_trial', type=int, default=1)
parser.add_argument('--run_trials', type=int, default=30)
parser.add_argument('--stat', action='store_true', default=True)
parser.add_argument('--n_top', type=int, default=20)
parser.add_argument('--io', action='store_true', default=False)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cpu')
args = parser.parse_args()


clas_datasets = ['adult', 'agaricus_lepiota', 'analcatdata_aids', 'breast', 'car']

nn_dirs = list(['{}_nn'.format(dataset) for dataset in clas_datasets])
nn_names = clas_datasets
save_dirs = clas_datasets

regressions = [False] * len(nn_dirs)
sources = ['classification'] * len(nn_dirs)

params = {
    'n_population': 200,
    'n_generation': 5000,
    'prob': 0.4,
    'verbose': 100,
    'stop_fitness': 1e-6,
    'n_row': 10,
    'n_col': 10,
    'levels_back': None,
    'n_eph': 1,
    'function_set': default_functions,
    'mnnes': False,
    'optim_interval': 100
}

for nn_dir, nn_name, save_dir, regression, source in zip(nn_dirs, nn_names, save_dirs, regressions, sources):
    result_dir = os.path.join(RESULT_PREFIX, save_dir, '')
    nn_dir = os.path.join(DATASET_PREFIX, nn_dir, '')
    io.mkdir(result_dir)

    train, test = io.get_dataset(nn_dir+'train'), io.get_dataset(nn_dir+'test')
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]

    X = torch.vstack((x_train, x_test))

    # get the blackbox
    if source == 'regression':
        # for old dataset
        blackbox = io.load_nn_model(nn_dir+'nn_module.pt', NN_MAP[nn_name]).cpu()
    else:
        blackbox = io.get_blackbox(nn_dir, NN_MAP[nn_name]).cpu()

    # preprocessing data
    def blackbox_predict(X):
        return blackbox(X)[-1].detach()


    X = torch.tensor(standard_data(X), dtype=torch.float)
    if not regression:
        X_train, Y_train, d_train = brute_force_class_sample(X, blackbox_predict)
        X_train = torch.vstack((X, X_train))
    else:
        X_train = X


    def cal_acc(prediction, label):
        return (prediction == label).float().mean()


    def get_srnet_dict(srnet):
        sr_train_pred, sr_test_pred = srnet.cpu()(x_train)[-1], srnet.cpu()(x_test)[-1]
        if not regression:
            sr_train_pred, sr_test_pred = sr_train_pred.argmax(dim=1), sr_test_pred.argmax(dim=1)
        loss_func = torch.nn.MSELoss() if regression else cal_acc
        srnet_dict = {
            'SRNet_bb_acc/loss': [loss_func(sr_train_pred, bb_train_pred).item(), loss_func(sr_test_pred, bb_test_pred).item()],
            'bb_real_acc/loss': [loss_func(bb_train_pred, y_train).item(), loss_func(bb_test_pred, y_test).item()],
            'SRNet_real_acc/loss': [loss_func(sr_train_pred, y_train).item(), loss_func(sr_test_pred, y_test).item()],
            'fitness': srnet.fitness,
            'fitness_list': srnet.fitness_list.tolist(),
            'cgp_exprs': str(srnet.get_cgp_expressions()),
            'expr': str(srnet)
        }
        return srnet_dict


    def save_trial_result(save_dir, result):
        conv_f, topn, run_time = result
        # store conv_f
        np.savetxt(save_dir + 'conv_f', conv_f)
        # store topn
        topn_dict = {
            'run_time': run_time
        }
        for i, srnet in enumerate(topn):
            with open(save_dir + 'SRNet_{}'.format(i), 'wb') as f:
                pickle.dump(srnet, f)
            # dict for acc
            topn_dict['SRNet_{}'.format(i)] = get_srnet_dict(srnet)

        with open(save_dir + 'analysis.json', 'w') as f:
            json.dump(topn_dict, f, indent=4)
        # store params
        with open(save_dir + 'params.json', 'w') as f:
            json.dump(params, f, indent=4)


    # run and save srnet
    # Note that we should evolve the srnet with preprocessed data X_loader
    print('Start dataset {} for saving at {}'.format(nn_dir, save_dir))
    results = Parallel(n_jobs=args.run_trials)(
        delayed(evolve_srnet)(
            X_train, blackbox, params,
            checkpoint_dir=os.path.join(result_dir, 'checkpoint_{}'.format(epoch)), device=args.device, prob_batch=0.3,
            regression=regression
        )
        for epoch in range(args.start_trial-1, args.start_trial+args.run_trials-1)
    )

    bb_train_pred, bb_test_pred = blackbox(x_train)[-1], blackbox(x_test)[-1]
    if not regression:
        bb_train_pred, bb_test_pred = bb_train_pred.argmax(dim=1), bb_test_pred.argmax(dim=1)
    for i, result in enumerate(results):
        trial_dir = os.path.join(result_dir, str(args.start_trial+i), '')
        io.mkdir(trial_dir)
        save_trial_result(trial_dir, result)

    if args.stat:
        stat_trials(result_dir, args.n_top)
