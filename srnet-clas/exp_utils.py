import json
import os
import pickle

import numpy as np
import torch

from data_utils import io


def stat_trials(dataset_dir, n_top):
    topn_dir = os.path.join(dataset_dir, 'topn', '')
    io.mkdir(topn_dir)
    all_cfs = []
    srnets_and_dicts, topn = [], []
    for file in os.listdir(dataset_dir):
        if not file.isdigit():
            continue
        trial_dir = os.path.join(dataset_dir, file, '')
        all_cfs.append(np.loadtxt(trial_dir+'conv_f'))
        with open(trial_dir+'analysis.json', 'r') as f:
            analysis_dict = json.load(f)
        for name in analysis_dict.keys():
            if not name.startswith('SRNet'):
                continue
            with open(trial_dir+name, 'rb') as f:
                srnet = pickle.load(f)
            srnets_and_dicts.append(['SRNet_{}:{}'.format(file, name[-1]), analysis_dict[name], srnet])

    np.savetxt(dataset_dir+'all_conv', all_cfs)
    srnets_and_dicts.sort(key=lambda x: x[2].fitness)
    result_dict = {}
    for i, dicts in enumerate(srnets_and_dicts[:n_top]):
        result_dict[dicts[0]] = dicts[1]
        with open(topn_dir + 'SRNet_{}'.format(i), 'wb') as f:
            pickle.dump(dicts[2], f)
    with open(dataset_dir + 'analysis.json', 'w') as f:
        json.dump(result_dict, f, indent=4)


def standard_data(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    features_mean = np.mean(X, axis=0)
    features_std = np.std(X, axis=0)

    features_std = np.where(features_std != 0, features_std, 1.)
    return (X - features_mean) / features_std


def brute_force_class_sample(features, predict_func, n_sample=1000, balance=False):
    predict_prob = predict_func(features[0:1, :])
    n_feature, n_class = features.shape[1], predict_prob.shape[1]
    uniform_prob = torch.ones(n_class) * (1 / n_class)
    n_class_sample = n_sample // n_class

    feature_min, _ = torch.min(features, dim=0)
    feature_max, _ = torch.max(features, dim=0)
    # uniform randomly generate data points
    random_points = torch.rand(10000, n_feature) * (feature_max - feature_min) + feature_min
    prob = predict_func(random_points)
    # for each class ci, we sample the point with prob_ci that closed to 1/n
    X_samples = []
    for ci in range(n_class):
        if balance:
            d_ci = prob[:, ci] - 1/n_class
            indices = d_ci > 0, d_ci < 0
            d_ci_p, d_ci_n = d_ci[indices[0]], d_ci[indices[1]]
            d_ci_p, p_indices = torch.sort(d_ci_p)
            d_ci_n, n_indices = torch.sort(d_ci_n)
            X_samples.append(random_points[indices[0], :][p_indices[:n_class_sample//2], :])
            X_samples.append(random_points[indices[1], :][n_indices[:n_class_sample//2], :])
        else:
            d_ci = torch.abs(prob[:, ci] - 1/n_class)
            d_ci, indices = torch.sort(d_ci)
            X_samples.append(random_points[indices[:n_class_sample], :])

    X_samples = torch.vstack(X_samples)
    y_prob_samples = predict_func(X_samples)
    d_samples = ((y_prob_samples - uniform_prob) ** 2).mean(dim=1)
    return X_samples, y_prob_samples, d_samples


def brute_force_random_class_sample(features, predict_func, n_sample=500):
    predict_prob = predict_func(features)
    n_feature, n_class = features.shape[1], predict_prob.shape[1]
    uniform_prob = torch.ones(n_class) * (1 / n_class)
    n_class_sample = n_sample // n_class

    # we generate random N(0, 1) data points
    n_random = n_sample * 10
    n_ci_samples = [n_class_sample] * n_class
    X_samples = []

    random_points = torch.randn(n_random, n_feature)
    random_prb = predict_func(features)
    random_predict = random_prb.argmax(dim=1)
    for ci in range(n_class):
        ci_indices = random_predict == ci

        ci_points = random_points[ci_indices, :]
        ci_prb = random_prb[ci_indices, :]

        d_ci = ((ci_prb - uniform_prob)**2).mean(dim=1)
        d_ci, d_indices = torch.sort(d_ci)

        X_samples.append(ci_points[d_indices[:n_class_sample]])
        n_ci_samples[ci] -= len(d_indices[:n_class_sample])


def grid_data(X, n_sample=200):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    if X.shape[1] == 2:
        x0_range = [np.min(X[:, 0]), np.max(X[:, 0])]
        x1_range = [np.min(X[:, 1]), np.max(X[:, 1])]
        x0s = np.linspace(*x0_range, n_sample)
        x1s = np.linspace(*x1_range, n_sample)
        x0, x1 = np.meshgrid(x0s, x1s)
        return x0, x1, np.c_[x0.ravel(), x1.ravel()]
    else:
        xi_ranges = [[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])]
        x_meshed = np.meshgrid(*[np.linspace(*xi_range, n_sample) for xi_range in xi_ranges])
        return x_meshed, np.hstack([xmesh.ravel().reshape(-1, 1) for xmesh in x_meshed])




