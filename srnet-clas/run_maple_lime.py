import argparse
import json
import os.path
from functools import partial

import numpy as np
import torch
from joblib import Parallel, delayed
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_utils import io
from exp_utils import standard_data, brute_force_class_sample
from maple.skmaple import MAPLE, BadClassifierMAPLE
from neural_networks.nn_models import NN_MAP

DATASET_PREFIX = 'dataset/'
RESULT_PREFIX = 'result/'

args = argparse.ArgumentParser('Runing and comparing the MAPLE and LIME model')
args.add_argument('--start_trial', type=int, default=1)
args.add_argument('--n_trials', type=int, default=2)
args = args.parse_args()

dataset_names = ['adult', 'agaricus_lepiota', 'analcatdata_aids', 'breast', 'car']
nn_dirs = ['{}_nn'.format(dataset) for dataset in dataset_names]
result_dirs = dataset_names


def get_default_MAPLE():
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.5)
    lr = LogisticRegression()
    return MAPLE(rf, lr)


def get_default_LIME(X_train):
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, mode="classification")
    return exp_lime


def LIME_predict(lime_model, X, y_prob, fn):
    lime_predict = []
    n_class = y_prob.shape[1]
    for i in range(X.shape[0]):
        x_instance = X[i]
        prob_instance = []
        for label in range(n_class):
            explainer = lime_model.explain_instance(
                x_instance, fn, labels=(label,)
            )
            prob_instance.append(explainer.local_pred[0])
        lime_predict.append(prob_instance)
    lime_predict = np.array(lime_predict).argmax(axis=1)
    return lime_predict


def blackbox_fn(blackbox, X):
    return blackbox(torch.from_numpy(X).float())[-1].detach().numpy()


def start_single_trial(x_train, prob_train, x_test, prob_test):
    mr_train, mr_test = prob_train.argmax(axis=1), prob_test.argmax(axis=1)
    # get MAPLE
    if len(np.unique(mr_train)) == 1:
        maple_model = BadClassifierMAPLE(None, None, mr_train[0])
    else:
        maple_model = get_default_MAPLE()
        maple_model.fit(x_train, mr_train)
    # get LIME
    lime_model = get_default_LIME(x_train)

    maple_train_predict = maple_model.predict(x_train)
    maple_test_predict = maple_model.predict(x_test)
    lime_train_predict = LIME_predict(
        lime_model, x_train, prob_train, partial(blackbox_fn, blackbox)
    )
    lime_test_predict = LIME_predict(
        lime_model, x_test, prob_test, partial(blackbox_fn, blackbox)
    )

    maple_train_accuracy = (maple_train_predict == mr_train).mean()
    maple_test_accuracy = (maple_test_predict == mr_test).mean()
    lime_train_accuracy = (lime_train_predict == mr_train).mean()
    lime_test_accuracy = (lime_test_predict == mr_test).mean()
    return (maple_model, lime_model), \
           (maple_train_accuracy, maple_test_accuracy, lime_train_accuracy, lime_test_accuracy)


for dataset, nn_dir, result_dir in zip(dataset_names, nn_dirs, result_dirs):
    save_dir = os.path.join(RESULT_PREFIX, result_dir)
    blackbox_dir = os.path.join(DATASET_PREFIX, nn_dir)
    io.mkdir(RESULT_PREFIX)
    io.mkdir(save_dir)

    train, test = io.get_dataset(os.path.join(blackbox_dir, 'train')), io.get_dataset(os.path.join(blackbox_dir, 'test'))
    x_train, y_train = train[:, :-1], train[:, -1]
    x_test, y_test = test[:, :-1], test[:, -1]
    x_train, x_test = torch.from_numpy(standard_data(x_train)), torch.from_numpy(standard_data(x_test))
    X = torch.tensor(np.vstack((x_train, x_test)))
    x_new, y_new, d_new = brute_force_class_sample(X, lambda x:blackbox(x)[-1].detach())
    x_train = torch.vstack((x_train, x_new))

    blackbox = io.get_blackbox(blackbox_dir, NN_MAP[dataset]).cpu()
    prob_train, prob_test = blackbox(x_train)[-1].detach().numpy(), blackbox(x_test)[-1].detach().numpy()
    x_train, x_test = x_train.numpy(), x_test.numpy()

    print('Start dataset {} for saving at {}'.format(dataset, save_dir))
    results = Parallel(n_jobs=args.n_trials)(
        delayed(start_single_trial)(
            x_train, prob_train, x_test, prob_test
        )
        for epoch in range(args.n_trials)
    )
    mtr_accs, mte_accs, ltr_accs, lte_accs = [], [], [], []
    txt_name = ['mtr', 'mte', 'ltr', 'lte']
    analysis_dict = {}
    for i, result in enumerate(results):
        models, accuracys = result
        trial_dir = os.path.join(save_dir, 'comparation_{}'.format(i+args.start_trial))
        io.mkdir(trial_dir)
        np.savetxt(os.path.join(trial_dir, 'accs'), list(accuracys))
        for a, accs in zip(accuracys, [mtr_accs, mte_accs, ltr_accs, lte_accs]):
            accs.append(a)
    for tn, accs in zip(txt_name, [mtr_accs, mte_accs, ltr_accs, lte_accs]):
        analysis_dict['{}_min'.format(tn)] = np.min(accs)
        analysis_dict['{}_mean'.format(tn)] = np.mean(accs)
        analysis_dict['{}_max'.format(tn)] = np.max(accs)
    with open(os.path.join(save_dir, 'comparation.json'), 'w') as f:
        json.dump(analysis_dict, f, indent=4)




