import os
import time
from functools import partial

import matplotlib
import numpy as np
import torch
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from data_utils import io
from exp_utils import standard_data
from maple.skmaple import MAPLE, BadClassifierMAPLE
from neural_networks.nn_models import NN_MAP

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_default_MAPLE(X_train, y_train):
    if len(np.unique(y_train)) == 1:
        maple_model = BadClassifierMAPLE(None, None, y_train[0])
    else:
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.5)
        lr = LogisticRegression()
        maple_model = MAPLE(rf, lr)
        maple_model.fit(X_train, y_train)
    return maple_model


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


RESULT_PREFIX = 'result/'
DATASET_PREFIX = 'dataset/'
result_dataset_names = ['adult', 'analcatdata_aids', 'agaricus_lepiota', 'breast', 'car']
dataset_names = ['{}_nn'.format(name) for name in result_dataset_names]
figure_dir = os.path.join(RESULT_PREFIX, 'figures')
io.mkdir(figure_dir)

names = ['adult', 'analcatdata_aids', 'agaricus_lepiota', 'breast', 'car']
alias = ['P{}'.format(i) for i in range(len(names))]

for name, dataset, result in zip(names, dataset_names, result_dataset_names):
    result_dir, dataset_dir = os.path.join(RESULT_PREFIX, result), os.path.join(DATASET_PREFIX, dataset)
    train, test = np.loadtxt(os.path.join(dataset_dir, 'train')), np.loadtxt(os.path.join(dataset_dir, 'test'))
    X_train, X_test = standard_data(train[:, :-1]), standard_data(test[:, :-1])
    if X_test.shape[0] > 200:
        random_indices = np.random.randint(0, X_test.shape[0], 200)
        X_test = X_test[random_indices, :]
        compare_test = np.hstack((X_test, test[random_indices, -2:-1]))
    else:
        compare_test = test
    np.savetxt(os.path.join(result_dir, 'compare_test'), compare_test)
    blackbox = io.get_blackbox(dataset_dir, NN_MAP[name])
    blackbox_pred_prob = blackbox_fn(blackbox, X_train), blackbox_fn(blackbox, X_test)

    # Models
    srnet = io.get_srnet(os.path.join(result_dir, 'topn', 'SRNet_0'))
    lime = get_default_LIME(X_train)
    maple = get_default_MAPLE(X_train, blackbox_pred_prob[0].argmax(axis=1))

    # Predictions
    print('start LIME predictions on {}:'.format(name))
    start_time = time.time()
    lime_predict_test = LIME_predict(lime, X_test, blackbox_pred_prob[1], partial(blackbox_fn, blackbox))
    print('LIME predicts times:{} on {}'.format(time.time() - start_time, name))

    print('start MAPLE predicting on {}'.format(name))
    maple_predict_test = maple.predict(X_test)
    print('MAPLE predicts times:{} on {}'.format(time.time() - start_time, name))

    # Save
    np.savetxt(os.path.join(result_dir, 'lime_predict_test'), lime_predict_test)
    np.savetxt(os.path.join(result_dir, 'maple_predict_test'), maple_predict_test)
