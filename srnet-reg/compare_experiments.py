import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import json
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from lime import lime_tabular

import exp_utils
from data_utils import io
from dataset_config import TEST_MAP, VALID_MAP, FUNC_MAP, INTER_MAP, CURVES_DATASET
from exp_utils import encode_individual_from_json
from maple.MAPLE import MAPLE
from maple.Misc import unpack_coefs
from neural_networks.nn_models import NN_MAP


datasets = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
            'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']

logs = ['kkk4', 'feynman1', 'feynman2', 'feynman4', 'feynman5']
log = True


def _protected_log(output):
    return np.log(np.abs(output))


def my_pred(model, x):

    return model(torch.from_numpy(x).float())[-1].detach().numpy().ravel()


def load_linspace_data(n_var, domain):
    x = torch.linspace(domain[0], domain[1], 1000).unsqueeze(1)
    return x.repeat(1, n_var).numpy()


def load_rand_data(domains):
    x = []
    for domain in domains:
        xi = (domain[1] - domain[0]) * torch.rand(1000, 1) + domain[0]
        x.append(xi)
    return torch.hstack(x).numpy()


def run_compare(dataset):

    # Fixes an issue where threads of inherit the same rng state
    scipy.random.seed()

    # Outpt
    out = {}
    file = open("gecco_compare/" + dataset + ".json", "w")

    # Load data
    train = io.get_dataset('dataset/' + dataset).numpy()
    n_var = train.shape[1]-1

    test_domains = TEST_MAP[dataset]
    valid_domains = VALID_MAP[dataset]
    if dataset in CURVES_DATASET:
        X_test = load_linspace_data(n_var, test_domains[0])
    else:
        X_test = load_rand_data(test_domains)

    true_test = FUNC_MAP[dataset](*[torch.from_numpy(X_test[:, i]) for i in range(n_var)])
    X_train = train[:, :-1]
    X_valid = load_rand_data(valid_domains)

    n = X_test.shape[0]
    d = X_train.shape[1]

    # get nn
    nn_model = io.load_nn_model('dataset/'+dataset+'_nn/nn_module.pt', load_type='dict', nn=NN_MAP[dataset]).cpu()

    # get srnet
    srnet = encode_individual_from_json(f'cgpnet_result/b_logs/{dataset}_30log.json', 'elite[0]')

    # Fit LIME and MAPLE explainers to the model
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")
    exp_maple = MAPLE(X_train, my_pred(nn_model, X_train), X_valid, my_pred(nn_model, X_valid))

    out["srnet_valid_rmse"] = np.sqrt(np.mean((my_pred(nn_model, X_valid) - my_pred(srnet, X_valid)) ** 2)).astype(
        float)

    lime_rmse = np.zeros(2)
    maple_rmse = np.zeros(2)
    srnet_rmse = np.zeros(2)

    # save predictions of all interpretable model
    model_test_output, srnet_test_output, lime_test_output, maple_test_output = [], [], [], []
    lime_valid_output, maple_valid_output = [], []
    for i in range(n):
        x_test = X_test[i, :]
        x_valid = X_valid[i, :]

        coefs_lime = unpack_coefs(exp_lime, x_test, partial(my_pred, nn_model), d, X_train)  # Allow full number of features

        e_maple = exp_maple.explain(x_test)
        coefs_maple = e_maple["coefs"]

        model_valid_pred = my_pred(nn_model, x_valid.reshape(1, -1))
        lime_valid_pred = np.dot(np.insert(x_valid, 0, 1), coefs_lime)
        maple_valid_pred = np.dot(np.insert(x_valid, 0, 1), coefs_maple)

        model_test_pred = my_pred(nn_model, x_test.reshape(1, -1))
        srnet_test_pred = my_pred(srnet, x_test.reshape(1, -1))
        lime_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_lime)
        maple_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_maple)

        lime_valid_output.append(lime_valid_pred.reshape(1, -1))
        maple_valid_output.append(maple_valid_pred.reshape(1, -1))

        model_test_output.append(model_test_pred.reshape(1, -1))
        srnet_test_output.append(srnet_test_pred.reshape(1, -1))
        lime_test_output.append(lime_test_pred.reshape(1, -1))
        maple_test_output.append(maple_test_pred.reshape(1, -1))

        lime_rmse[1] += (lime_valid_pred - model_valid_pred) ** 2
        maple_rmse[1] += (maple_valid_pred - model_valid_pred) ** 2

        lime_rmse[0] += (lime_test_pred - model_test_pred) ** 2
        maple_rmse[0] += (maple_test_pred - model_test_pred) ** 2
        srnet_rmse[0] += (srnet_test_pred - model_test_pred) ** 2

    lime_rmse /= n
    maple_rmse /= n
    srnet_rmse /= n

    lime_rmse = np.sqrt(lime_rmse)
    maple_rmse = np.sqrt(maple_rmse)
    srnet_rmse = np.sqrt(srnet_rmse)

    out["lime_valid_rmse"] = lime_rmse[1]
    out["maple_valid_rmse"] = maple_rmse[1]

    out["srnet_test_rmse"] = srnet_rmse[0]
    out["lime_test_rmse"] = lime_rmse[0]
    out["maple_test_rmse"] = maple_rmse[0]
    #
    model_test_output = np.vstack(model_test_output)
    srnet_test_output = np.vstack(srnet_test_output)
    lime_test_output = np.vstack(lime_test_output)
    maple_test_output = np.vstack(maple_test_output)

    json.dump(out, file, indent=4)
    file.close()

    ys = [true_test, model_test_output, srnet_test_output, lime_test_output, maple_test_output]
    ys = [exp_utils.standard_data(y) for y in ys]
    labels = ['True', 'MLP', 'SRNet', 'LIME', 'MAPLE']

    if dataset in CURVES_DATASET:
        exp_utils.draw_output_compare_curves(X_test[:, 0], ys, labels, n_var=n_var, savepath=f"gecco_compare/{dataset}.pdf", inter_range=INTER_MAP[dataset][0])
    else:
        exp_utils.draw_project_output_scatter(X_test, ys, labels, savepath=f"gecco_compare/{dataset}.pdf", inter_ranges=INTER_MAP[dataset])


io.mkdir('gecco_compare/')
for dataset in datasets:
    run_compare(dataset)

with open(f'gecco_compare/{datasets[0]}.json', 'r') as f:
    data = json.load(f)

columns = list(data.keys())
df = pd.DataFrame(0, index=datasets, columns=columns)
for dataset in datasets:
    with open(f'gecco_compare/{dataset}.json', 'r') as f:
        data = json.load(f)
    for name in columns:
        df.loc[dataset, name] = data[name]

df.to_csv('gecco_compare/result.csv')
