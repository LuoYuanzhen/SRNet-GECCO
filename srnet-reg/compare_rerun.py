import json
from functools import partial

import numpy as np
import pandas as pd
import scipy
import torch
from joblib import Parallel, delayed
from lime import lime_tabular

from CGPNet.config import clas_optim_map, clas_net_map, clas_cgp_map
from CGPNet.functions import default_functions
from CGPNet.methods import Evolution
from data_utils import io
from dataset_config import TEST_MAP, VALID_MAP, CURVES_DATASET
from maple.MAPLE import MAPLE
from maple.Misc import unpack_coefs
from neural_networks.nn_models import NN_MAP


evo_params = {
    'clas_net': 'OneVectorCGPNet',  # do not change
    'clas_cgp': 'OneExpOneOutCGPLayer',  # do not change
    'optim': 'Newton',  # Newton-Rapson optimization method do not change
    'n_rows': 5,  # rows of function nodes in each CGP
    'n_cols': 5,  # cols of function nodes in each CGP
    'levels_back': None,
    'function_set': default_functions,
    'n_eph': 1,  # number of constant added in each CGP
    'add_bias': True,  # do not change

    'n_population': 200,  # population size in each generation
    'n_generation': 5000,  # number of evoled generation
    'prob': 0.4,  # point mutation prob
    'verbose': 10,  # 0 would not be reported
    'stop_fitness': 1e-5,
    'random_state': None,
    'n_epoch': 0,  # useless, but do not delete
    'end_to_end': False,  # do not change
    'validation': True,  # do not change
    'evolution_strategy': 'chromosome_select'  # do not change
}

datasets = ['kkk5',
            'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
trials = 30


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


def run_srnet(data_list, valid_data_list):
    trainer = clas_optim_map[evo_params['optim']](end_to_end=evo_params['end_to_end'])
    clas_net, clas_cgp = clas_net_map[evo_params['clas_net']], clas_cgp_map[evo_params['clas_cgp']]
    controller = Evolution(evo_params=evo_params,
                           clas_net=clas_net,
                           clas_cgp=clas_cgp)
    elites, _ = controller.start(data_list, trainer, valid_data_list)
    return elites[0]


def run_compare(dataset, nn_model, trial):

    print(f'{dataset}_{trial} start.')
    # Fixes an issue where threads of inherit the same rng state
    scipy.random.seed()

    # Outpt
    out = {}
    file = open(f"compare_result/{dataset}_{trial}.json", "w")

    # Load data
    train = io.get_dataset('dataset/' + dataset).numpy()
    n_var = train.shape[1]-1

    test_domains = TEST_MAP[dataset]
    valid_domains = VALID_MAP[dataset]
    if dataset in CURVES_DATASET:
        X_test = load_linspace_data(n_var, test_domains[0])
    else:
        X_test = load_rand_data(test_domains)

    X_train = train[:, :-1]
    X_valid = load_rand_data(valid_domains)

    n = X_test.shape[0]
    d = X_train.shape[1]

    # get data list for srnet
    valid_data_list = None
    if evo_params['validation']:
        valid_data_list = [torch.from_numpy(X_valid)] + list(nn_model(torch.from_numpy(X_valid)))
    train_data_list = [torch.from_numpy(X_train)] + list(nn_model(torch.from_numpy(X_train)))

    # train srnet
    srnet = run_srnet(train_data_list, valid_data_list)

    # Fit LIME and MAPLE explainers to the model
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, discretize_continuous=False, mode="regression")
    exp_maple = MAPLE(X_train, my_pred(nn_model, X_train), X_valid, my_pred(nn_model, X_valid))

    out["srnet_valid_rmse"] = np.sqrt(np.mean((my_pred(nn_model, X_valid) - my_pred(srnet, X_valid)) ** 2)).astype(float)
    out["srnet_test_rmse"] = np.sqrt(np.mean((my_pred(nn_model, X_test) - my_pred(srnet, X_test)) ** 2)).astype(float)

    lime_rmse = np.zeros(2)
    maple_rmse = np.zeros(2)

    # save predictions of all interpretable model
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
        lime_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_lime)
        maple_test_pred = np.dot(np.insert(x_test, 0, 1), coefs_maple)

        lime_rmse[1] += (lime_valid_pred - model_valid_pred) ** 2
        maple_rmse[1] += (maple_valid_pred - model_valid_pred) ** 2

        lime_rmse[0] += (lime_test_pred - model_test_pred) ** 2
        maple_rmse[0] += (maple_test_pred - model_test_pred) ** 2

    lime_rmse /= n
    maple_rmse /= n

    lime_rmse = np.sqrt(lime_rmse)
    maple_rmse = np.sqrt(maple_rmse)

    out["lime_valid_rmse"] = lime_rmse[1]
    out["maple_valid_rmse"] = maple_rmse[1]

    out["lime_test_rmse"] = lime_rmse[0]
    out["maple_test_rmse"] = maple_rmse[0]

    json.dump(out, file, indent=4)
    file.close()


io.mkdir('compare_result/')

for dataset in datasets:
    nn_model = io.load_nn_model('dataset/' + dataset + '_nn/nn_module.pt', load_type='dict', nn=NN_MAP[dataset]).cpu()
    Parallel(n_jobs=trials)(
        delayed(run_compare)(dataset, nn_model, trial)
        for trial in range(trials))

columns = ['srnet_valid_mean', 'srnet_valid_min', 'srnet_valid_max',
           'lime_valid_mean', 'lime_valid_min', 'lime_valid_max',
           'maple_valid_mean', 'maple_valid_min', 'maple_valid_max',
           'srnet_test_mean', 'srnet_test_min', 'srnet_test_max',
           'lime_test_mean', 'lime_test_min', 'lime_test_max',
           'maple_test_mean', 'maple_test_min', 'maple_test_max',
           ]

df = pd.DataFrame(0, index=datasets, columns=columns)

datasets = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
            'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
for dataset in datasets:
    for column in columns:
        if column.endswith('min'):
            df.loc[dataset, column] = 1e10

print('saving all result...')
for dataset in datasets:
    for trial in range(trials):
        with open(f'compare_result/{dataset}_{trial}.json', 'r') as f:
            data = json.load(f)
        for name in ['srnet_valid', 'lime_valid', 'maple_valid', 'srnet_test', 'lime_test', 'maple_test']:
            rmse = data[f'{name}_rmse']
            df.loc[dataset, f'{name}_mean'] += rmse / trials
            df.loc[dataset, f'{name}_min'] = min(rmse, df.loc[dataset, f'{name}_min'])
            df.loc[dataset, f'{name}_max'] = max(rmse, df.loc[dataset, f'{name}_max'])

df.to_csv('compare_result/result.csv')
