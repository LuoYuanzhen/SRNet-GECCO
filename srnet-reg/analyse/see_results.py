import numpy as np
import pandas as pd
import torch
import sympy as sp

import exp_utils
from CGPNet.utils import pretty_net_exprs
from data_utils import io
from dataset_config import vars_map


class _Elites:
    def __init__(self, name, exp, fitness):
        self.name = name
        self.exp = exp
        self.fitness = fitness[0]
        self.fitness_list = fitness[1]
        self.fit = fitness[1][-1]


def calculate_fitness_range():
    fnames = [f'kkk{i}_30log.json' for i in range(6)] + [f'feynman{i}_30log.json' for i in range(6)]

    import json
    for fname in fnames:
        file = f'{log_dir}{fname}'
        with open(file, 'r') as f:
            records = json.load(f)

        elite_keys = [key for key in records.keys() if key.startswith('elite')]

        fitnesses = []
        final_fits = []
        for num, key in enumerate(elite_keys):
            name = f'elite[{num}]'
            elite_dict = records[name]
            fitnesses.append(elite_dict['fitness'][0])
            final_fits.append(elite_dict['fitness'][1][-1])

        print(f'{fname}: {np.mean(fitnesses)}({np.mean(final_fits)}), {np.min(fitnesses)}({np.min(final_fits)}), {np.max(fitnesses)}({np.max(final_fits)})')


def sort_by_fitness():
    import json

    json_file = f'{log_dir}{data_name}30log.json'
    with open(json_file, 'r') as f:
        records = json.load(f)

    elite_keys = [key for key in records.keys() if key.startswith('elite')]

    elites_ = []
    for num, key in enumerate(elite_keys):
        name = f'elite[{num}]'
        elite_dict = records[name]
        elites_.append(_Elites(name, elite_dict['final_expression'], elite_dict['fitness']))

    elites_.sort(key=lambda x: x.fit)

    for elite in elites_:
        print(elite.__dict__)


def see_fitness_trend():
    fname = f'{data_name}_30cfs'
    save_name = f'{save_dir}{data_name}_trend.pdf'

    cfs = io.get_dataset(f'{log_dir}{fname}')
    # 30, 5000
    exp_utils.draw_f_trend(save_name, cfs.shape[0], [cfs.T.tolist()], legends=['srnn'], title=None)


def see_hidden_semantics():
    datasets = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
                'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
    columns = ['h_0^s(x)', 'h_1^s(h_0^s)', 'h_2^s(h_1^s)', 'y^s', 'O^s(x)']
    import json

    df = pd.DataFrame('-', index=datasets, columns=columns)
    for dataset in datasets:
        with open(f'../cgpnet_result/b_logs/{dataset}_30log.json', 'r') as f:
            records = json.load(f)

        elite = records['elite[0]']
        exps = elite['expressions'].replace('[', '').replace(']', '').split(',')
        print(exps)
        fitness = elite['fitness']

        individual = exp_utils.encode_individual_from_json(f'../cgpnet_result/b_logs/{dataset}_30log.json', 'elite[0]')
        if dataset in vars_map.keys():
            vars_name = vars_map[dataset][0]
        else:
            vars_name = None
        final = pretty_net_exprs(individual, vars_name)

        if len(exps) == 2:
            hidden_cols = columns[:2] + columns[3:4]
        else:
            hidden_cols = columns[:4]

        for exp, fit, col in zip(exps, fitness[1], hidden_cols):
            df.loc[dataset, col] = f"{sp.simplify(exp).evalf(4)}\n({format(fit, '.2e')})"

        df.loc[dataset, columns[-1]] = f"{final[0][0]}\n{format(fitness[0], '.2e')}"

    df.to_csv('../cgpnet_result/b_logs/semantics.csv')


if __name__ == '__main__':
    log_dir = '../cgpnet_result/b_logs/'
    data_name = 'kkk1'

    save_dir = '../cgpnet_result/b_imgs/'
    # sort_by_fitness()
    # see_fitness_trend()
    # calculate_fitness_range()
    see_hidden_semantics()
