"""Util methods for experiments"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import json

import numpy as np
import pmlb
import torch
from matplotlib import pyplot as plt
from torch import nn

from CGPNet.config import clas_net_map, clas_cgp_map
from CGPNet.params import NetParameters
from CGPNet.utils import pretty_net_exprs, linear_layer_expression
from data_utils import draw, io
from dataset_config import VALID_MAP, TEST_MAP, FUNC_MAP


class ClassifierModel(nn.Module):
    def __init__(self, num_input, num_output):
        super(ClassifierModel, self).__init__()
        self.Linear1 = nn.Sequential(
            nn.Linear(num_input, 3),
            nn.ReLU()
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU()
        )
        self.Linear3 = nn.Linear(3, num_output)

    def forward(self, x):
        output1 = self.Linear1(x.view(x.shape[0], -1))
        output2 = self.Linear2(output1)
        output = self.Linear3(output2)
        return output1, output2, output


def load_test_valid_data(dataset):
    valid_ranges = VALID_MAP[dataset]
    test_ranges = TEST_MAP[dataset]

    X_valid = []
    for valid_range in valid_ranges:
        xi_valid = (valid_range[1] - valid_range[0]) * torch.rand(1000, 1) + valid_range[0]
        X_valid.append(xi_valid)
    X_valid = torch.hstack(X_valid)

    X_test = []
    for test_range in test_ranges:
        xi_test = (test_range[1] - test_range[0]) * torch.rand(1000, 1) + test_range[0]
        X_test.append(xi_test)
    X_test = torch.hstack(X_test)

    y_test = FUNC_MAP[dataset](*[X_test[:, i] for i in range(X_test.shape[1])]).unsqueeze(dim=1)
    y_valid = FUNC_MAP[dataset](*[X_valid[:, i] for i in range(X_valid.shape[1])]).unsqueeze(dim=1)

    return torch.hstack((X_test, y_test)), torch.hstack((X_valid, y_valid))


def range_ydatas(ydatas, x_max):
    # ydatas : (30, 5000)
    mean_ys, min_ys, max_ys = [], [], []
    for i in range(x_max):
        ax_ys = [ydata[i] if i < len(ydata) else ydata[-1] for ydata in ydatas]
        # len of ax_ys = 30
        mean_ys.append(np.mean(ax_ys))
        min_ys.append(np.min(ax_ys))
        max_ys.append(np.max(ax_ys))
    return torch.tensor(mean_ys), torch.tensor(min_ys), torch.tensor(max_ys)


def draw_f_trend(filename, n_gen, cfs_list, legends=None, title=None, xlabel='gen', ylabel='fitness'):
    n_bar = 10
    interval = (n_gen - n_bar) // n_bar
    mean_fs_list, min_fs_list, max_fs_list = [], [], []
    for cfs in cfs_list:
        mean_fs, min_fs, max_fs = range_ydatas(cfs, n_gen)
        mean_fs_list.append(mean_fs)
        min_fs_list.append(min_fs)
        max_fs_list.append(max_fs)
    draw.draw_error_bar_line(list(range(n_gen)), min_fs_list, max_fs_list, mean_fs_list,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             title=title,
                             legends=legends,
                             savefile=filename,
                             errorevery=(0, interval))


def draw_hidden_heat_compare_img(filename, nndatas, srnndatas, title=""):
    # nndatas: (n_layer, (n_samples, n_hiddens))
    def normalize(_datas):
        _data_maxs, _data_mins = [], []
        for _data in _datas:
            _data_maxs.append(torch.max(_data))
            _data_mins.append(torch.min(_data))
        _data_max, _data_min = max(_data_maxs), min(_data_mins)
        for _i, _data in enumerate(_datas):
            _datas[_i] = (_data - _data_min) / (_data_max - _data_min)
        return _datas

    n_row, n_col = 3, 3
    n_samples = n_row * n_col

    n_h = len(nndatas)
    space = nndatas[0].shape[0] // n_samples
    idxs = [i*space for i in range(n_samples)]

    nn_samples = [torch.vstack([nndatas[i][j] for j in idxs]) for i in range(n_h)]  # (n_layer, 9, n_hiddens)
    srnn_samples = [torch.vstack([srnndatas[i][j] for j in idxs]) for i in range(n_h)]  # (n_layer, 9, n_hiddens)
    nn_norms = normalize(nn_samples)
    srnn_norms = normalize(srnn_samples)

    for h_idx, hs in enumerate(zip(nn_norms, srnn_norms)):
        nnh, srnnh = hs  # (9, n_hiddens)
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, sharex=True, sharey=True)
        for i, ax in enumerate(axes.flat):
            grid = torch.vstack((nnh[i], srnnh[i]))
            # YlGn or summer would be great
            im = ax.imshow(grid, cmap='summer', vmax=1, vmin=0)
            for h in range(2):
                [ax.text(j, h, round(grid[h, j].item(), 1), ha='center', va='center', color='b') for j in range(grid.shape[1])]

        fig.subplots_adjust(hspace=0)
        fig.suptitle(f'({title}-h{h_idx})')

        cax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
        fig.colorbar(im, cax=cax, orientation='horizontal')

        plt.savefig(f'{filename}_{h_idx}.pdf', dpi=300)
        plt.show()


def draw_output_compare_curves(x, ys, labels, n_var, inter_range=None, savepath=None, title=None):
    fig, ax = plt.subplots()

    for y, lb in zip(ys, labels):
        ax.plot(x.reshape(-1), y.reshape(-1), label=lb)

    if inter_range is not None:
        ax.vlines([inter_range[0], inter_range[1]], ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashdot')

    ax.set_xlabel('X')
    ax.legend(loc='best')

    if title:
        fig.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', dpi=600, pad_inches=0.0)

    plt.show()


def draw_project_output_scatter(x, ys, labels, inter_ranges=None, savepath=None, title=None):
    fig = plt.figure()
    n_var = x.shape[1]
    if n_var == 2:
        # plot 3d
        n_row, n_col, begin, end = 1, 3, 2, 4
        ax3d = fig.add_subplot(n_row, 3, 1, projection='3d')
        for z, legend in zip(ys, labels):
            ax3d.scatter(*list(x[:, i] for i in range(n_var)), z, label=legend, s=0.1)
        ax3d.set_xlabel('x0')
        ax3d.set_ylabel('x1')
    else:
        n_row, n_col, begin, end = 1 if n_var % 2 > 0 else n_var // 2, n_var if n_var % 2 > 0 else 2, 1, n_var + 1

    var_idx = 0
    for i in range(begin, end):
        ax2d = fig.add_subplot(n_row, n_col, i)
        for z, legend in zip(ys, labels):
            ax2d.scatter(x[:, var_idx], z.reshape(-1), label=legend, s=0.1)

        if inter_ranges is not None:
            x_min, x_max = inter_ranges[var_idx]
            ax2d.vlines([x_min, x_max], ymin=plt.ylim()[0], ymax=plt.ylim()[1], linestyles='dashdot')

        if n_var % 2 > 0 and i > begin:
            ax2d.set_yticks(())
        elif n_var % 2 == 0 and i % 2 == 0:
            ax2d.set_yticks(())
        ax2d.set_xlabel('x{}'.format(var_idx))
        var_idx += 1

    plt.legend(loc='best', markerscale=20)

    if title:
        fig.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight',dpi=600, pad_inches=0.0)

    plt.show()


def draw_decision_bound(data_dir, data_name, individual, from_pmlb):
    def _draw():
        fig, ax = plt.subplots()

        x, y = dataset[:, :-1], dataset[:, -1]
        type0, type1 = x[y == 0], x[y == 1]

        ax.scatter(type0[:, 0], type0[:, 1], label='0')
        ax.scatter(type1[:, 0], type1[:, 1], label='1')

        x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
        y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
        h = 0.01

        xx, yy = torch.meshgrid(torch.arange(x_min, x_max, h), torch.arange(y_min, y_max, h))
        xy = torch.stack((xx.ravel(), yy.ravel()), dim=1)

        z_indiv = individual(xy)[-1]
        z_net = clas_net.to('cpu')(xy)[-1].detach()
        # z_func = _func(xx, yy)

        z_indiv = (torch.sigmoid(z_indiv) >= 0.5).float().reshape(xx.shape)
        z_net = (torch.sigmoid(z_net) >= 0.5).float().reshape(xx.shape)
        # z_func = (torch.sigmoid(z_func) >= 0.5).float().reshape(xx.shape)

        cs_indiv = ax.contour(xx, yy, z_indiv, cmap=plt.cm.jet, linestyles='dotted')
        cs_net = ax.contour(xx, yy, z_net, cmap=plt.cm.gray)
        # cs_func = ax.contour(xx, yy, z_func, cmap=plt.cm.jet, linestyles='dotted')

        cs_indiv.levels = ['srnn']
        cs_net.levels = ['nn']
        # cs_func.levels = ['func']

        ax.clabel(cs_indiv, cs_indiv.levels, inline=True, fontsize=10)
        ax.clabel(cs_net, cs_net.levels, inline=True, fontsize=10)
        # ax.clabel(cs_func, cs_func.levels, inline=True, fontsize=10)
        plt.legend(loc='best')
        plt.show()

    if from_pmlb:
        dataset = pmlb.fetch_data(data_name, local_cache_dir=data_dir)
        dataset = torch.from_numpy(dataset.values).float()
    else:
        dataset = io.get_dataset(f'{data_dir}{data_name}')

    clas_net = ClassifierModel(dataset.shape[1]-1, 1)
    clas_net = io.load_nn_model(f'{data_dir}{data_name}_nn/nn_module.pt', load_type='dict', nn=clas_net)

    _draw()


def generate_domains_data(num_sample, domains):
    x_valid = []
    for valid_domain in domains:
        valid_xi = (valid_domain[1] - valid_domain[0]) * torch.rand(num_sample, 1) + valid_domain[0]
        x_valid.append(valid_xi)
    x_valid = torch.hstack(x_valid)

    return x_valid


def save_cfs(save_name, cfs):
    cfs_save = torch.t(torch.tensor(cfs, dtype=torch.float))
    io.save_parameters(cfs_save, save_name)


def standard_data(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().numpy()
    features_mean = np.mean(X, axis=0)
    features_std = np.std(X, axis=0)

    features_std = np.where(features_std != 0, features_std, 1.)
    return (X - features_mean) / features_std


def individual_to_dict(indiv, var_names=None):
    end_exp = pretty_net_exprs(indiv, var_names)
    expressions = indiv.get_cgp_expressions()
    if indiv.__class__.__name__ == 'LinearOutputCGPNet':
        last_nn = indiv.last_nn_layer
        if last_nn.add_bias:
            expressions.append(linear_layer_expression(indiv.neurons[-2], last_nn.get_weight(), last_nn.get_bias()))
        else:
            expressions.append(linear_layer_expression(indiv.neurons[-2], last_nn.get_weight()))

    indiv_dict = {'final_expression': str(end_exp),
                  'fitness': (indiv.fitness, indiv.fitness_list),
                  'expressions': str(expressions)
                  }
    return indiv_dict


def encode_individual_from_json(json_file, elite_name):
    """old version when the weights and bias are all in the json file."""
    with open(json_file, 'r') as f:
        records = json.load(f)

    evo_params = records['evolution_parameters']
    indiv_dict = records[elite_name]
    neurons = records['neurons']

    if 'add_bias' in evo_params.keys():
        add_bias = evo_params['add_bias']
    else:
        add_bias = False
    net_params = NetParameters(
        neurons=neurons,
        n_rows=evo_params['n_rows'],
        n_cols=evo_params['n_cols'],
        levels_back=evo_params['levels_back'],
        function_set=evo_params['function_set'],
        n_eph=evo_params['n_eph'],
        add_bias=add_bias
    )

    clas_net = clas_net_map[evo_params['clas_net']]
    clas_cgp = clas_cgp_map[evo_params['clas_cgp']]
    return clas_net.encode_net(net_params,
                               genes_list=indiv_dict['genes'],
                               ephs_list=indiv_dict['constants'],
                               w_list=indiv_dict['weights'],
                               bias_list=indiv_dict['bias'] if net_params.add_bias else None,
                               clas_cgp=clas_cgp)



