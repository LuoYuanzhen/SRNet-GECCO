import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data_utils import io

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


RESULT_PREFIX = '../cgpnet_result/b_logs/'
result_dataset_names = list(['kkk{}_30cfs'.format(i) for i in range(6)]) + list(['feynman{}_30cfs'.format(i) for i in range(6)])
figure_dir = os.path.join(RESULT_PREFIX, 'figures')
io.mkdir(figure_dir)
ALOT = 1e6

names = list(['K{}'.format(i) for i in range(6)]) + list(['F{}'.format(i) for i in range(6)])


def regression_convergence():
    i = 1
    fig = plt.figure()
    for result_dataset, name in zip(result_dataset_names, names):
        ax2d = fig.add_subplot(4, 3, i)
        all_conv_path = os.path.join(RESULT_PREFIX, result_dataset)
        print(all_conv_path)
        all_conv = np.loadtxt(all_conv_path)  # n_trials, n_generations
        if all_conv.shape[0] < 5000:
            fill = np.repeat(all_conv[-1, :].reshape(1, -1), 5000-all_conv.shape[0], axis=0)
            all_conv = np.vstack((all_conv, fill))
        max_fitness = np.max(all_conv)
        n_gen = 1
        while np.isinf(max_fitness):
            max_fitness = np.max(all_conv[n_gen:, :])
            n_gen += 1
        # draw_convergence(all_conv, title=name)
        n_generations = all_conv.shape[0]
        x = list(range(n_generations))
        min_conv, max_conv, mean_conv = np.min(all_conv, axis=1), np.max(all_conv, axis=1), np.mean(all_conv, axis=1)
        ax2d.fill_between(x, min_conv, max_conv, alpha=0.2)
        ax2d.plot(x, mean_conv, 'o-', markevery=500)
        ax2d.spines['right'].set_visible(False)
        ax2d.spines['top'].set_visible(False)
        ax2d.grid()
        ax2d.set_yticks([max_fitness, max_fitness / 2, 0])
        if i >= 10:
            ax2d.set_xticks([0, 2500, 5000])
        ax2d.set_title(name)
        i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'regression_convergence.pdf'), dpi=600)
    plt.show()
    plt.close()


def example_regression_convergence():
    i = 1
    fig = plt.figure(figsize=(6, 5))
    example_result_datasets = result_dataset_names[:4] + result_dataset_names[6:8]
    example_names = names[:4] + names[6:8]
    for result_dataset, name in zip(example_result_datasets, example_names):
        ax2d = fig.add_subplot(2, 3, i)
        all_conv_path = os.path.join(RESULT_PREFIX, result_dataset)
        print(all_conv_path)
        all_conv = np.loadtxt(all_conv_path)  # n_trials, n_generations
        if all_conv.shape[0] < 5000:
            fill = np.repeat(all_conv[-1, :].reshape(1, -1), 5000 - all_conv.shape[0], axis=0)
            all_conv = np.vstack((all_conv, fill))
        max_fitness = np.max(all_conv)
        n_gen = 1
        while np.isinf(max_fitness):
            max_fitness = np.max(all_conv[n_gen:, :])
            n_gen += 1
        n_generations = all_conv.shape[0]
        x = list(range(n_generations))
        min_conv, max_conv, mean_conv = np.min(all_conv, axis=1), np.max(all_conv, axis=1), np.mean(all_conv, axis=1)
        ax2d.fill_between(x, min_conv, max_conv, alpha=0.2)
        ax2d.plot(x, mean_conv, 'o-', markevery=500)
        ax2d.spines['right'].set_visible(False)
        ax2d.spines['top'].set_visible(False)
        ax2d.grid()
        ax2d.set_yticks([max_fitness, max_fitness / 2, 0])
        if i >= 10:
            ax2d.set_xticks([0, 2500, 5000])
        ax2d.set_title(name)
        i += 1

    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'example_regression_convergence.pdf'), dpi=600)
    plt.show()
    plt.close()


example_regression_convergence()