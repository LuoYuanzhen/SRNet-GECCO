import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from data_utils import io
from data_utils.draw import draw_convergence

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


RESULT_PREFIX = '/home/luoyuanzhen/STORAGE/result/'
result_dataset_names = ['pmlb/adult', 'pmlb/analcatdata_aids', 'pmlb/agaricus_lepiota', 'pmlb/breast', 'pmlb/car']
# result_dataset_names = list(['regression/kkk{}'.format(i) for i in range(6)]) + list(['regression/feynman{}'.format(i) for i in range(6)])
figure_dir = os.path.join(RESULT_PREFIX, 'figures')
io.mkdir(figure_dir)

names = ['P0', 'P1', 'P2', 'P3', 'P4']
# names = list(['K{}'.format(i) for i in range(6)]) + list(['F{}'.format(i) for i in range(6)])


# def classification_convergence():
#     max_fitness = 0.
#     for result_dataset, name in zip(result_dataset_names, names):
#         all_conv_path = os.path.join(RESULT_PREFIX, result_dataset, 'all_conv')
#         all_conv = np.loadtxt(all_conv_path)  # n_trials, n_generations
#         if all_conv.shape[1] < 5000:
#             fill = np.repeat(all_conv[:, -1].reshape(-1, 1), 5000-all_conv.shape[1], axis=1)
#             all_conv = np.hstack((all_conv, fill))
#         max_fitness = max(max_fitness, np.max(all_conv))
#         draw_convergence(all_conv, label=name)
#
#     ax = plt.axes()
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     plt.grid()
#     plt.xticks([0, 2500, 5000])
#     plt.yticks([max_fitness, max_fitness / 2, 0])
#     plt.legend(ncol=5, loc='upper center')
#     plt.savefig(os.path.join(figure_dir, 'classification_convergence.pdf'), dpi=600)
#     plt.show()


def classification_convergence():
    i = 0
    fig = plt.figure(figsize=(4, 4))
    gs = GridSpec(3, 2, figure=fig)
    for result_dataset, name in zip(result_dataset_names, names):
        if i == 4:
            ax2d = fig.add_subplot(gs[2, :])
        else:
            ax2d = fig.add_subplot(gs[i//2, i%2])
        all_conv_path = os.path.join(RESULT_PREFIX, result_dataset, 'all_conv')
        all_conv = np.loadtxt(all_conv_path)  # n_trials, n_generations
        if all_conv.shape[1] < 5000:
            fill = np.repeat(all_conv[:, -1].reshape(-1, 1), 5000 - all_conv.shape[1], axis=1)
            all_conv = np.hstack((all_conv, fill))
        n_generations = all_conv.shape[1]
        x = list(range(n_generations))
        min_conv, max_conv, mean_conv = np.min(all_conv, axis=0), np.max(all_conv, axis=0), np.mean(all_conv, axis=0)
        ax2d.fill_between(x, min_conv, max_conv, alpha=0.2)
        ax2d.plot(x, mean_conv, 'o-', markevery=500)

        max_fitness = np.max(all_conv)
        ax2d.spines['right'].set_visible(False)
        ax2d.spines['top'].set_visible(False)
        ax2d.grid()
        ax2d.set_title(name)
        ax2d.set_xticks([0, 2500, 5000])
        ax2d.set_yticks([max_fitness, max_fitness / 2, 0])
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'classification_convergence.pdf'), dpi=600)
    plt.show()


def example_classification_convergence():
    i = 1
    fig = plt.figure(figsize=(4, 2))
    example_result_names = [result_dataset_names[0], result_dataset_names[2]]
    example_names = [names[0], names[2]]
    for result_dataset, name in zip(example_result_names, example_names):
        ax2d = fig.add_subplot(1, len(example_names), i)
        all_conv_path = os.path.join(RESULT_PREFIX, result_dataset, 'all_conv')
        all_conv = np.loadtxt(all_conv_path)  # n_trials, n_generations
        if all_conv.shape[1] < 5000:
            fill = np.repeat(all_conv[:, -1].reshape(-1, 1), 5000 - all_conv.shape[1], axis=1)
            all_conv = np.hstack((all_conv, fill))
        n_generations = all_conv.shape[1]
        x = list(range(n_generations))
        min_conv, max_conv, mean_conv = np.min(all_conv, axis=0), np.max(all_conv, axis=0), np.mean(all_conv, axis=0)
        ax2d.fill_between(x, min_conv, max_conv, alpha=0.2)
        ax2d.plot(x, mean_conv, 'o-', markevery=500)

        max_fitness = np.max(all_conv)
        ax2d.spines['right'].set_visible(False)
        ax2d.spines['top'].set_visible(False)
        ax2d.grid()
        ax2d.set_title(name)
        ax2d.set_xticks([0, 2500, 5000])
        ax2d.set_yticks([max_fitness, max_fitness / 2, 0])
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, 'example_classification_convergence.pdf'), dpi=600)
    plt.show()


example_classification_convergence()