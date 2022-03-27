import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import numpy as np
import torch


colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:pink']


def draw_convergence(all_convs, title=None, savefile=None, label=None):
    n_generations = all_convs.shape[1]
    x = list(range(n_generations))
    min_conv, max_conv, mean_conv = np.min(all_convs, axis=0), np.max(all_convs, axis=0), np.mean(all_convs, axis=0)
    plt.fill_between(x, min_conv, max_conv, alpha=0.2)
    if label is None:
        plt.plot(x, mean_conv, 'o-', markevery=500)
    else:
        plt.plot(x, mean_conv, 'o-', markevery=500, label=label)
    if title is not None:
        plt.title(title)

    if savefile is not None:
        plt.savefig(savefile, dpi=600)


def draw_mul_curves_and_save(x, ys,
                             savepath=None,
                             title=None,
                             labels=None,
                             xlabel='x',
                             ylabel='y'):

    if labels is None:
        labels = [None for _ in ys]
    if title:
        plt.title(title)
    for y, lb in zip(ys, labels):
        plt.plot(x, y, label=lb)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()


def draw_polate_projection(X, x_ranges, ys, labels=None, title=None, savefile=None):
    n_var = X.shape[1]
    assert n_var == len(x_ranges)
    xlabels = ['x{}'.format(i) for i in range(n_var)]
    fig = plt.figure()
    if n_var == 2:
        n_row, n_col, begin, end = 1, 3, 2, 4
        ax3d = fig.add_subplot(n_row, n_col, 1, projection='3d')
        for i, y in enumerate(ys):
            if labels is not None:
                ax3d.scatter(*[X[:, j] for j in range(n_var)], y, label=labels[i], s=0.1, c=colors[i])
            else:
                ax3d.scatter(*[X[:, j] for j in range(n_var)], y, s=0.1, c=colors[i])
        ax3d.set_xlabel(xlabels[0])
        ax3d.set_ylabel(xlabels[1])
    else:
        n_row, n_col, begin, end = 1 if n_var % 2 > 0 else n_var // 2, n_var if n_var % 2 > 0 else 2, 1, n_var+1
    var_idx = 0
    for i in range(begin, end):
        ax2d = fig.add_subplot(n_row, n_col, i)
        ymin, ymax = np.min(ys[0]), np.max(ys[0])
        for j, y in enumerate(ys):
            ymin, ymax = min(ymin, np.min(y)), max(ymax, np.max(y))
            if labels is not None:
                ax2d.scatter(X[:, var_idx], y, label=labels[j], s=0.1, c=colors[j])
            else:
                ax2d.scatter(X[:, var_idx], y, s=0.1, c=colors[j])
        var_idx += 1
        if n_var % 2 > 0 and i > begin:
            ax2d.set_yticks(())
        elif n_var % 2 == 0 and i % 2 == 0:
            ax2d.set_yticks(())
        ax2d.vlines(list(x_ranges[i-begin]), ymin=ymin, ymax=ymax, linestyles='dashdot')
    if title:
        fig.suptitle(title)
    if savefile:
        plt.savefig(savefile, dpi=300)


def draw_polate_curves(x, x_range, ys, labels=None, xlabel=None, title=None, savefile=None):
    ymin, ymax = np.min(ys[0]), np.max(ys[0])
    for i, y in enumerate(ys):
        ymin = min(ymin, np.min(y))
        ymax = max(ymax, np.max(y))
        if labels is not None:
            plt.plot(x, y, label=labels[i], c=colors[i])
        else:
            plt.plot(x, y, c=colors[i])
    plt.vlines([x_range[0], x_range[1]], ymin=ymin, ymax=ymax, linestyles='dashdot')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if title is not None:
        plt.title(title)
    if savefile is not None:
        plt.savefig(savefile, dpi=600)


def draw_clf_scatter(X, y):
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    n_cls = len(np.unique(y))
    labels = list(range(n_cls))

    for cls in range(n_cls):
        plt.scatter(X[y==labels[cls], 0], X[y==labels[cls], 1], label=labels[cls])

    plt.legend()
    plt.show()
    plt.close()
