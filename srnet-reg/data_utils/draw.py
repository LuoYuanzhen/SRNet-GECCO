import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.kernel_ridge import KernelRidge


def draw_range_line(x,
                    ys_mins,
                    ys_maxs,
                    ys_means,
                    xlabel,
                    ylabel,
                    legends=None,
                    savefile=None,
                    ):

    plt.figure()

    clf = KernelRidge(alpha=0.1, kernel='rbf')
    x = np.array(x)
    xs = np.arange(0, len(x), 0.5)
    for y_mins, y_maxs, y_means, l in zip(ys_mins, ys_maxs, ys_means, legends):
        clf.fit(x.reshape(-1, 1), y_means)
        plt.plot(xs, clf.predict(xs.reshape(-1, 1)), label=l)

        max_clf = KernelRidge(alpha=0.1, kernel='rbf')
        clf.fit(x.reshape(-1, 1), y_mins)
        max_clf.fit(x.reshape(-1, 1), y_maxs)
        plt.fill_between(xs, clf.predict(xs.reshape(-1, 1)), max_clf.predict(xs.reshape(-1, 1)), alpha=0.4)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(legends) > 1:
        plt.legend(loc=2, markerscale=20, bbox_to_anchor=(1.05, 1))
    if savefile:
        plt.savefig(savefile, dpi=600, bbox_inches='tight')
    plt.show()


def draw_error_bar_line(x,
                        ys_mins,
                        ys_maxs,
                        ys_means,
                        xlabel,
                        ylabel,
                        title='error bar',
                        legends=None,
                        savefile=None,
                        errorevery=1):
    fig, ax = plt.subplots()

    for y_mins, y_maxs, y_means, l in zip(ys_mins, ys_maxs, ys_means, legends):
        low_err, up_err = y_means-y_mins, y_maxs-y_means
        yerr = torch.vstack((low_err, up_err))
        # for i in np.arange(0, 5000, errorevery[1]):
        #     if y_means[i] - yerr[0, i] < 0.1:
        #         yerr[0, i] = y_means[i] - 0.78
        #     elif y_means[i] - yerr[0, i] < 0.15:
        #         yerr[0, i] = (y_means[i] - 0.8) * 0.9
        #     elif y_means[i] - yerr[0, i] < 0.6:
        #         yerr[0, i] = (y_means[i] - 0.8) * 0.8
        #     else:
        #         yerr[0, i] = (y_means[i] - 0.8) * 0.7
        #
        # ytick = ['0.950', '0.925', '0.900', '0.875', '0.850', '0.640', '0.100', '0.000']
        # plt.yticks([0.95, 0.925, 0.90, 0.875, 0.850, 0.825, 0.800, 0.775], ytick)
        ax.errorbar(x, y_means, yerr=yerr, label=l, ecolor='pink', capsize=2, capthick=1,
                    errorevery=errorevery)
        start_point = str(round(y_means[0].item(), 3))
        end_point = str(round(y_means[-1].item(), 3))
        ax.text(0, y_means[0], start_point)
        ax.text(len(y_means)-1, y_means[-1], end_point)
    if title is not None:
        fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legends is not None and len(legends) > 1:
        plt.legend(loc=2, markerscale=20, bbox_to_anchor=(1.05, 1))
    if savefile:
        plt.savefig(savefile, dpi=600, bbox_inches='tight')
        print(f'saved at {savefile}')
    plt.show()


def draw_fitness_box(filename, ys_list, xlabel=None):
    if not xlabel:
        xlabels = [f'F{i}' for i in range(len(ys_list))]
    else:
        xlabels = [f'{xlabel}{i}' for i in range(len(ys_list))]

    fig, ax = plt.subplots()
    ax.boxplot(ys_list, vert=True, patch_artist=True, labels=xlabels)
    ax.set_xlabel('Problem')
    ax.set_ylabel('Fitness')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


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


def project_to_2d_and_save(vars: tuple, zs: tuple, savefile=None,
                           vars_labels=None,
                           y_label=None,
                           zs_legends=None):
    if vars_labels is None:
        vars_labels = [f'x{i}' for i in range(len(vars))]
    if y_label is None:
        y_label = 'y'
    if zs_legends is None:
        zs_legends = [None for _ in range(len(zs))]
    fig = plt.figure()
    n_var = len(vars)
    if n_var == 2:
        # plot 3d
        n_row, n_col, begin, end = 1, 3, 2, 4
        ax3d = fig.add_subplot(n_row, 3, 1, projection='3d')
        for z, legend in zip(zs, zs_legends):
            ax3d.scatter(*vars, z, label=legend, s=0.1)
        ax3d.set_xlabel(vars_labels[0])
        ax3d.set_ylabel(vars_labels[1])
        ax3d.set_zlabel(y_label)
    else:
        n_row, n_col, begin, end = 1 if n_var % 2 > 0 else n_var // 2, n_var if n_var % 2 > 0 else 2, 1, n_var+1

    var_idx = 0
    for i in range(begin, end):
        ax2d = fig.add_subplot(n_row, n_col, i)

        for z, legend in zip(zs, zs_legends):
            ax2d.scatter(vars[var_idx], z, label=legend, s=0.1)
        if n_var % 2 > 0 and i > begin:
            ax2d.set_yticks(())
        elif n_var % 2 == 0 and i % 2 == 0:
            ax2d.set_yticks(())
        # ax2d.set_xlabel(vars_labels[var_idx])
        # ax2d.set_ylabel(y_label)
        var_idx += 1

    plt.legend(loc='best', markerscale=20)
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def draw_polate_data_curves(x_ranges, models, original_func, model_labels, savepath=None, title=None, var_names=None):

    # generate input datas
    x = []
    for x_range in x_ranges:
        x_min, x_max = x_range[0], x_range[1]
        interval = x_max - x_min

        xi = torch.linspace(x_min - interval * 2, x_max + interval * 2, 1000)
        # xi, _ = torch.sort(xi)
        x.append(xi)
    x = torch.stack(x, dim=1)
    n_var = x.shape[1]

    ys = [original_func(*[x[:, i] for i in range(n_var)])] + [model(x)[-1].detach() for model in models]
    labels = ['true'] + model_labels

    # I simply draw first dim of x as x axis
    fig, ax = plt.subplots()

    ymin, ymax = torch.min(ys[0]), torch.max(ys[0])
    for y, lb in zip(ys, labels):
        mini, maxi = torch.min(y), torch.max(y)
        ymin = mini if ymin > mini else ymin
        ymax = maxi if ymax < maxi else ymax
        ax.plot(x[:, 0], y.view(-1), label=lb)

    ax.vlines([x_min, x_max], ymin=ymin, ymax=ymax, linestyles='dashdot')

    if var_names is None:
        if n_var == 1:
            xlabel = 'x'
        else:
            xlabel = 'x0'
            for i in range(1, n_var):
                xlabel = xlabel + f'=x{i}'
    else:
        xlabel = var_names[0]
        for i in range(1, n_var):
            xlabel += f'={var_names[i]}'

    ax.set_xlabel(xlabel)
    ax.set_ylabel('y')
    ax.legend(loc='best')

    if title:
        fig.suptitle(title)

    if savepath is not None:
        plt.savefig(savepath)

    plt.show()

