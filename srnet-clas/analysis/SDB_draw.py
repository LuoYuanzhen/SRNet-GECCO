import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import os.path

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from data_utils import io
from exp_utils import grid_data, brute_force_class_sample, standard_data

DATASET_PREFIX = '/home/luoyuanzhen/STORAGE/result/'
figure_dir = os.path.join(DATASET_PREFIX, 'figures')
io.mkdir(figure_dir)

n_class, markers, colors = 4, ['o', '^', 's', 'p'], ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_redundant=0,
    n_classes=n_class,
    n_clusters_per_class=1
)
X = standard_data(X)

blackbox = SVC()
blackbox.probability = True
blackbox.fit(X, y)


def scatter(feature, label, c=None, color=True, show_bar=True, marker=None):
    for i in range(n_class):
        m = markers[i] if marker is None else marker
        if c is None:
            plt.scatter(feature[label == i, 0], feature[label == i, 1], c=colors[i], marker=m, s=20) if color else plt.scatter(feature[label == i, 0], feature[label == i, 1], marker=m, s=20)
        else:
            plt.scatter(feature[label == i, 0], feature[label == i, 1], c=c[label == i], marker=m, s=20)
    if c is not None and show_bar:
        plt.colorbar()


def predict(x):
    return blackbox.predict_proba(x)


def np_predict(x):
    return torch.from_numpy(predict(x.numpy())).float()


def contour(x, colors=None):
    x0, x1, X_grid = grid_data(x)
    pred = predict(X_grid).argmax(axis=1).reshape(x0.shape)
    if colors is None:
        plt.contour(x0, x1, pred, linestyles='dashdot')
    plt.contour(x0, x1, pred, linestyles='dashdot', colors=colors)


def cal_distance(probs):
    n_classes = probs.shape[1]
    return np.mean((probs - 1 / n_classes) ** 2, axis=1)


prob_blackbox = predict(X)
predict_blackbox = prob_blackbox.argmax(axis=1)
d_blackbox = cal_distance(prob_blackbox)

plt.subplot(1, 2, 1)
# scatter(X, predict_blackbox, d_blackbox, show_bar=False)
scatter(X, predict_blackbox)
contour(X, colors='black')
plt.title('(a)')


X_samples, pr_samples, d_samples = brute_force_class_sample(torch.from_numpy(X).float(), np_predict, n_sample=500)
X_new, pr_new, d_new = np.vstack((X, X_samples.numpy())), np.vstack((prob_blackbox, pr_samples.numpy())), \
                       np.concatenate((d_blackbox, d_samples.numpy()), axis=0)
predict_new = pr_new.argmax(axis=1)

plt.subplot(1, 2, 2)
plt.yticks([])
scatter(X, predict_blackbox)
scatter(X_samples, pr_samples.argmax(dim=1), marker='x')
# scatter(X_new, predict_new, d_new)
# scatter(X_new, predict_new)
contour(X, colors='black')
plt.title('(b)')

plt.savefig(os.path.join(figure_dir, 'USDB.pdf'), dpi=600)
plt.show()
plt.close()