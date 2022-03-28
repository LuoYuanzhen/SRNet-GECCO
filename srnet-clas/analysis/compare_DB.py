import os
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import torch
from lime import lime_tabular
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

from data_utils import io
from exp_utils import standard_data, grid_data
from maple.skmaple import MAPLE
from neural_networks.nn_models import NN_MAP

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_default_MAPLE():
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=10, max_features=0.5)
    lr = LogisticRegression()
    return MAPLE(rf, lr)


def get_default_LIME(X_train):
    exp_lime = lime_tabular.LimeTabularExplainer(X_train, mode="classification")
    return exp_lime


def find_great_point(X, clf, n=2):
    x0_min, x0_max = np.min(X[:, 0]), np.max(X[:, 0])
    x1_min, x1_max = np.min(X[:, 1]), np.max(X[:, 1])
    x0_mean = x0_min + (x0_max - x0_min) / n
    x1 = x1_min
    label = clf.predict(np.array([[x0_mean, x1]]))
    while clf.predict(np.array([[x0_mean, x1+0.05]])) == label:
        x1 += 0.5

    return np.array([x0_mean, x1])


RESULT_PREFIX = 'result/'
DATASET_PREFIX = 'dataset/'
names = ['agaricus_lepiota']
result_dataset_names = names
dataset_names = ['{}_nn'.format(name) for name in result_dataset_names]
figure_dir = os.path.join(RESULT_PREFIX, 'figures')
io.mkdir(figure_dir)

alias = ['P{}'.format(i) for i in range(len(names))]
y_labels = ['original', 'MLP', 'LIME', 'MAPLE', 'SRNet']

for alia, name, dataset, result in zip(alias, names, dataset_names, result_dataset_names):
    result_dir, dataset_dir = os.path.join(RESULT_PREFIX, result), os.path.join(DATASET_PREFIX, dataset)
    train, test = np.loadtxt(os.path.join(dataset_dir, 'train')), np.loadtxt(os.path.join(dataset_dir, 'test'))
    Xy = np.vstack((train, test))
    X, y = Xy[:, :-1], Xy[:, -1]
    n_class = len(np.unique(y))
    X = standard_data(X)

    blackbox = io.get_blackbox(dataset_dir, NN_MAP[name])
    srnet = io.get_srnet(os.path.join(result_dir, 'topn', 'SRNet_0'))

    blackbox_pred = blackbox(torch.from_numpy(X).float())[-1].detach().numpy().argmax(axis=1)
    srnet_pred = srnet(torch.from_numpy(X).float())[-1].detach().numpy().argmax(axis=1)

    # reduce dimensions to 2d
    tsne = TSNE(n_components=2)
    X_tnse = tsne.fit_transform(X)
    # just scatter the original dataset
    colors = ['tab:blue', 'tab:orange']
    fig = plt.figure()
    ax2d = fig.add_subplot(1, 1, 1)
    for cls in range(n_class):
        ax2d.scatter(X_tnse[y==cls, 0], X_tnse[y==cls, 1], s=2)
    x0, x1, X_grid = grid_data(X_tnse)

    # We use K-neighbors classifer as background classifier of blackbox and SRNet for drawing DB
    blackbox_background_model = KNeighborsClassifier(n_neighbors=1).fit(X_tnse, blackbox_pred)
    srnet_background_odel = KNeighborsClassifier(n_neighbors=1).fit(X_tnse, srnet_pred)


    def lime_linear_model(lime_explainer, instances):
        # extract the linear line for LIME model
        predict_prob = []
        for label in range(n_class):
            feature_coefs = lime_explainer.local_exp[label]
            proba = 0.
            for feature_idx, coef in feature_coefs:
                proba += instances[:, feature_idx] * coef
            proba += lime_explainer.intercept[label]
            predict_prob.append(np.array(proba).reshape(-1, 1))
        return np.hstack(predict_prob).argmax(axis=1)
        # feature_coefs = lime_explainer.local_exp[1]
        # x0_coef = list([feature_coef[1] for feature_coef in feature_coefs if feature_coef[0] == 0])[0]
        # x1_coef = list([feature_coef[1] for feature_coef in feature_coefs if feature_coef[0] == 1])[0]
        # intercept = lime_explainer.intercept[0]
        # return (-x0_coef * instances_x0 - intercept) / x1_coef


    def maple_linear_model(maple_explainer, instances):
        # extract the linear line for MAPLE
        return maple_explainer.predict(instances)
        # coefs = maple_explainer.coef_[0]
        # return (-coefs[0] * instances_x0 - maple_explainer.intercept_) / coefs[1]


    # plot the DB line of MLP and SRNet
    blackbox_contour = blackbox_background_model.predict(X_grid)
    srnet_contour = srnet_background_odel.predict(X_grid)
    ax2d.contour(x0, x1, blackbox_contour.reshape(x0.shape), linestyles='dashdot', colors='black')
    ax2d.contour(x0, x1, srnet_contour.reshape(x0.shape), colors='green', alpha=0.2)
    # we draw the 2 local linear models for LIME and MAPLE
    for i in range(2):
        # 'example' can be any other random 2d point
        example = find_great_point(X_tnse, blackbox_background_model, i+2)
        ax2d.scatter(example[0], example[1], color=colors[int(blackbox_background_model.predict(example.reshape(1, -1)))])
        x0s = np.linspace(example[0]-15, example[0]+15, 10)
        x1s = np.linspace(example[1]-15, example[1]+15, 10)
        rect = plt.Rectangle((example[0]-15, example[1]-15), 30, 30, fill=True, color='gray', alpha=0.3)
        ax2d.add_patch(rect)
        x0, x1 = np.meshgrid(x0s, x1s)
        x_grid = np.c_[x0.ravel(), x1.ravel()]
        ax2d.annotate(
            'Explained Local Point', xy=tuple([example[0], example[1]]), xytext=(-0.6, +0.4), weight='bold',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2')
        )
        # LIME
        lime = get_default_LIME(x_grid)
        lime_explainer = lime.explain_instance(
            example, lambda x: blackbox_background_model.predict_proba(x), labels=(0, 1)
        )
        # lime_x1 = lime_linear_model(lime_explainer, x0s)
        ax2d.contour(
            x0, x1, lime_linear_model(lime_explainer, x_grid).reshape(x0.shape), colors='tab:red', linewidths=1
        )
        # plt.plot(x0s, lime_x1, color='tab:red')
        # MAPLE
        maple = get_default_MAPLE()
        maple.fit(x_grid, blackbox_background_model.predict(x_grid))
        maple.predict(example.reshape(1, 2))
        maple_explainer = maple.fitted_linear_models_[0]
        # maple_x1 = maple_linear_model(maple_explainer, x0s)
        ax2d.contour(x0, x1, maple_linear_model(maple_explainer, x_grid).reshape(x0.shape), colors='tab:purple', linewidths=1)
        # plt.contourf(x0, x1, maple_linear_model(maple_explainer, x_grid).reshape(x0.shape), alpha=0.3)
        # plt.plot(x0s, maple_x1, color='tab:purple')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:orange'),
        Line2D([0], [0], linestyle='dashdot', color='black'),
        Line2D([0], [0], linestyle='solid', color='green'),
        Line2D([0], [0], linestyle='solid', color='tab:red'),
        Line2D([0], [0], linestyle='solid', color='tab:purple')
    ]
    plt.legend(legend_elements, ['Label 0', 'Label 1', 'MLP', 'SRNet', 'LIME', 'MAPLE'], ncol=3, loc='upper center')
    plt.axis('off')
    # plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, '{}_local_compare.pdf'.format(name)), dpi=600)
    plt.show()
    plt.close()


