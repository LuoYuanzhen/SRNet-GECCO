import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt


class DatasetProperty:
    def __init__(self, data_name, n_samples, n_var, belong, n_target=1):
        self.data_name = data_name
        self.n_samples = n_samples
        self.n_var = n_var
        self.belong = belong
        self.n_target = n_target


datasets = [
    DatasetProperty('kkk0', 200, 1, 'Regression'),
    DatasetProperty('kkk1', 200, 2, 'Regression'),
    DatasetProperty('kkk2', 200, 1, 'Regression'),
    DatasetProperty('kkk3', 25, 2, 'Regression'),
    DatasetProperty('kkk4', 1000, 3, 'Regression'),
    DatasetProperty('kkk5', 20, 2, 'Regression'),

    DatasetProperty('Feynman.I.10.7', 10000, 3, 'Regression'),
    DatasetProperty('Feynman.I.12.2', 10000, 4, 'Regression'),
    DatasetProperty('Feynman.I.13.12', 10000, 5, 'Regression'),
    DatasetProperty('Feynman.I.14.4', 10000, 2, 'Regression'),
    DatasetProperty('Feynman.test_9', 10000, 5, 'Regression'),
    DatasetProperty('Feynman.test_12', 10000, 5, 'Regression'),

    DatasetProperty('adult', 48842, 14, 'Classification', n_target=2),
    DatasetProperty('analcatdata_aids', 50, 4, 'Classification', n_target=2),
    DatasetProperty('agaricus_lepiota', 8145, 22, 'Classification', n_target=2),
    DatasetProperty('breast', 699, 10, 'Classification', n_target=2),
    DatasetProperty('car', 1728, 6, 'Classification', n_target=4),

    DatasetProperty('mnist_digit', 70000, 28*28, 'MNIST', n_target=10)
]

colors_map = {
    'Regression': 'tab:blue',
    'Classification': 'tab:green',
    'SR': 'tab:blue',
    'Feynman': 'tab:orange',
    'PMLB': 'tab:purple',
    'MNIST': 'tab:brown',
    'other': 'tab:pink'
}

belonged = 'o'
for dataset in datasets:
    if dataset.belong == 'MNIST':
        continue
    if belonged != dataset.belong:
        plt.scatter(dataset.n_samples, dataset.n_var, c=colors_map[dataset.belong], label=dataset.belong)
        belonged = dataset.belong
    else:
        plt.scatter(dataset.n_samples, dataset.n_var, c=colors_map[dataset.belong])
plt.xlabel('No. of Samples')
plt.ylabel('No. of Features')
plt.legend()
plt.grid()
plt.savefig('/home/amd01/lyz/result/figures/dataset_plot.pdf', dpi=600)
plt.show()


