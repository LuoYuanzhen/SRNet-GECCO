import matplotlib
import torch
from matplotlib import pyplot as plt

from data_utils import io
from exp_utils import encode_individual_from_json

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# nndatas: (n_layer, (n_samples, n_hiddens))


data_dir = '../dataset/'
datasets = list(['kkk{}'.format(i) for i in range(6)]) + list(['feynman{}'.format(i) for i in range(6)])
snaps = list(['K{}'.format(i) for i in range(6)]) + list(['F{}'.format(i) for i in range(6)])

for filename, snap in zip(datasets, snaps):
    nn_dir = f"{data_dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{data_dir}{filename}')
    x_inner = true_inner[:, :-1]

    json_file = f'../cgpnet_result/b_logs/{filename}_30log.json'
    img_dir = f'../cgpnet_result/figures/'
    individual = encode_individual_from_json(json_file, 'elite[0]')

    srnn_layer_inner = individual(x_inner)
    nn_layer_inner = io.get_nn_datalist(nn_dir)[1:]
    for i in range(len(srnn_layer_inner)):
        srnn_layer_inner[i] = srnn_layer_inner[i].detach()
    nndatas = nn_layer_inner[:-1]
    srnndatas = srnn_layer_inner[:-1]

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
        n_hiddens = nnh.shape[1]
        fig, axes = plt.subplots(nrows=n_row, ncols=n_col, figsize=(6, 4))
        for i, ax in enumerate(axes.flat):
            grid = torch.vstack((nnh[i], srnnh[i]))
            # YlGn or summer would be great
            im = ax.imshow(grid, cmap='Blues', vmax=1, vmin=0)
            for h in range(2):
                [ax.text(j, h, round(grid[h, j].item(), 1), ha='center', va='center', color='black') for j in range(grid.shape[1])]
            if i >= 6:
                ax.set_xticks([j for j in range(n_hiddens)])
            else:
                ax.set_xticks(())
            if i % 3 == 0:
                ax.set_yticks([0, 1])
                ax.set_yticklabels(['NN', 'SR'])
            else:
                ax.set_yticks(())
        fig.subplots_adjust(wspace=0, hspace=0)

        plt.tight_layout()
        plt.savefig(f'{img_dir}{filename}_{h_idx}.pdf', bbox_inches='tight', dpi=600, pad_inches=0.0)
        plt.show()