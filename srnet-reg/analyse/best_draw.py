import torch

from CGPNet.utils import pretty_net_exprs
from dataset_config import FUNC_MAP, INTER_MAP, TEST_MAP, vars_map, CURVES_DATASET
from data_utils import io
from exp_utils import encode_individual_from_json, draw_hidden_heat_compare_img, draw_decision_bound, \
    draw_output_compare_curves, draw_project_output_scatter
from neural_networks.nn_models import NN_MAP


def _protected_log(output):
    return torch.log(torch.abs(output))


def hidden_heat_map():

    nn_dir = f"{data_dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{data_dir}{filename}')
    x_inner = true_inner[:, :-1]

    individual = encode_individual_from_json(json_file, which)

    srnn_layer_inner = individual(x_inner)
    nn_layer_inner = io.get_nn_datalist(nn_dir)[1:]

    print(len(srnn_layer_inner), len(nn_layer_inner))
    for i in range(len(srnn_layer_inner)):
        srnn_layer_inner[i] = srnn_layer_inner[i].detach()

    draw_hidden_heat_compare_img(f'{img_dir}{filename}_{which}',
                                 nn_layer_inner[:-1],
                                 srnn_layer_inner[:-1],
                                 title=snap)


def output_curves():
    """draw the output curves, note that every variable x woule be the same as x[0]"""

    nn_dir = f"{data_dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{data_dir}{filename}')
    n_var = true_inner.shape[1] - 1

    x0_range = TEST_MAP[filename][0]
    inter_range = INTER_MAP[filename][0]
    x = torch.linspace(x0_range[0], x0_range[1], 1000).unsqueeze(1)
    x = x.repeat(1, n_var)
    x, _ = torch.sort(x, dim=0)

    func = FUNC_MAP[filename]
    true_output = func(*[x[:, i] for i in range(n_var)])

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()

    nn_output = nn(x)[-1].detach()
    individual = encode_individual_from_json(json_file, which)
    srnn_output = individual(x)[-1].detach()
    ys = [true_output, nn_output, srnn_output]
    labels = ['True', 'MLP', 'CGPNet']
    draw_output_compare_curves(x[:, 0], ys, labels,
                               inter_range=inter_range,
                               n_var=n_var,
                               title=f'({snap})',
                               savepath=f'{img_dir}{filename}_curves_{which}.pdf')


def output_curves_interpolate():

    nn_dir = f"{data_dir}{filename}_nn/"

    true_inner = io.get_dataset(f'{data_dir}{filename}')
    n_var = true_inner.shape[1] - 1

    x0_range = INTER_MAP[filename][0]
    x = torch.linspace(x0_range[0], x0_range[1], 1000).unsqueeze(1)
    x = x.repeat(1, n_var)
    x, _ = torch.sort(x, dim=0)

    func = FUNC_MAP[filename]
    true_output = func(*[x[:, i] for i in range(n_var)])

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    nn_output = nn(x)[-1].detach()

    individual = encode_individual_from_json(json_file, which)
    srnn_output = individual(x)[-1].detach()
    ys = [true_output, nn_output, srnn_output]
    labels = ['True', 'MLP', 'CGPNet']
    draw_output_compare_curves(x[:, 0], ys, labels,
                               n_var=n_var,
                               title=f'({snap})',
                               savepath=f'{img_dir}{filename}_curves_{which}_interpolation.pdf')


def project_output_scatter():

    nn_dir = f"{data_dir}{filename}_nn/"

    x_inner = io.get_dataset(f'{data_dir}{filename}_nn/input')

    x_range, inter_ranges = TEST_MAP[filename], INTER_MAP[filename]
    n_sample = x_inner.shape[0]
    x = []
    for i, ranges in enumerate(zip(x_range, inter_ranges)):
        inter, extra = ranges
        xi_left = (inter[0] - extra[0]) * torch.rand(n_sample//2, 1) + extra[0]
        xi_right = (extra[1] - inter[1]) * torch.rand(n_sample//2, 1) + inter[1]
        xi_inner = x_inner[:, i].unsqueeze(1)
        x.append(torch.vstack((xi_left, xi_inner, xi_right)))
    x = torch.hstack(x)

    n_var = x.shape[1]

    individual = encode_individual_from_json(json_file, which)
    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    func = FUNC_MAP[filename]

    srnn_output = individual(x)[-1].detach()
    true_output = func(*[x[:, i] for i in range(n_var)])
    nn_output = nn(x)[-1].detach()

    if is_log:
        srnn_output = _protected_log(srnn_output)
        true_output = _protected_log(true_output)
        nn_output = _protected_log(nn_output)

    draw_project_output_scatter(x, [true_output, nn_output, srnn_output], ['True', 'MLP', 'CGPNet'], inter_ranges,
                                title=f'({snap})',
                                savepath=f'{img_dir}{filename}_scatter_{which}.pdf')


def project_output_scatter_interpolate():

    nn_dir = f"{data_dir}{filename}_nn/"

    x = io.get_dataset(f'{data_dir}{filename}_nn/input')

    nn = io.load_nn_model(f'{nn_dir}nn_module.pt', load_type='dict', nn=NN_MAP[filename]).cpu()
    nn_output = nn(x)[-1].detach()

    true_output = FUNC_MAP[filename](*list([x[:, i] for i in range(x.shape[1])]))

    individual = encode_individual_from_json(json_file, which)
    srnn_output = individual(x)[-1].detach()

    ys = [nn_output, true_output, srnn_output]
    labels = ['True', 'MLP', 'CGPNet']

    if is_log:
        for i in range(3):
            ys[i] = _protected_log(ys[i])

    draw_project_output_scatter(x, ys, labels,
                                title=f'({snap})',
                                savepath=f'{img_dir}{filename}_scatter_{which}_interpolation.pdf')


def get_latex_expression():
    import sympy as sp

    datasets = ['kkk0', 'kkk1', 'kkk2', 'kkk3', 'kkk4', 'kkk5',
                'feynman0', 'feynman1', 'feynman2', 'feynman3', 'feynman4', 'feynman5']
    import json

    for dataset in datasets:
        with open(f'../cgpnet_result/b_logs/{dataset}_30log.json', 'r') as f:
            records = json.load(f)

        individual = encode_individual_from_json(f'../cgpnet_result/b_logs/{dataset}_30log.json', 'elite[0]')
        if dataset in vars_map.keys():
            vars_name = vars_map[dataset][0]
        else:
            vars_name = None
        final = pretty_net_exprs(individual, vars_name)

        exp = sp.latex(final[0][0])

        fitness = records['elite[0]']['fitness'][0]
        strs = f"${dataset}$ & ${exp}$ & {fitness}\\\\"

        print(strs)


datasets = list(['kkk{}'.format(i) for i in range(6)]) + list(['feynman{}'.format(i) for i in range(6)])
snaps = list(['K{}'.format(i) for i in range(6)]) + list(['F{}'.format(i) for i in range(6)])

for filename, snap in zip(datasets, snaps):
    # where is training data
    data_dir = '../dataset/'

    # where is the result json file
    json_file = f'../cgpnet_result/b_logs/{filename}_30log.json'
    # the figures saving directory
    img_dir = f'../cgpnet_result/figures/'
    # which elite in json_file you want to pick to draw
    which = 'elite[0]'

    is_log = False
    # draw extrapolation for MLP, SRNet
    if filename in CURVES_DATASET:
        output_curves()
        output_curves_interpolate()
    else:
        project_output_scatter()
        project_output_scatter_interpolate()

    hidden_heat_map()

    # output_curves()
    # output_curves_interpolate()
    # project_output_scatter()
    # project_output_scatter_interpolate()
    # hidden_heat_map()
    # get_latex_expression()












