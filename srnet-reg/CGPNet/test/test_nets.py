import sympy as sp
import torch
from torch import nn

from CGPNet.functions import default_functions
from CGPNet.methods import NewtonTrainer
from CGPNet.nets import MulExpCGPLayer, OneLinearCGPNet, OneExpCGPLayer, OneVectorCGPNet
from CGPNet.params import CGPParameters, NetParameters
from CGPNet.utils import pretty_net_exprs
from data_utils import io


def _print_net(net, x):
    print('###expressions:')
    print(net.get_expressions())

    results = net(x)
    print('###results:')
    [print(result, result.shape) for result in results]

    w_list, bias_list = net.get_ws(), net.get_bias()
    print('###weights:')
    [print(w, w.shape) for w in w_list]

    print('###biases:')
    [print(bias, bias.shape) for bias in bias_list]

    print('###parameters:')
    print(net.get_net_parameters())

    print('###final expression:')
    # print(get_net_expression(w_list, net.get_expressions()))


def test_net_expression():
    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 1],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneVectorCGPNet(net_params)
    print('get_expressions():')
    print(net.get_cgp_expressions())

    print('net ws:')
    print(net.get_ws())

    print('new get expressions:')
    exprs = pretty_net_exprs(net)
    print(exprs)


def test_net():
    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params)
    print(net.get_expressions())
    print(net.get_ws())
    results = net(x)
    [print(result, result.shape) for result in results]


def test_layer_torch():
    x = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float)

    var_names = ['x0', 'x1', 'x2']
    params = CGPParameters(1, 1,
                           5, 5, 101,
                           default_functions,
                           1)
    cgp_layer = OneExpCGPLayer(params)
    exprs = cgp_layer.get_expressions()
    print(exprs)
    print('CGP:', cgp_layer(x))


def test_net_torch():

    x = torch.tensor([[0, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params)
    _print_net(net, x)

    print('###encode it')
    encoded_net = OneLinearCGPNet.encode_net(net_params,
                                             genes_list=net.get_genes(),
                                             ephs_list=net.get_ephs(),
                                             w_list=net.get_ws(),
                                             bias_list=net.get_bias())
    _print_net(encoded_net, x)


def test_OneLinearCGPNet_OneExp():
    x = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=torch.float)  # 3, 3
    net_params = NetParameters([4, 4, 4, 2],
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = OneLinearCGPNet(net_params, clas_cgp=OneExpCGPLayer)
    _print_net(net, x)


def get_inverse_hessian(loss, weight):
    """Calculate the inverse Hessian matrix"""

    # save the gradient
    gradient = torch.autograd.grad(loss, weight, retain_graph=True, create_graph=True)[0]
    if not torch.all(torch.isfinite(gradient)):
        return None, None

    hessian = []
    for grad in gradient.view(-1):
        order2_gradient = torch.autograd.grad(grad, weight, retain_graph=True)[0]  # weight.shape
        hessian.append(order2_gradient.view(-1))
    hessian = torch.stack(hessian, dim=1)

    if not torch.all(torch.isfinite(hessian)):
        return None, None

    determinant = torch.det(hessian)
    if determinant == 0:
        return None, None

    eigenvalues, _ = torch.eig(hessian)
    if not torch.all(eigenvalues[:, 0] > 0):
        return None, None

    hessian_inv = torch.inverse(hessian)
    if not torch.all(torch.isfinite(hessian_inv)):
        return None, None

    return gradient, hessian_inv


def test_parameters():
    data_list = io.get_nn_datalist('../../dataset/kkk0_nn/')
    neurons = [data.shape[1] for data in data_list]

    net_params = NetParameters(neurons,
                               5, 5,
                               None,
                               default_functions,
                               1,
                               add_bias=True)
    net = OneVectorCGPNet(net_params)
    print(net.get_net_parameters())

    mse = nn.MSELoss()
    x = data_list[0]
    print('before MSE', [mse(h, net(x)[i]).item() for i, h in enumerate(data_list[1:])])

    print(pretty_net_exprs(net))

    nothing = 0
    for i, linear in enumerate(net.nn_layers):
        # print(list(linear.parameters()))
        # for name, parameter in linear.named_parameters():
        #     print(name, parameter)
        loss = mse(data_list[i+1], net(x)[i])
        for param in linear.parameters():
            gradient, hessian_inv = get_inverse_hessian(loss, param)
            if hessian_inv is not None:
                param.data = param.data - torch.matmul(gradient, hessian_inv)
                print(param)
                # linear.set_weight(linear.weight - torch.matmul(gradient, hessian_inv))
            if param.grad is not None:
                param.grad.data.zero_()
                nothing += 1
                # linear.weight.grad.data.zero_()
    print(nothing)
    print('after MSE', [mse(h, net(x)[i]).item() for i, h in enumerate(data_list[1:])])
    print(net.get_net_parameters())


def test():
    wi = torch.tensor([integer for integer in range(1, 5)])
    print(wi)
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    print(x)

    print(x*wi)
    print(wi*x)

    print(wi + 0.5)
    print(torch.sqrt(wi))


if __name__ == '__main__':
    # for _ in range(5):
    # test_net()
    # test_layer_torch()
    # test_net_torch()
    # test_net_expression()
    # test_OneLinearCGPNet_OneExp()
    # test_parameters()
    test()
