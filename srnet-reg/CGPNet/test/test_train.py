import copy

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter

from CGPNet.config import clas_optim_map, clas_net_map
from CGPNet.functions import default_functions
from CGPNet.params import NetParameters
from data_utils import io


def evaluate(net, data_list):
    x = data_list[0]
    fitness_list = []
    lf = nn.MSELoss()

    outputs = net(x)
    for output, y in zip(outputs, data_list[1:]):
        fitness_list.append(lf(output, y).item())
    return fitness_list


def train(net, data_list, end_to_end, title, which):
    print(title)

    trainer = clas_optim_map[which](n_epoch=100, end_to_end=end_to_end)
    trainer.train(net, data_list)

    print(evaluate(net, data_list))


def test_train_end_to_end():
    data_dir = '/home/luoyuanzhen/Datasets/regression/kkk/kkk0_nn/'
    data_list = io.get_nn_datalist(data_dir)

    neurons = [data.shape[1] for data in data_list]
    net_params = NetParameters(neurons,
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = clas_net_map['OneLinearCGPNet'](net_params)
    print(net.get_expressions())

    same_nets = []
    for _ in range(3):
        same_nets.append(copy.deepcopy(net))

    end_to_end = False
    train(net, data_list, end_to_end, f'lbfgs end-to-end:{end_to_end}', 'lbfgs')
    train(same_nets[0], data_list, end_to_end, f'sgd end-to-end:{end_to_end}', 'sgd')

    end_to_end = True
    train(same_nets[1], data_list, end_to_end, f'lbfgs end-to-end:{end_to_end}', 'lbfgs')
    train(same_nets[2], data_list, end_to_end, f'sgd end-to-end:{end_to_end}', 'sgd')


def test_grad():
    data_dir = '/home/luoyuanzhen/Datasets/regression/kkk/kkk0_nn/'
    data_list = io.get_nn_datalist(data_dir)

    neurons = [data.shape[1] for data in data_list]
    net_params = NetParameters(neurons,
                               5, 5,
                               26,
                               default_functions,
                               1)
    net = clas_net_map['OneLinearCGPNet'](net_params)
    print(net.get_expressions())

    x = torch.tensor([2.])
    w = Parameter(torch.tensor([3.]))
    loss_1 = ((x * w) ** 2).mean()            # 2x^2*w
    loss_1.backward()
    print(loss_1.grad_fn, w.grad)
    w.grad.zero_()

    loss_2 = ((x * w) ** 2).mean() / 2       #
    # loss_2.backward()
    print(loss_2.grad_fn, w.grad)

    loss_2 = ((x * w) ** 2).mean() * 2 + 2   #
    print(loss_2.grad_fn, w.grad)

    loss_3 = ((x * w) ** 2).mean()   #
    loss_3.backward()
    print(loss_3.grad_fn, w.grad)

    with torch.no_grad():
        loss = ((x * w) ** 2).mean() * 3 + 2
        print(loss.grad_fn, w.grad)
    print('')


def test_torch_gradient():
    weight = Parameter(torch.tensor([[1., 2., 3.], [1., 2., 3.]]))
    x = torch.tensor([[1., 1.],
                      [2., 2.],
                      [3., 3.]])
    temp = torch.matmul(x, weight ** 2) + 0.5
    loss = temp.mean()

    loss.backward(create_graph=True)
    print(weight.grad)
    gradient = weight.grad.data.clone()

    weight.grad.data.zero_()

    hessian = []
    for grad in weight.grad.view(-1):
        order2_gradient = torch.autograd.grad(grad, weight, retain_graph=True)[0]
        hessian.append(order2_gradient.view(-1))
    hessian = torch.stack(hessian, dim=1)
    print(hessian, hessian.shape, weight.grad)
    # gradient,  = torch.autograd.grad(loss, weight, retain_graph=True, create_graph=True)
    # print(gradient)
    #
    # del_grad,  = torch.autograd.grad(gradient, weight, torch.ones_like(weight))
    # print(del_grad)
    # for grad in gradient.view(-1):
    #     del_grad = torch.autograd.grad(grad, weight)
    #     print(del_grad)


if __name__ == '__main__':
    # test_train_end_to_end()
    # test_grad()
    test_torch_gradient()
