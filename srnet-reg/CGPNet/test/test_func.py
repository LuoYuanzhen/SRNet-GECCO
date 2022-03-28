import sympy as sp
import torch

from CGPNet.functions import function_map
from CGPNet.utils import linear_layer_expression


def test_function():
    vars = [sp.Symbol('x0'), sp.Symbol('x1')]
    for str_func in function_map:
        func = function_map[str_func]
        if func.arity == 1:
            print(func(vars[0], is_pt=False))
        else:
            print(func(*vars, is_pt=False))

    vals = [torch.tensor([0.2], dtype=torch.float), torch.tensor([0.1], dtype=torch.float)]
    for str_func in function_map:
        func = function_map[str_func]
        if func.arity == 1:
            print(func(vals[0]))
        else:
            print(func(*vals))


def test_sp():
    inputs = ['x0', 'x1']
    w = torch.tensor([[1.], [2.]])
    b = torch.tensor([3.])

    print(linear_layer_expression(2, w, b))


if __name__ == '__main__':
    # test_function()
    test_sp()
