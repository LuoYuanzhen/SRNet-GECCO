import copy
import random
import sympy as sp
import torch

from torch import nn

from CGPNet.layers import CGPLayer


class CGPNet(nn.Module):
    def __init__(self, n_input, n_output, n_hiddens, params, chromes=None):
        super(CGPNet, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hiddens = n_hiddens
        self.n_layer = len(n_hiddens)+1
        if chromes is not None:
            assert self.n_layer == len(chromes)
            for i, chrome in enumerate(chromes):
                setattr(self, 'chrome{}'.format(i+1), chrome)
        else:
            n_cgp_input = n_input
            for num, n_hidden in enumerate(n_hiddens):
                setattr(self, 'chrome{}'.format(num+1), nn.Sequential(
                    CGPLayer(n_cgp_input, 1, params),
                    nn.Linear(1, n_hidden)
                ))
                n_cgp_input = n_hidden

            setattr(self, 'chrome{}'.format(self.n_layer), nn.Sequential(
                CGPLayer(n_cgp_input, 1, params),
                nn.Linear(1, n_output)
            ))

        self.fitness, self.fitness_list = None, None

    def forward(self, x):
        input_data = self.chrome1(x)
        outputs = [input_data]

        for i in range(1, self.n_layer):
            input_data = getattr(self, 'chrome{}'.format(i+1))(input_data)
            outputs.append(input_data)

        return outputs

    def predict(self, x):
        outputs = []
        with torch.no_grad():
            input_data = self.chrome1(x)
            outputs.append(input_data)

            for i in range(1, self.n_layer):
                input_data = getattr(self, 'chrome{}'.format(i + 1))(input_data)
                outputs.append(input_data)

        return outputs

    def get_cgp_expressions(self):
        exprs = []
        for i in range(self.n_layer):
            exprs.append(getattr(self, 'chrome{}'.format(i+1))[0].get_expression())
        return exprs

    def mutate(self, prob):
        """TODO: implementing another mutation method"""
        mutated_net = copy.deepcopy(self)
        for i in range(mutated_net.n_layer):
            cgp = getattr(mutated_net, 'chrome{}'.format(i+1))[0]
            low, up = cgp.bounds[0], cgp.bounds[1]

            for gidx in range(len(cgp.genes)):
                chance = random.random()
                if chance < prob:
                    candicates = [gene for gene in range(low[gidx], up[gidx]+1) if gene != cgp.genes[gidx]]
                    if len(candicates) == 0:
                        continue
                    cgp.genes[gidx] = random.choice(candicates)
            cgp.build()

        return mutated_net

    def __repr__(self, var_names=None):
        if not var_names:
            var_names = list([f'x{i}' for i in range(self.n_input)]) if self.n_input > 1 else ['x']
        exprs = var_names

        for i in range(self.n_layer):
            sequential_layer = getattr(self, 'chrome{}'.format(i + 1))
            cgp_layer, linear_layer = sequential_layer[0], sequential_layer[1]

            exprs = cgp_layer.get_expression(input_vars=exprs)
            weight = linear_layer.weight.detach().cpu()
            bias = linear_layer.bias.detach().cpu()

            exprs = sp.Matrix(exprs) * weight.T + bias.reshape(1, -1)

        return str(exprs)

