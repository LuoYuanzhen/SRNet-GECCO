from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

from CGPNet.utils import CGPFactory


class BaseCGP:
    def __init__(self, params, genes=None, ephs=None):
        factory = CGPFactory(params)
        if genes is None:
            genes, bounds = factory.create_genes_and_bounds()
        else:
            bounds = factory.create_bounds()
        nodes = factory.create_nodes(genes)

        self.bounds = bounds
        self.params = params
        self.n_inputs = params.n_inputs
        self.n_outputs = params.n_outputs
        self.n_rows = params.n_rows
        self.n_cols = params.n_cols
        self.max_arity = params.max_arity
        self.levels_back = params.levels_back
        self.n_eph = params.n_eph
        self.function_set = params.function_set
        self.n_f_node = self.n_rows * self.n_cols
        self.n_f = len(self.function_set)

        self.fitness = None
        self.genes = genes
        self.nodes = nodes

        self.active_paths = self._get_active_paths()
        self.active_nodes = set(reduce(lambda l1, l2: l1 + l2, self.active_paths))

        if ephs is None:
            self.ephs = torch.normal(mean=0., std=1., size=(self.n_eph,))
        else:
            self.ephs = ephs

    def __call__(self, x):
        """normal CGP call way. Seeing x[:, i] as a single variable.
         INPUT: Make sure x.shape[1] == self.n_inputs
        OUTPUT: y where y.shape[1] == self.n_outputs """
        for path in self.active_paths:
            for gene in path:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.ephs[node.no - self.n_inputs] if node.no >= self.n_inputs else x[:, node.no]
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        outputs = []
        for node in self.nodes[-self.n_outputs:]:
            if len(node.value.shape) == 0:
                outputs.append(node.value.repeat(x.shape[0]))
            else:
                outputs.append(node.value)

        return torch.stack(outputs, dim=1)

    def _get_active_paths(self):
        stack = []
        active_path, active_paths = [], []
        for node in reversed(self.nodes):
            if node.is_output:
                stack.append(node)
            else:
                break

        while len(stack) > 0:
            node = stack.pop()

            if len(active_path) > 0 and node.is_output:
                active_paths.append(list(reversed(active_path)))
                active_path = []

            active_path.append(node.no)

            for input in reversed(node.inputs):
                stack.append(self.nodes[input])

        if len(active_path) > 0:
            active_paths.append(list(reversed(active_path)))

        return active_paths

    def get_active_genes_idx(self):
        """ Return all active genes. """
        active_genes = []
        for node_idx in self.active_nodes:
            node = self.nodes[node_idx]
            if node.is_input:
                continue

            if node.is_output:
                active_genes.append(node.start_gidx)
                continue

            active_genes += range(node.start_gidx, node.start_gidx + node.arity + 1)
        return active_genes

    def get_genes(self):
        return self.genes

    def get_ephs(self):
        return self.ephs

    def get_expressions(self, input_vars=None, symbol_constant=False):
        """return a list of self.n_outputs formulas"""
        if input_vars is not None and len(input_vars) != self.n_inputs:
            raise ValueError(f'Expect len(input_vars)={self.n_inputs}, but got {len(input_vars)}')

        symbol_stack = []
        results = []
        for path in self.active_paths:
            for i_node in path:
                node = self.nodes[i_node]
                if node.is_input:
                    if i_node >= self.n_inputs:
                        c = f'c{i_node - self.n_inputs}' if symbol_constant \
                            else self.ephs[i_node - self.n_inputs].item()
                    else:
                        if input_vars is None:
                            c = f'x{i_node}' if self.n_inputs > 1 else 'x'
                        else:
                            c = input_vars[i_node]
                    symbol_stack.append(c)
                elif node.is_output:
                    results.append(symbol_stack.pop())
                else:
                    f = node.func
                    # get a sympy symbolic expression.
                    symbol_stack.append(f(*[symbol_stack.pop() for _ in range(f.arity)], is_pt=False))

        return results

    def generate_clas_offspring(self, gidxs, mutant_genes, cls):
        genes = self.genes[:]
        for gidx, mutant in zip(gidxs, mutant_genes):
            genes[gidx] = mutant
        return cls(self.params, genes, self.ephs)

    def generate_offspring(self, gidxs, mutant_genes):
        return self.generate_clas_offspring(gidxs, mutant_genes, BaseCGP)


class OneExpOneOutCGPLayer(BaseCGP):
    """v1 version"""

    def generate_offspring(self, gidxs, mutant_genes):
        return super(OneExpOneOutCGPLayer, self).generate_clas_offspring(gidxs, mutant_genes, OneExpOneOutCGPLayer)

    def clone(self):
        return OneExpOneOutCGPLayer(self.params, self.genes, self.ephs)


class MulExpCGPLayer(BaseCGP):
    """overwrite the __call__ method, apply self.n_outputs different functions to x's each dimension"""

    def __call__(self, x):
        """here applys different functions to each dimension of x e.g. f0(x[:, 0]), f1(x[:, 1])..."""
        for output_idx in range(len(self.active_paths)):
            for gene in self.active_paths[output_idx]:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.ephs[node.no - self.n_inputs] if node.no >= self.n_inputs else x[:, output_idx]
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        outputs = []
        for node in self.nodes[-self.n_outputs:]:
            if len(node.value.shape) == 0:
                outputs.append(node.value.repeat(x.shape[0]))
            else:
                outputs.append(node.value)

        return torch.stack(outputs, dim=1)

    def generate_offspring(self, gidxs, mutant_genes):
        return super(MulExpCGPLayer, self).generate_clas_offspring(gidxs, mutant_genes, MulExpCGPLayer)


class OneExpCGPLayer(BaseCGP):
    def __call__(self, x):
        """here applys same functions to each dimension of x e.g. f(x[:, 0]), f(x[:, 1])..."""
        for output_idx in range(len(self.active_paths)):
            for gene in self.active_paths[output_idx]:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.ephs[node.no - self.n_inputs] if node.no >= self.n_inputs else x
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        # Since OneExpCGP only has one output node
        output_node = self.nodes[-1]
        if len(output_node.value.shape) == 0:
            return output_node.value.repeat(x.shape[0], x.shape[1])

        return output_node.value

    def generate_offspring(self, gidxs, mutant_genes):
        return super(OneExpCGPLayer, self).generate_clas_offspring(gidxs, mutant_genes, OneExpCGPLayer)


class LinearLayer(nn.Module):
    def __init__(self, in_features=None, out_features=None, weight=None, bias=None, add_bias=False):
        super(LinearLayer, self).__init__()
        self.add_bias = add_bias
        if weight is not None:
            self.weight = Parameter(weight)
        else:
            self.weight = Parameter(torch.normal(mean=0., std=1., size=(in_features, out_features)))
        if bias is not None:
            self.bias = Parameter(bias)
            self.add_bias = True
        else:
            if add_bias:
                self.bias = Parameter(torch.normal(mean=0., std=1., size=(out_features,)))

    def forward(self, x):
        if self.add_bias:
            return torch.matmul(x, self.weight) + self.bias

        return torch.matmul(x, self.weight)

    def set_weight(self, weight):
        if weight.shape != self.weight.shape:
            raise ValueError(f"expected weight's shape {self.weight.shape}, but got {weight.shape}")
        if isinstance(weight, np.ndarray):
            weight = torch.from_numpy(weight).float()
        elif isinstance(weight, list):
            weight = torch.tensor(weight).float()

        self.weight = Parameter(weight)

    def set_bias(self, bias):
        if bias.shape != self.bias.shape:
            raise ValueError(f"expected bias's shape {self.bias.shape}, but got {bias.shape}")
        if isinstance(bias, np.ndarray):
            bias = torch.from_numpy(bias).float()
        elif isinstance(bias, list):
            bias = torch.tensor(bias).float()

        self.weight = Parameter(bias)

    def get_weight(self):
        return self.weight.detach()

    def get_bias(self):
        if self.add_bias:
            return self.bias.detach()
        return 0

    def clone(self):
        if self.add_bias:
            return LinearLayer(weight=self.weight.detach().clone(), bias=self.bias.detach().clone())
        return LinearLayer(weight=self.weight.detach().clone())

