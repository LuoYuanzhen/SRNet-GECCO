from functools import reduce

import torch
from torch import nn

from CGPNet.utils import CGPParameters, create_genes_and_bounds, create_nodes, get_active_paths


class CGPLayer(nn.Module):
    def __init__(self, n_input, n_output, params:dict, genes=None, bounds=None, ephs=None):
        super(CGPLayer, self).__init__()
        self.cp = CGPParameters(n_input, n_output, params)
        if genes is None:
            self.genes, self.bounds = create_genes_and_bounds(self.cp)
        else:
            assert bounds is not None
            self.genes, self.bounds = genes, bounds

        self.nodes = None
        self.active_paths = None
        self.active_nodes = None
        self.build()

        if ephs is None:
            self.ephs = nn.Parameter(torch.normal(mean=0., std=1., size=(self.cp.n_eph,)))
        else:
            self.ephs = ephs

    def build(self):
        self.nodes = create_nodes(self.cp, self.genes)
        self.active_paths = get_active_paths(self.nodes)
        self.active_nodes = set(reduce(lambda l1, l2: l1 + l2, self.active_paths))

    def forward(self, x):
        """normal CGP call way. Seeing x[:, i] as a single variable.
         INPUT: Make sure x.shape[1] == n_input
        OUTPUT: y where y.shape[1] == n_output """
        for path in self.active_paths:
            for gene in path:
                node = self.nodes[gene]
                if node.is_input:
                    node.value = self.ephs[node.no - self.cp.n_input] if node.no >= self.cp.n_input else x[:, node.no]
                elif node.is_output:
                    node.value = self.nodes[node.inputs[0]].value
                else:
                    f = node.func
                    operants = [self.nodes[node.inputs[i]].value for i in range(node.arity)]
                    node.value = f(*operants)

        outputs = []
        for node in self.nodes[-self.cp.n_output:]:
            if len(node.value.shape) == 0:
                outputs.append(node.value.repeat(x.shape[0]))
            else:
                outputs.append(node.value)

        return torch.stack(outputs, dim=1)

    def get_expression(self, input_vars=None, symbol_constant=False):
        if input_vars is not None and len(input_vars) != self.cp.n_input:
            raise ValueError(f'Expect len(input_vars)={self.n_inputs}, but got {len(input_vars)}')

        symbol_stack = []
        results = []
        for path in self.active_paths:
            for i_node in path:
                node = self.nodes[i_node]
                if node.is_input:
                    if i_node >= self.cp.n_input:
                        c = f'c{i_node - self.n_inputs}' if symbol_constant \
                            else self.ephs[i_node - self.cp.n_input].item()
                    else:
                        if input_vars is None:
                            c = f'x{i_node}' if self.cp.n_input > 1 else 'x'
                        else:
                            c = input_vars[i_node]
                    symbol_stack.append(c)
                elif node.is_output:
                    results.append(symbol_stack.pop())
                else:
                    f = node.func
                    # get a sympy symbolic expression.
                    symbol_stack.append(f(*reversed([symbol_stack.pop() for _ in range(f.arity)]), is_pt=False))

        return results
