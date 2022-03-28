import random

import func_timeout
import numpy as np
import sympy as sp


class Node:
    def __init__(self, no, func, arity, inputs=[], start_gidx=None):
        """

        :param no: index of node
        :param func: node's function, None when node is input or output
        :param arity: func's arity
        :param inputs: node's input genes.
        :param start_gidx: start position in the genes.
        """
        self.no = no
        self.func = func
        self.arity = arity
        self.inputs = inputs
        self.value = None

        self.is_input = False
        self.is_output = False
        if func is None:
            if len(self.inputs) == 0:
                self.is_input = True
            else:
                self.is_output = True

        self.start_gidx = start_gidx

    def __repr__(self):
        return f'Node({self.no}, {self.func}, {self.inputs})'


class CGPFactory:
    def __init__(self, params):
        self.params = params
        self.n_inputs = params.n_inputs
        self.n_outputs = params.n_outputs
        self.n_rows = params.n_rows
        self.n_cols = params.n_cols
        self.max_arity = params.max_arity
        self.levels_back = params.levels_back
        self.function_set = params.function_set
        self.n_eph = params.n_eph
        self.n_f_node = self.n_rows * self.n_cols
        self.n_f = len(self.function_set)

    def create_genes_and_bounds(self):
        genes = []
        uppers, lowers = [], []
        for i in range(self.n_inputs + self.n_eph, self.n_inputs + self.n_eph + self.n_f_node):
            # first bit is node function
            f_gene = random.randint(0, self.n_f - 1)

            lowers.append(0)
            uppers.append(self.n_f - 1)
            genes.append(f_gene)

            # next bits are input of the node function.
            col = (i - self.n_inputs - self.n_eph) // self.n_rows
            up = self.n_inputs + self.n_eph + col * self.n_rows - 1
            low = max(0, up - self.levels_back)
            for i_arity in range(self.max_arity):
                lowers.append(low)
                uppers.append(up)
                in_gene = random.randint(low, up)
                genes.append(in_gene)

        # output genes
        up = self.n_inputs + self.n_eph + self.n_f_node - 1
        low = max(0, up - self.levels_back)
        for i in range(self.n_outputs):
            lowers.append(low)
            uppers.append(up)
            out_gene = random.randint(low, up)
            genes.append(out_gene)

        return genes, (lowers, uppers)

    def create_bounds(self):
        """when provide genes, create bounds by genes"""
        uppers, lowers = [], []
        f_pos = 0
        for i in range(self.n_inputs + self.n_eph, self.n_inputs + self.n_eph + self.n_f_node):
            lowers.append(0)
            uppers.append(self.n_f - 1)

            col = (i - self.n_inputs - self.n_eph) // self.n_rows
            up = self.n_inputs + self.n_eph + col * self.n_rows - 1
            low = max(0, up - self.levels_back)
            for i_arity in range(self.max_arity):
                lowers.append(low)
                uppers.append(up)
            f_pos += self.max_arity + 1

        up = self.n_inputs + self.n_eph + self.n_f_node - 1
        low = max(0, up - self.levels_back)
        for i in range(self.n_outputs):
            lowers.append(low)
            uppers.append(up)

        return lowers, uppers

    def create_nodes(self, genes):
        nodes = []
        for i in range(self.n_inputs + self.n_eph):
            nodes.append(Node(i, None, 0, []))

        f_pos = 0
        for i in range(self.n_f_node):
            f_gene = genes[f_pos]
            f = self.function_set[f_gene]
            input_genes = genes[f_pos + 1: f_pos + f.arity + 1]
            nodes.append(Node(i + self.n_inputs + self.n_eph, f, f.arity, input_genes, start_gidx=f_pos))
            f_pos += self.max_arity + 1

        idx_output_node = self.n_inputs + self.n_eph + self.n_f_node
        for gene in genes[-self.n_outputs:]:
            nodes.append(Node(idx_output_node, None, 0, [gene], start_gidx=f_pos))
            f_pos += 1
            idx_output_node += 1

        return nodes


@func_timeout.func_set_timeout(10)
def timeout_simplify(expr):
    sim_str = sp.simplify(expr)
    return sim_str


def pretty_net_exprs(net, var_names=None):
    """Forward funcs_list and w_list, get final expressions w.r.t var_names"""
    net_input = net.neurons[0]
    if not var_names:
        var_names = list([f'x{i}' for i in range(net_input)]) if net_input > 1 else ['x']

    if net.__class__.__name__ == 'OneVectorCGPNet':
        # h_i = f_i(h_{i-1}) * w
        exprs = var_names  # n_var, 1
        for linear, cgp in zip(net.nn_layers, net.cgp_layers):
            exprs = cgp.get_expressions(input_vars=exprs)
            bias = linear.get_bias()
            if isinstance(bias, int):
                exprs = sp.Matrix(exprs) * linear.get_weight()
            else:
                exprs = sp.Matrix(exprs) * linear.get_weight() + bias.reshape(1, -1)
    elif net.__class__.__name__ == 'LinearOutputCGPNet':
        # last layer is linear, not wf+b
        exprs = var_names  # n_var, 1
        for linear, cgp in zip(net.nn_layers, net.cgp_layers):
            exprs = cgp.get_expressions(input_vars=exprs)
            bias = linear.get_bias()
            if isinstance(bias, int):
                exprs = sp.Matrix(exprs) * linear.get_weight()
            else:
                exprs = sp.Matrix(exprs) * linear.get_weight() + bias.reshape(1, -1)
        bias = net.last_nn_layer.get_bias()
        if isinstance(bias, int):
            exprs = exprs * net.last_nn_layer.get_weight()
        else:
            exprs = exprs * net.last_nn_layer.get_weight() + bias.reshape(1, -1)
    else:
        # h_i = f_i(h_{i-1} * W)
        w_list = net.get_ws()
        funcs_list = net.get_cgp_expressions()

        expr = sp.Matrix(var_names).T  # 1, n_var

        for w, funcs in zip(w_list, funcs_list):
            expr = layer_expression(expr, w, funcs)

        exprs = []
        for i in range(expr.shape[1]):
            try:
                exprs.append(str(timeout_simplify(expr[0, i])))
            except func_timeout.exceptions.FunctionTimedOut:
                exprs.append(str(expr[0, i]))
        exprs = sp.Matrix(exprs).T

    return exprs.tolist()


def linear_layer_expression(n_input, w, b=None):
    inputs = [f'x{i}' for i in range(n_input)] if n_input > 1 else ['x']
    if b is not None:
        expr = sp.Matrix(inputs).T * sp.Matrix(w) + b.reshape(1, -1)  # 1, n_var
    else:
        expr = sp.Matrix(inputs).T * sp.Matrix(w)
    return expr.tolist()[0]


def layer_expression(inputs, w, funcs):
    expr = inputs * sp.Matrix(w)  # 1, n_output

    # apply funcs to expr
    for i in range(expr.shape[0]):
        for j in range(expr.shape[1]):
            funcj = funcs[j] if expr.shape[1] == len(funcs) else funcs[0]
            if isinstance(funcj, float):
                # a single float may occur
                expr[i, j] = funcj
                continue
            if isinstance(funcj, str):
                # a single symbol x may occor
                funcj = sp.Symbol(funcj)
            expr[i, j] = funcj.replace(sp.Symbol('x'), expr[i, j])

    return expr


def probabilistic_mutate_net(parent, probability):
    gidx_list, mutatant_genes_list = [], []
    for cgp in parent.cgp_layers:
        gidxs, mutant_genes = _probabilistic_mutate_genes(cgp, probability)
        gidx_list.append(gidxs)
        mutatant_genes_list.append(mutant_genes)
    return parent.generate_offspring(gidx_list, mutatant_genes_list)


def _probabilistic_mutate_genes(cgp, probability):
    mutant_idx, mutant_genes = [], []
    low, up = cgp.bounds[0], cgp.bounds[1]

    for gidx in range(len(cgp.genes)):
        chance = random.random()
        if chance < probability:
            candicates = [gene for gene in range(low[gidx], up[gidx]+1)
                          if gene != cgp.genes[gidx]]
            if len(candicates) == 0:
                continue
            mutant_idx.append(gidx)
            mutant_genes.append(random.choice(candicates))

    return mutant_idx, mutant_genes


def partition_n_jobs(n_jobs, n_pop):
    np_per_job = n_pop // n_jobs * np.ones(n_jobs, dtype=np.int32)
    np_per_job[:n_pop % n_jobs] += 1

    idx_starts = np.cumsum(np_per_job)

    return idx_starts.tolist()


def parallel_optimize(sub_pop, data_list, trainer):
    new_subps = []
    for indiv in sub_pop:
        trainer.train(indiv, data_list)
        new_subps.append(indiv)

    return new_subps


def report(indiv=None, gen=None):
    def _sub(flist):
        str_list = []
        for f in flist:
            str_list.append(str(f)[:6]+'..')
        return str_list

    if indiv:
        print('|', format(gen, ' ^10'), '|', format(str(indiv.fitness)[:6]+'..', ' ^20'),
              '|', format(str(_sub(indiv.fitness_list)), ' ^40'), '|',
              format(str(indiv.get_cgp_expressions())[:30]+'..', ' ^40'), '|')
    else:
        print(format('', '_^123'))
        print('|', format('Gen', ' ^10'), '|', format('BestFitness', ' ^20'),
              '|', format('BestFitnessList', ' ^40'), '|', format('BestExpression', ' ^40'), '|')






