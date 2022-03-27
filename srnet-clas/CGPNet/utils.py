import os
import pickle
import random

import numpy as np
import torch.utils.data

from CGPNet.functions import function_map
from data_utils import io


class SRNetDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if self.targets is not None:
            return self.data[index, :], [output[index, :] for output in self.targets]
        return self.data[index, :]


class CGPParameters:
    def __init__(self, n_input, n_output, params:dict):
        self.n_input = n_input
        self.n_output = n_output
        self.n_row = params['n_row']
        self.n_col = params['n_col']
        self.levels_back = params['levels_back']
        self.n_eph = params['n_eph']

        self.function_set = []
        self.max_arity = 1
        for str_fun in params['function_set']:
            if str_fun not in function_map:
                raise ValueError("%s function is not in 'function_map' in functions.py." % str_fun)
            self.max_arity = max(function_map[str_fun].arity, self.max_arity)
            self.function_set.append(function_map[str_fun])

        self.n_f = len(self.function_set)
        self.n_fnode = self.n_row * self.n_col
        if self.levels_back is None:
            self.levels_back = self.n_row * self.n_col + self.n_input + 1


class Node:
    def __init__(self, no, func, arity, inputs=[], start_gidx=None):
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


def create_genes_and_bounds(cp: CGPParameters):
    genes = []
    uppers, lowers = [], []
    for i in range(cp.n_input + cp.n_eph, cp.n_input + cp.n_eph + cp.n_fnode):
        f_gene = random.randint(0, cp.n_f - 1)

        lowers.append(0)
        uppers.append(cp.n_f - 1)
        genes.append(f_gene)

        # next bits are input of the node function.
        col = (i - cp.n_input - cp.n_eph) // cp.n_row
        up = cp.n_input + cp.n_eph + col * cp.n_row - 1
        low = max(0, up - cp.levels_back)
        for i_arity in range(cp.max_arity):
            lowers.append(low)
            uppers.append(up)
            in_gene = random.randint(low, up)
            genes.append(in_gene)
    # output genes
    up = cp.n_input + cp.n_eph + cp.n_fnode - 1
    low = max(0, up - cp.levels_back)
    for i in range(cp.n_output):
        lowers.append(low)
        uppers.append(up)
        out_gene = random.randint(low, up)
        genes.append(out_gene)

    return genes, (lowers, uppers)


def create_nodes(cp, genes):
    nodes = []
    for i in range(cp.n_input + cp.n_eph):
        nodes.append(Node(i, None, 0, []))

    f_pos = 0
    for i in range(cp.n_fnode):
        f_gene = genes[f_pos]
        f = cp.function_set[f_gene]
        input_genes = genes[f_pos + 1: f_pos + f.arity + 1]
        nodes.append(Node(i + cp.n_input + cp.n_eph, f, f.arity, input_genes, start_gidx=f_pos))
        f_pos += cp.max_arity + 1

    idx_output_node = cp.n_input + cp.n_eph + cp.n_fnode
    for gene in genes[-cp.n_output:]:
        nodes.append(Node(idx_output_node, None, 0, [gene], start_gidx=f_pos))
        f_pos += 1
        idx_output_node += 1

    return nodes


def get_active_paths(nodes):
    stack = []
    active_path, active_paths = [], []
    for node in reversed(nodes):
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
            stack.append(nodes[input])

    if len(active_path) > 0:
        active_paths.append(list(reversed(active_path)))

    return active_paths


def report(indiv=None, gen=None):
    def _sub(flist):
        str_list = []
        for f in flist:
            str_list.append(str(f)[:10]+'..')
        return str_list

    if indiv:
        print('|', format(gen, ' ^10'), '|', format(str(indiv.fitness)[:10]+'..', ' ^24'),
              '|', format(str(_sub(indiv.fitness_list)), ' ^80'), '|',
              format(str(indiv.get_cgp_expressions())[:60]+'..', ' ^80'), '|')
    else:
        print(format('', '_^207'))
        print('|', format('Gen', ' ^10'), '|', format('BestFitness', ' ^24'),
              '|', format('BestFitnessList', ' ^80'), '|', format('BestExpression', ' ^80'), '|')


def save_checkpoint(population, conv_f, checkpoint_dir):
    io.mkdir(checkpoint_dir)
    pop_dir = os.path.join(checkpoint_dir, 'populations')
    io.mkdir(pop_dir)

    population = sorted(population, key=lambda x: x.fitness)
    for i, indiv in enumerate(population):
        with open(os.path.join(pop_dir, 'SRNet_{}'.format(i)), 'wb') as f:
            pickle.dump(indiv, f)

    np.savetxt(os.path.join(checkpoint_dir, 'conv_f'), conv_f)