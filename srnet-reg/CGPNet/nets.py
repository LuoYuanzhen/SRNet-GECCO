import torch
from torch import nn

from CGPNet.layers import BaseCGP, OneExpOneOutCGPLayer, LinearLayer, MulExpCGPLayer, OneExpCGPLayer
from CGPNet.params import NetParameters, CGPParameters


net_cgp_dims = {
    BaseCGP.__name__: (None, None),
    OneExpOneOutCGPLayer.__name__: (None, 1),
    MulExpCGPLayer.__name__: (1, None),
    OneExpCGPLayer.__name__: (1, 1)
}


def _create_cgp_layer(clas_cgp, net_params, idx_neurons, genes=None, ephs=None):
    dims = net_cgp_dims[clas_cgp.__name__]
    n_inputs = dims[0] if dims[0] else net_params.neurons[idx_neurons-1]
    n_outputs = dims[1] if dims[1] else net_params.neurons[idx_neurons]

    cgp_params = CGPParameters(n_inputs=n_inputs,
                               n_outputs=n_outputs,
                               n_rows=net_params.n_rows,
                               n_cols=net_params.n_cols,
                               levels_back=net_params.levels_back,
                               function_set=net_params.function_set,
                               n_eph=net_params.n_eph)
    return clas_cgp(cgp_params, genes, ephs)


class BaseCGPNet(nn.Module):
    def __init__(self, net_params, cgp_layers=None, clas_cgp=BaseCGP):
        super(BaseCGPNet, self).__init__()
        self.net_params = net_params

        self.clas_cgp = clas_cgp
        self.neurons = net_params.neurons
        self.n_layer = len(self.neurons)  # n_layer includes input layer, hidden layers, output layer.
        self.fitness = None
        self.fitness_list = []

        self.cgp_layers = []
        if cgp_layers:
            self.cgp_layers = cgp_layers
        else:
            for i in range(1, self.n_layer):
                self.cgp_layers.append(_create_cgp_layer(self.clas_cgp, self.net_params, i))

    def get_cgp_expressions(self):
        """Note that this method only return sympy expressions of each layer, not including w"""
        exprs = []
        for layer in self.cgp_layers:
            exprs.append(layer.get_expressions())
        return exprs

    def get_genes(self):
        """get a list of genes in CGPLayer"""
        genes_list = []
        for layer in self.cgp_layers:
            genes_list.append(layer.get_genes())
        return genes_list

    def get_ephs(self):
        """get a list of ephs in CGPLayer"""
        ephs_list = []
        for layer in self.cgp_layers:
            ephs_list.append(layer.get_ephs())
        return ephs_list


class OneVectorCGPNet(BaseCGPNet):
    def __init__(self, net_params, cgp_layers=None, nn_layers=None, clas_cgp=OneExpOneOutCGPLayer):
        super(OneVectorCGPNet, self).__init__(net_params, cgp_layers, clas_cgp=clas_cgp)

        self.add_bias = self.net_params.add_bias
        self.nn_layers = []
        if nn_layers:
            self.nn_layers = nn_layers
        else:
            for i in range(1, self.n_layer):
                self.nn_layers.append(LinearLayer(1, self.neurons[i], add_bias=self.add_bias))

    def __call__(self, x):
        layer_output = x
        outputs = []

        for cgp_layer, nn_layer in zip(self.cgp_layers, self.nn_layers):
            layer_output = nn_layer(cgp_layer(layer_output))
            outputs.append(layer_output)

        return outputs

    def get_ws(self):
        """get all the weights in nn_layers, and make them detach, shape: (n_input, n_output)"""
        w_list = []
        for layer in self.nn_layers:
            w_list.append(layer.get_weight())
        return w_list

    def get_bias(self):
        """get all the biases in nn_layers, and make them detach, shape: (n_output, 1)"""
        bias_list = []
        if self.add_bias:
            for layer in self.nn_layers:
                bias_list.append(layer.get_bias().view(1, -1))
        return bias_list

    def get_net_parameters(self):
        parameters = []
        for layer in self.nn_layers:
            parameters += list(layer.parameters())
        return parameters

    def clone(self):
        cgp_layers, nn_layers = [], []
        for cgp, linear in zip(self.cgp_layers, self.nn_layers):
            cgp_layers.append(cgp.clone())
            nn_layers.append(linear.clone())
        return OneVectorCGPNet(self.net_params, cgp_layers, nn_layers)

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   genes_list, ephs_list, w_list, bias_list=None, clas_cgp=MulExpCGPLayer):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        add_bias = bias_list is not None
        if net_params.add_bias != add_bias:
            raise ValueError(f'net_params.add_bias is {net_params.add_bias} while bias_list is {bias_list}')
        if len(genes_list) != len(ephs_list) != len(w_list) and n_layers != len(genes_list) + 1:
            raise ValueError('length of genes, ephs, W should all be equal to n_layer - 1!')
        if add_bias and len(bias_list) != len(w_list):
            raise ValueError('length of bias, genes, ephs, W should all be eqaul to n_layer - 1!')

        cgp_layers, nn_layers = [], []
        for i in range(1, n_layers):
            genes, ephs = genes_list[i - 1], torch.tensor(ephs_list[i - 1])
            W = torch.tensor(w_list[i - 1])
            b = torch.tensor(bias_list[i-1]) if add_bias else None

            cgp_layers.append(_create_cgp_layer(clas_cgp, net_params, i, genes, ephs))
            nn_layers.append(LinearLayer(weight=W, bias=b, add_bias=add_bias))

        return cls(net_params, cgp_layers, nn_layers, clas_cgp=clas_cgp)

    def generate_offspring(self, gidxs_list, mutant_genes_list):
        cgp_layers, nn_layers = [], []
        for n, cgp, gidxs, mutant_genes in zip(self.nn_layers, self.cgp_layers, gidxs_list, mutant_genes_list):
            cgp_layers.append(cgp.generate_offspring(gidxs, mutant_genes))
            nn_layers.append(n.clone())
        return OneVectorCGPNet(self.net_params, cgp_layers, nn_layers)


class LinearOutputCGPNet(OneVectorCGPNet):
    def __init__(self, net_params, cgp_layers=None, nn_layers=None, last_nn_layer=None, clas_cgp=OneExpOneOutCGPLayer):
        # knowing that len(nn_layers) is equal to len(cgp_layers)
        # we first ignore the last layer (output layer)
        neurons = net_params.neurons[:]
        net_params.neurons = net_params.neurons[:-1]
        super(LinearOutputCGPNet, self).__init__(net_params, cgp_layers, nn_layers, clas_cgp=clas_cgp)

        # alfter construct the cgp_layers and nn_layers, we can set the params back
        self.net_params.neurons = neurons
        self.neurons = neurons

        self.last_nn_layer = last_nn_layer
        # then we construct our linear output layer
        if self.last_nn_layer is None:
            self.last_nn_layer = LinearLayer(self.neurons[-2], self.neurons[-1], add_bias=self.add_bias)

    def __call__(self, x):
        outputs = super(LinearOutputCGPNet, self).__call__(x)
        outputs.append(self.last_nn_layer(outputs[-1]))
        return outputs

    def get_ws(self):
        """get all the weights in nn_layers, and make them detach, shape: (n_input, n_output)"""
        w_list = super(LinearOutputCGPNet, self).get_ws().append(self.last_nn_layer.get_weight())
        return w_list

    def get_bias(self):
        """get all the biases in nn_layers, and make them detach, shape: (n_output, 1)"""
        bias_list = super(LinearOutputCGPNet, self).get_bias().append(self.last_nn_layer.get_bias())
        return bias_list

    def get_net_parameters(self):
        parameters = super(LinearOutputCGPNet, self).get_net_parameters().append(self.last_nn_layer.parameters())
        return parameters

    @classmethod
    def encode_net(cls,
                   net_params: NetParameters,
                   genes_list, ephs_list, w_list, bias_list=None, clas_cgp=MulExpCGPLayer):
        """class method that encode the net based on existing params"""
        neurons = net_params.neurons
        n_layers = len(neurons)

        add_bias = bias_list is not None
        if net_params.add_bias != add_bias:
            raise ValueError(f'net_params.add_bias is {net_params.add_bias} while bias_list is {bias_list}')
        if len(genes_list) != len(ephs_list) != len(w_list)-1 and n_layers != len(genes_list) + 2:
            raise ValueError('length of genes, ephs should all be equal to n_layer - 2, and the length of w_list should be equal to n_layer-1!')
        if add_bias and len(bias_list) != len(w_list):
            raise ValueError('length of bias, genes, ephs, W should all be eqaul to n_layer - 1!')

        cgp_layers, nn_layers = [], []
        for i in range(1, n_layers-1):
            genes, ephs = genes_list[i - 1], torch.tensor(ephs_list[i - 1])
            W = torch.tensor(w_list[i - 1])
            b = torch.tensor(bias_list[i-1]) if add_bias else None

            cgp_layers.append(_create_cgp_layer(clas_cgp, net_params, i, genes, ephs))
            nn_layers.append(LinearLayer(weight=W, bias=b, add_bias=add_bias))

        # construct last nn layer:
        w = torch.tensor(w_list[-1])
        b = torch.tensor(bias_list[-1]) if add_bias else None
        last_nn_layer = LinearLayer(weight=w, bias=b, add_bias=add_bias)

        return cls(net_params, cgp_layers, nn_layers, last_nn_layer, clas_cgp=clas_cgp)

    def generate_offspring(self, gidxs_list, mutant_genes_list):
        cgp_layers = []
        nn_layers = []
        for n, cgp, gidxs, mutant_genes in zip(self.nn_layers, self.cgp_layers, gidxs_list, mutant_genes_list):
            cgp_layers.append(cgp.generate_offspring(gidxs, mutant_genes))
            nn_layers.append(n.clone())

        last_nn_layer = self.last_nn_layer.clone()
        return LinearOutputCGPNet(self.net_params, cgp_layers, nn_layers, last_nn_layer)





