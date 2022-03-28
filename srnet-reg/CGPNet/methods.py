import itertools
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch.optim
from joblib import Parallel, delayed
from torch import nn
from torch.autograd import Variable

from CGPNet.functions import default_functions
from CGPNet.params import NetParameters
from CGPNet.utils import partition_n_jobs, report, probabilistic_mutate_net, parallel_optimize


class OptimTrainer:
    """provide 4 training ways: SGD(Adam), Newton, LBFGS, PSO
    All of them use tool pyTorch"""
    def __init__(self, n_epoch, end_to_end, l_func=nn.MSELoss()):
        self.l_func = l_func
        self.n_epoch = n_epoch
        self.end_to_end = end_to_end
        self.optimizer = None

    def evaluate(self, net, x, idx, y):
        with torch.no_grad():
            loss = self.l_func(net(x)[idx], y)
        return loss.item()

    def train(self, net, data_list):
        pass


class SGDTrainer(OptimTrainer):

    def train(self, net, data_list):
        self.optimizer = torch.optim.Adam(net.get_net_parameters())
        x = Variable(data_list[0], requires_grad=True)
        l_trend = []

        def _sgd_update(_idx):
            for epoch in range(self.n_epoch):
                y_hat = net(x)[_idx]
                loss = self.l_func(y_hat, y)
                self.optimizer.zero_grad()
                if loss.requires_grad:
                    loss.backward()
                self.optimizer.step()
                l_trend.append(loss.item())

        if self.end_to_end:
            y = Variable(data_list[-1])
            _sgd_update(-1)
        else:
            for idx in range(len(data_list) - 1):
                l_trend = []
                y = Variable(data_list[idx + 1])
                _sgd_update(idx)

        return l_trend


class LBFGSTrainer(OptimTrainer):

    def __init__(self, n_epoch, end_to_end, l_func=nn.MSELoss()):
        super(LBFGSTrainer, self).__init__(n_epoch, end_to_end, l_func)

    def train(self, net, data_list):
        if self.optimizer is None:
            self.optimizer = torch.optim.LBFGS(net.get_net_parameters(), history_size=10, max_iter=5)
        x = Variable(data_list[0], requires_grad=True)
        l_trend = []

        if self.end_to_end:
            y = Variable(data_list[-1])
            for epoch in range(self.n_epoch):
                y_hat = net(x)[-1]
                l = self._lbfgs_closure(y_hat, y)
                l_trend.append(l)
        else:
            # we optimize weights one by one to make every hidden layer fits the data
            for idx in range(len(data_list) - 1):
                l_trend = self.apply_optim(data_list[idx+1], net(x)[idx-1], net.nn_layers[idx])

        return l_trend

    def apply_optim(self, y_true, layer_input, nn_layer):
        self.optimizer = torch.optim.LBFGS(nn_layer.parameters(), history_size=10, max_iter=5)
        l_trend = []
        # y = Variable(y_true)
        x = layer_input
        for epoch in range(self.n_epoch):

            self.optimizer.zero_grad()
            loss = self.l_func(nn_layer(x), y_true)
            loss.backward(retain_graph=True)

            self.optimizer.step(lambda: self.l_func(nn_layer(x), y_true))
            l_trend.append(loss.item())
        return l_trend


class NewtonTrainer(OptimTrainer):
    """Only apply on the only one linear layer net, whose layer is named nn_layers"""

    def __init__(self, end_to_end, l_func=nn.MSELoss()):
        super(NewtonTrainer, self).__init__(1, end_to_end, l_func)

    def train(self, net, data_list):
        x = data_list[0]
        if self.end_to_end:
            pass
        else:
            for i, nn_layer in enumerate(net.nn_layers):
                self.apply_optim(data_list[i+1], net(x)[i], nn_layer)

    def apply_optim(self, prediction, y_true, nn_layer):
        loss = self.l_func(y_true, prediction)
        for param in nn_layer.parameters():
            gradient, hessian_inv = self._get_inverse_hessian(loss, param)
            if hessian_inv is not None:
                if len(gradient.shape) == 2 and gradient.shape[1] != hessian_inv.shape[0]:
                    param.data = param.data - torch.matmul(hessian_inv, gradient)
                else:
                    param.data = param.data - torch.matmul(gradient, hessian_inv)
            if param.grad is not None:
                param.grad.data.zero_()

    @staticmethod
    def _get_inverse_hessian(loss, param):
        """Calculate the inverse Hessian matrix"""

        # save the gradient
        gradient = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True)[0]
        if not torch.all(torch.isfinite(gradient)):
            return None, None

        hessian = []
        for grad in gradient.view(-1):
            order2_gradient = torch.autograd.grad(grad, param, retain_graph=True)[0]  # weight.shape
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


evolution_strategy = ['fitness_select', 'chromosome_select']


class Evolution:
    def __init__(self,
                 evo_params,
                 clas_net,
                 clas_cgp
                 ):
        self.evo_params = evo_params
        self.clas_net = clas_net
        self.clas_cgp = clas_cgp

        self.neurons = None
        self.n_rows = evo_params['n_rows']
        self.n_cols = evo_params['n_cols']
        self.levels_back = evo_params['levels_back']

        if self.levels_back is None:
            self.levels_back = self.n_cols * self.n_rows + 1

        self.function_set = evo_params['function_set']
        self.n_eph = evo_params['n_eph']
        self.add_bias = evo_params['add_bias']

        self.evo_strategy = evo_params['evolution_strategy']
        if self.evo_strategy is None:
            self.evo_strategy = 'fitness_select'
        elif self.evo_strategy not in evolution_strategy:
            raise ValueError(f"Not evolution strategy name is {self.evo_strategy}.")

    def _set_parameters(self):
        self.net_parameters = NetParameters(neurons=self.neurons,
                                            n_rows=self.n_rows,
                                            n_cols=self.n_cols,
                                            levels_back=self.levels_back,
                                            function_set=self.function_set,
                                            n_eph=self.n_eph,
                                            add_bias=self.add_bias)

    @staticmethod
    def _get_protected_loss(output, y):
        loss = nn.MSELoss()(output, y)
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(float('inf'))
        return loss.item()

    @staticmethod
    def _calculate_weighted_fitness(fitness_list):
        fitness = 0.0
        if len(fitness_list) == 1:
            sam_weights = [1.]
        else:
            n_hidden = len(fitness_list) - 1
            weight = 1 / n_hidden
            sam_weights = [weight for _ in range(n_hidden)] + [1.]

        for layer_fitness, weight in zip(fitness_list, sam_weights):
            fitness += weight * layer_fitness

        return fitness

    def _apply_optim(self, trainer, nn_layer, layer_input, y_true):
        if trainer.__class__.__name__ == "LBFGSTrainer":
            trainer.apply_optim(y_true, layer_input, nn_layer)
        else:
            trainer.apply_optim(nn_layer(layer_input), y_true, nn_layer)

    def _evaluate_fitness(self, pop, data_list, valid_data_list):
        if valid_data_list is None:
            x = data_list[0]
            nn_outputs = data_list[1:]
        else:
            x = torch.vstack((data_list[0], valid_data_list[0]))
            nn_outputs = [torch.vstack((data_list[i], valid_data_list[i])) for i in range(1, len(data_list))]
        for indiv in pop:
            predictions = indiv(x)
            indiv.fitness_list = []
            for nn_output, prediction in zip(nn_outputs, predictions):
                loss = self._get_protected_loss(nn_output, prediction)
                indiv.fitness_list.append(loss)
            indiv.fitness = self._calculate_weighted_fitness(indiv.fitness_list)

    def _apply_evolution_strategy(self, population, trainer, data_list, valid_data_list):
        parent = None
        if self.evo_strategy == 'fitness_select':
            for indiv in population:
                trainer.train(net=indiv, data_list=data_list)
            self._evaluate_fitness(population, data_list, valid_data_list)
            parent = min(population, key=lambda x: x.fitness)
        elif self.evo_strategy == 'chromosome_select':
            cgp_layers, nn_layers = [], []
            num_chrom = len(population[0].cgp_layers)

            newton_input = data_list[0]
            if valid_data_list is not None:
                valid_input = torch.vstack((data_list[0], valid_data_list[0]))
            else:
                valid_input = newton_input

            # Training each chromosome one by one, and select a best parent.
            for chrom_idx in range(num_chrom):
                # for each chromosome, apply Newton method.
                chrom_losses = []
                for i, indiv in enumerate(population):
                    chrom_cgp, chrom_linear = indiv.cgp_layers[chrom_idx], indiv.nn_layers[chrom_idx]
                    # training with training set
                    self._apply_optim(
                        trainer=trainer,
                        nn_layer=chrom_linear,
                        layer_input=chrom_cgp(newton_input),
                        y_true=data_list[chrom_idx+1]
                    )
                    # trainer.apply_optim(chrom_linear(chrom_cgp(newton_input)), data_list[chrom_idx+1], chrom_linear)
                    # choose from validation set
                    his = chrom_linear(chrom_cgp(valid_input))
                    if valid_data_list is not None:
                        hi = torch.vstack((data_list[chrom_idx+1], valid_data_list[chrom_idx+1]))
                    else:
                        hi = data_list[chrom_idx+1]
                    chrom_losses.append(self._get_protected_loss(his, hi))
                # choose the best chromosome from all (chrom_idx)th chromosomes of all individuals.
                best_idx = np.argmin(chrom_losses)

                best_cgp = population[best_idx].cgp_layers[chrom_idx]
                best_linear = population[best_idx].nn_layers[chrom_idx]

                cgp_layers.append(best_cgp)
                nn_layers.append(best_linear)

                newton_input = best_linear(best_cgp(newton_input))
                valid_input = best_linear(best_cgp(valid_input))

            last_nn_layer = None
            if len(population[0].cgp_layers) < len(population[0].neurons)-1:
                # construct last linear layer
                chrom_losses = []
                for i, indiv in enumerate(population):
                    chrom = indiv.last_nn_layer
                    # with torch.no_grad():
                    #     pre_loss = torch.nn.MSELoss()(data_list[-1], chrom(newton_input))
                    self._apply_optim(
                        trainer=trainer,
                        nn_layer=chrom,
                        layer_input=newton_input,
                        y_true=data_list[-1])
                    # trainer.apply_optim(chrom(newton_input), data_list[-1], chrom)
                    # with torch.no_grad():
                    #     alfter_loss = torch.nn.MSELoss()(data_list[-1], chrom(newton_input))
                    # print(f'Previous loss: {pre_loss.item()}; After loss: {alfter_loss.item()}')
                    his = chrom(valid_input)
                    if valid_data_list is not None:
                        hi = torch.vstack((data_list[-1], valid_data_list[-1]))
                    else:
                        hi = data_list[-1]
                    chrom_losses.append(self._get_protected_loss(his, hi))
                last_nn_layer = population[np.argmin(chrom_losses)].last_nn_layer

            # choose any individual in the population would be ok, since its layers would be replaced in the end.
            parent = population[0]
            # simpily replace its layers and fitness
            parent.cgp_layers, parent.nn_layers = cgp_layers, nn_layers
            if last_nn_layer is not None:
                parent.last_nn_layer = last_nn_layer
            # finally, evaluate its fitness.
            self._evaluate_fitness([parent], data_list, valid_data_list)

        return parent

    def start(self,
              data_list,
              trainer,
              valid_data_list=None
              ):
        self.neurons = [data.shape[1] for data in data_list]

        if len(data_list) != len(self.neurons):
            raise ValueError(f"Data_list's length {len(data_list)} != neurons' length {len(self.neurons)}")
        for data, n_neuron in zip(data_list, self.neurons):
            if data.shape[1] != n_neuron:
                raise ValueError(f"Shape[1] of data in data_list {data.shape[1]} != n_neuron {n_neuron}")

        self._set_parameters()

        random_state = self.evo_params['random_state']
        n_pop = self.evo_params['n_population']
        verbose = self.evo_params['verbose']
        n_gen = self.evo_params['n_generation']
        prob = self.evo_params['prob']
        stop_fitness = self.evo_params['stop_fitness']

        if random_state:
            random.seed(random_state)

        conv_f, population, history_elites = [], None, []
        parent, gen = None, 0
        if verbose:
            report()
        for gen in range(1, n_gen + 1):
            if not population:
                # init
                population = [self.clas_net(self.net_parameters, clas_cgp=self.clas_cgp) for _ in range(n_pop)]
            else:
                # mutate, note that the inital population would not be mutated
                # E(1,n-1)
                population = [parent] + \
                             [probabilistic_mutate_net(parent, prob)
                              for _ in range(n_pop - 1)]

            if not parent:
                parent = self._apply_evolution_strategy(population, trainer, data_list, valid_data_list)
            else:
                new_parent = self._apply_evolution_strategy(population[1:], trainer, data_list, valid_data_list)
                # print(f'parent.fitness:{parent.fitness}; new.fitness:{new_parent.fitness}')
                parent = new_parent if new_parent.fitness < parent.fitness else parent

            conv_f.append(parent.fitness)
            _add_history_elite(history_elites, parent)

            if verbose and gen % verbose == 0:
                report(parent, gen)
            if parent.fitness <= stop_fitness:
                break

        if gen < n_gen - 1:
            condition = 'reach stop fitness'
        else:
            condition = 'reach n generation'

        if verbose:
            print(f'Stop evolution, condition:{condition}')

        # population.sort(key=lambda x: x.fitness)
        history_elites.sort(key=lambda x: x.fitness)

        return history_elites, conv_f


def _add_history_elite(history_elites, elite):
    if len(history_elites) == 0 or history_elites[-1] != elite:
        history_elites.append(elite)





