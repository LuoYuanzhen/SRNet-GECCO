import copy
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from CGPNet.nets import CGPNet
from CGPNet.utils import SRNetDataset, report, save_checkpoint


def _loss_func(y_hat_list, y_list, item=False, regression=False):
    mse = nn.MSELoss()
    loss_list = []
    # calculate hidden loss
    for output, y in zip(y_hat_list[:-1], y_list[:-1]):
        l = mse(output, y)
        if item:
            l = l.item()
        loss_list.append(l)
    # calculate output loss
    if regression:
        l = mse(y_hat_list[-1], y_list[-1])
    else:
        label = y_list[-1].argmax(dim=1)
        l = nn.CrossEntropyLoss()(y_hat_list[-1], label)
    loss = l.item() if item else l
    loss_list.append(loss)

    # considering the situation that len(y_list) == 1
    weight = 1 / max(1, (len(y_list) - 1))
    for i in range(len(loss_list)-1):
        loss += weight * loss_list[i]

    return loss_list, loss


def _apply_optim(pop, x_loader, blackbox, device, regression):
    blackbox = blackbox.to(device)
    for indiv in pop:
        indiv = indiv.to(device)
        optimizer = torch.optim.LBFGS(indiv.parameters())
        for x_train in x_loader:
            x = x_train.to(device)
            with torch.no_grad():
                ys = blackbox(x)

            def closure():
                optimizer.zero_grad()
                train_y_hat_list = indiv(x)
                _, loss = _loss_func(train_y_hat_list, ys, regression=regression)
                loss.backward()
                return loss
            optimizer.step(closure)


def _select_parent(pop, x_loader, blackbox, regression):
    for indiv in pop:
        n_batch, fitness_list, fitness = 0, [], 0.0
        for x_test in x_loader:
            x = x_test
            with torch.no_grad():
                ys = blackbox(x)

            y_hat_list = indiv.predict(x)
            batch_f_list, batch_f = _loss_func(y_hat_list, ys, item=True, regression=regression)
            fitness_list.append(batch_f_list)
            fitness += batch_f

            n_batch += 1
        indiv.fitness_list, indiv.fitness = np.mean(fitness_list, axis=0), fitness / n_batch
        if np.isnan(indiv.fitness):
            indiv.fitness = float('inf')

    return min(pop, key=lambda x: x.fitness)


def mnnsr_es(pop, x_loader, blackbox, device, regression):
    n_chrome = pop[0].n_layer
    n_input, n_output, n_hiddens = pop[0].n_input, pop[0].n_output, pop[0].n_hiddens

    loss_func = torch.nn.MSELoss()
    parent = []
    blackbox = blackbox.to(device)
    for i in range(n_chrome):
        chrom_losses = []
        for indiv in pop:
            # we apply gradient-based training for each chrom
            indiv = indiv.to(device)
            chromosome = getattr(indiv, 'chrome{}'.format(i+1))
            optimizer = torch.optim.LBFGS(chromosome.parameters())
            for x_train in x_loader:
                x = x_train.to(device)
                with torch.no_grad():
                    y = blackbox(x)[i]

                def closure():
                    optimizer.zero_grad()
                    y_hat = indiv(x)[i]
                    loss = loss_func(y_hat, y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
            # calculate chrom's loss
            with torch.no_grad():
                losses = []
                for x_train in x_loader:
                    x = x_train.to(device)
                    y = blackbox(x)[i]
                    losses.append(loss_func(indiv(x)[i], y).item())
            aver_chrom_loss = np.mean(losses)
            if np.isnan(aver_chrom_loss):
                aver_chrom_loss = float('inf')
            chrom_losses.append(aver_chrom_loss)
            indiv = indiv.cpu()
        best_idx = np.argmin(chrom_losses)
        best_chrom = getattr(pop[best_idx], 'chrome{}'.format(i+1))
        parent.append(best_chrom)
    parent = CGPNet(n_input, n_output, n_hiddens, None, parent)
    blackbox = blackbox.cpu()
    return _select_parent([parent], x_loader, blackbox, regression)


def evolve_srnet(x_train, blackbox, params, checkpoint_dir=None, device='cpu', regression=False, prob_batch=1):
    assert device == 'cuda' or device == 'cpu'
    if device == 'cuda':
        assert torch.cuda.is_available()
    device = torch.device(device)

    # Due to the large #dimensions and multiple #layers, we do not load the y_train_list for saving the meomery
    batch_size = int(prob_batch * x_train.shape[0])
    x_train_dataset = SRNetDataset(x_train)
    x_train_loader = DataLoader(x_train_dataset, batch_size=batch_size, shuffle=True)

    # get the # of input dimensions
    x_instance = x_train[0:1, :]
    with torch.no_grad():
        y_instance = blackbox(x_instance)
    n_input, n_output = x_instance.shape[1], y_instance[-1].shape[1]
    n_hiddens = [data.shape[1] for data in y_instance[:-1]]

    start_time = time.time()
    conv_f, topn = [], []
    # initialization
    population = [CGPNet(n_input, n_output, n_hiddens, params) for _ in range(params['n_population'])]
    if params['verbose']:
        report()
    for gen in range(1, params['n_generation'] + 1):
        # LBFGS training the weight and bias
        if gen % params['optim_interval'] == 0:
            # note that the original parent would not be optimized again
            _apply_optim(population if gen == 1 else population[1:], x_train_loader, blackbox, device, regression)
        # evaluation and select
        parent = _select_parent(population, x_train_loader, blackbox, regression)
        # adding the parent to the topn list
        if parent not in topn:
            if len(topn) == 0:
                topn.append(parent)
            elif len(topn) < 20 and parent.fitness < max(topn, key=lambda x: x.fitness).fitness:
                topn.append(parent)
            else:
                max_indiv = max(topn, key=lambda x: x.fitness)
                if parent.fitness < max_indiv.fitness:
                   topn[topn.index(max_indiv)] = parent
        # saving the best fitness
        conv_f.append(parent.fitness)
        # saving the checkpoint
        if checkpoint_dir is not None and gen % 100 == 0:
            save_checkpoint(population, conv_f, checkpoint_dir)
        # report the training information
        if params['verbose'] and gen % params['verbose'] == 0:
            report(parent, gen)
        # stop condition reached
        if parent.fitness <= params['stop_fitness']:
            break
        # mutation
        population = [parent] + [parent.mutate(params['prob']) for _ in range(params['n_population'] - 1)]
    if params['verbose']:
        print('stop evolution at gen {}'.format(gen))

    topn.sort(key=lambda x: x.fitness)

    if len(conv_f) < params['n_generation']:
        conv_f = conv_f + [conv_f[-1] for _ in range(params['n_generation'] - len(conv_f))]

    return conv_f, topn, (time.time()-start_time) / 60

