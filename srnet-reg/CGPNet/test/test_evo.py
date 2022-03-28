import torch
import pyswarms as ps
import numpy as np
import sympy as sp

from CGPNet.layers import OneExpCGPLayer, BaseCGP, OneExpOneOutCGPLayer
from CGPNet.methods import LBFGSTrainer, Evolution, SGDTrainer, PSOTrainer, NewtonTrainer
from CGPNet.nets import OneLinearCGPNet, OneVectorCGPNet
from data_utils import io


def test_evolution():
    data_dir = '/home/luoyuanzhen/Datasets/regression/feynman/feynman2_nn/'
    data_list = io.get_nn_datalist(data_dir)

    trainer = LBFGSTrainer(n_epoch=5, end_to_end=True)
    evolutioner = Evolution(clas_net=OneLinearCGPNet, clas_cgp=OneExpCGPLayer)
    evolutioner.start(data_list=data_list,
                      trainer=trainer,
                      n_pop=100,
                      n_gen=100,
                      verbose=1)


def test_pso():
    data_dir = '/home/luoyuanzhen/Datasets/regression/feynman/feynman2_nn/'
    data_list = io.get_nn_datalist(data_dir)

    trainer = PSOTrainer(n_epoch=100, end_to_end=True)
    evolutioner = Evolution(clas_net=OneLinearCGPNet, clas_cgp=OneExpCGPLayer)
    evolutioner.start(data_list=data_list,
                      trainer=trainer,
                      n_pop=100,
                      n_gen=100,
                      verbose=1)


def test_one_vector_net():
    data_dir = '/home/luoyuanzhen/Datasets/regression/feynman/feynman2_nn/'
    data_list = io.get_nn_datalist(data_dir)

    trainer = NewtonTrainer(end_to_end=False)
    evolutioner = Evolution(clas_net=OneVectorCGPNet, clas_cgp=OneExpOneOutCGPLayer)
    evolutioner.start(data_list=data_list,
                      trainer=trainer,
                      n_pop=100,
                      n_gen=100,
                      verbose=1)


def test_v1_net():
    data_dir = '/home/luoyuanzhen/Datasets/regression/feynman/feynman2_nn/'
    data_list = io.get_nn_datalist(data_dir)


def f(pos):
    return np.full(pos.shape[0], np.inf)


def test_pso2():
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=55, options=options)
    optimizer.optimize(f, iters=100)


def test_np():
    x = np.full(100, np.nan)
    if np.isnan(x).all():
        print(np.isnan(x).all())
    else:
        print('not nan')


if __name__ == '__main__':
    test_one_vector_net()
    # test_evolution()
    # test_pso()
    # test_pso()
    # test_np()
