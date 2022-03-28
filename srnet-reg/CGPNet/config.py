from CGPNet.layers import OneExpOneOutCGPLayer, MulExpCGPLayer, OneExpCGPLayer
from CGPNet.methods import LBFGSTrainer, SGDTrainer, NewtonTrainer
from CGPNet.nets import OneVectorCGPNet, LinearOutputCGPNet

clas_net_map = {
    'OneVectorCGPNet': OneVectorCGPNet,
    'LinearOutputCGPNet': LinearOutputCGPNet
}


clas_cgp_map = {
    'OneExpOneOutCGPLayer': OneExpOneOutCGPLayer,
    'OneExpCGPLayer': OneExpCGPLayer,
    'MulExpCGPLayer': MulExpCGPLayer
}


clas_optim_map = {
    'SGD': SGDTrainer,
    'Newton': NewtonTrainer,
    'LBFGS': LBFGSTrainer
}






