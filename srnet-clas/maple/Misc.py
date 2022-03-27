import numpy as np
import torch


def normaliza_data(data, thresh=.0000000001):
    std = torch.std(data, dim=0)

    keep_feature_idx, = torch.where(std[:-1] > thresh)
    keep_feature_idx = torch.hstack([keep_feature_idx, torch.tensor(data.shape[1]-1)])

    data = data[:, keep_feature_idx]

    std, mean = torch.std_mean(data, dim=0)
    data = (data - mean) / std

    # convert to numpy array
    return data.numpy(), std, mean


# get LIME's coefficients for a particular point
# This num_samples is the default parameter from LIME's github implementation of explain_instance
def unpack_coefs(explainer, x, predict_fn, num_features, x_train, num_samples = 5000):
    d = x_train.shape[1]
    coefs = np.zeros((d))
    
    u = np.mean(x_train, axis = 0)
    sd = np.sqrt(np.var(x_train, axis = 0))
    
    exp = explainer.explain_instance(x, predict_fn, num_features=num_features, num_samples = num_samples)
    
    coef_pairs = exp.local_exp[1]
    for pair in coef_pairs:
        coefs[pair[0]] = pair[1]
    
    coefs = coefs / sd

    intercept = exp.intercept[1] - np.sum(coefs * u)

    return np.insert(coefs, 0, intercept)
