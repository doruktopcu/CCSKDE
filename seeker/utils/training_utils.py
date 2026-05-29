import torch.optim as optim
from functools import partial

def init_optimizer(type_str, **kwargs):
    if type_str.lower() == 'adam':
        opt_f = optim.Adam
    elif type_str.lower() == 'adamx':
        opt_f = optim.Adamax
    elif type_str.lower() == 'sgd':
        opt_f = optim.SGD
    else:
        return None

    return partial(opt_f, **kwargs)