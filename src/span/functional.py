import torch.nn as nn

def create_activation(name: str, **kwargs):
    activation_class = getattr(nn, name)
    return activation_class(**kwargs)
