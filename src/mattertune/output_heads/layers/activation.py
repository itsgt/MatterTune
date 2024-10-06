import torch
import torch.nn as nn

class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor
    
    
def get_activation_cls(activation: str) -> type[nn.Module]:
    """
    Get the activation class from the activation name
    """
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU
    elif activation == "silu" or activation == "swish":
        return nn.SiLU
    elif activation == "scaled_silu" or activation == "scaled_swish":
        return ScaledSiLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "identity":
        return nn.Identity
    else:
        raise ValueError(f"Activation {activation} is not supported")