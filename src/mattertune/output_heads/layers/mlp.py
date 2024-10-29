from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from typing import Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn
from typing_extensions import override


class ScaledSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = nn.SiLU()

    def forward(self, x):
        return self._activation(x) * self.scale_factor


@runtime_checkable
class LinearModuleConstructor(Protocol):
    def __call__(
        self, in_features: int, out_features: int, bias: bool = True
    ) -> nn.Module: ...


class ResidualSequential(nn.Sequential):
    @override
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + super().forward(input)


def MLP(
    dims: Sequence[int],
    activation_cls: nn.Module | Callable[[], nn.Module] = nn.ReLU,
    bias: bool = True,
    no_bias_scalar: bool = True,
    ln: bool | Literal["pre", "post"] = False,
    dropout: float | None = None,
    residual: bool = False,
    pre_layers: Sequence[nn.Module] = [],
    post_layers: Sequence[nn.Module] = [],
    linear_cls: LinearModuleConstructor = nn.Linear,
):
    """
    Constructs a multi-layer perceptron (MLP) with the given dimensions and activation function.

    Args:
        dims (Sequence[int]): List of integers representing the dimensions of the MLP.
        activation_cls (Callable[[], nn.Module]): Activation function to use between layers.
        bias (bool, optional): Whether to include bias terms in the linear layers. Defaults to True.
        no_bias_scalar (bool, optional): Whether to exclude bias terms when the output dimension is 1. Defaults to True.
        ln (bool | Literal["pre", "post"], optional): Whether to apply layer normalization before or after the linear layers. Defaults to False.
        dropout (float | None, optional): Dropout probability to apply between layers. Defaults to None.
        residual (bool, optional): Whether to use residual connections between layers. Defaults to False.
        pre_layers (Sequence[nn.Module], optional): List of layers to insert before the linear layers. Defaults to [].
        post_layers (Sequence[nn.Module], optional): List of layers to insert after the linear layers. Defaults to [].

    Returns:
        nn.Sequential: The constructed MLP.
    """

    if len(dims) < 2:
        raise ValueError("mlp requires at least 2 dimensions")
    if ln is True:
        ln = "pre"
    elif isinstance(ln, str) and ln not in ("pre", "post"):
        raise ValueError("ln must be a boolean or 'pre' or 'post'")

    layers: list[nn.Module] = []
    if ln == "pre":
        layers.append(nn.LayerNorm(dims[0]))

    layers.extend(pre_layers)

    for i in range(len(dims) - 1):
        in_features = dims[i]
        out_features = dims[i + 1]
        bias_ = bias and not (no_bias_scalar and out_features == 1)
        layers.append(linear_cls(in_features, out_features, bias=bias_))
        if dropout is not None:
            layers.append(nn.Dropout(dropout))
        if i < len(dims) - 2:
            match activation_cls:
                case nn.Module():
                    # In this case, we create a deep copy of the module to avoid sharing parameters (if any).
                    layers.append(copy.deepcopy(activation_cls))
                case Callable():
                    layers.append(activation_cls())
                case _:
                    raise ValueError(
                        "activation_cls must be a nn.Module or a callable that returns a nn.Module"
                    )

    layers.extend(post_layers)

    if ln == "post":
        layers.append(nn.LayerNorm(dims[-1]))

    cls = ResidualSequential if residual else nn.Sequential
    return cls(*layers)
