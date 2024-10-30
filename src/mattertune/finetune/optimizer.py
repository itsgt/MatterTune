from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

import nshconfig as C
import torch
from typing_extensions import override


class OptimizerBaseConfig(C.Config, ABC):
    lr: C.PositiveFloat
    """Learning rate."""

    @abstractmethod
    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> torch.optim.Optimizer: ...


class AdamConfig(OptimizerBaseConfig):
    name: Literal["Adam"] = "Adam"
    """Name of the optimizer."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""

    @override
    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ):
        return torch.optim.Adam(
            parameters,
            lr=self.lr,
            eps=self.eps,
            betas=self.betas,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class AdamWConfig(OptimizerBaseConfig):
    name: Literal["AdamW"] = "AdamW"
    """Name of the optimizer."""
    eps: C.NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[C.PositiveFloat, C.PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: C.NonNegativeFloat = 0.01
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""

    @override
    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ):
        return torch.optim.AdamW(
            parameters,
            lr=self.lr,
            eps=self.eps,
            betas=self.betas,
            weight_decay=self.weight_decay,
            amsgrad=self.amsgrad,
        )


class SGDConfig(OptimizerBaseConfig):
    name: Literal["SGD"] = "SGD"
    """Name of the optimizer."""
    momentum: C.NonNegativeFloat = 0.0
    """Momentum."""
    weight_decay: C.NonNegativeFloat = 0.0
    """Weight decay."""
    nestrov: bool = False
    """Whether to use nestrov."""

    @override
    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ):
        return torch.optim.SGD(
            parameters,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            nesterov=self.nestrov,
        )


OptimizerConfig: TypeAlias = Annotated[
    AdamConfig | AdamWConfig | SGDConfig,
    C.Field(
        discriminator="name",
    ),
]
