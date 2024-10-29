from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Annotated, Literal, TypeAlias

import torch
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat


class OptimizerBaseConfig(BaseModel, ABC):
    lr: PositiveFloat
    """Learning rate."""

    @abstractmethod
    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ) -> torch.optim.Optimizer: ...


class AdamConfig(OptimizerBaseConfig):
    name: Literal["Adam"] = "Adam"
    """Name of the optimizer."""
    eps: NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[PositiveFloat, PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: NonNegativeFloat = 0.0
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""

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
    eps: NonNegativeFloat = 1e-8
    """Epsilon."""
    betas: tuple[PositiveFloat, PositiveFloat] = (0.9, 0.999)
    """Betas."""
    weight_decay: NonNegativeFloat = 0.01
    """Weight decay."""
    amsgrad: bool = False
    """Whether to use AMSGrad variant of Adam."""

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
    momentum: NonNegativeFloat = 0.0
    """Momentum."""
    weight_decay: NonNegativeFloat = 0.0
    """Weight decay."""
    nestrov: bool = False
    """Whether to use nestrov."""

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
    Field(
        discriminator="name",
    ),
]
