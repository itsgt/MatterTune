from typing import Literal
from collections.abc import Iterable
from pydantic import BaseModel, PositiveFloat, NonNegativeFloat
from abc import ABC, abstractmethod
import torch


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
        
        
        
## TODO: Consider a more generic approach to handle optimizers
class GeneralOptimizerConfig(OptimizerBaseConfig):
    optimizer_class: type
    optimizer_params: dict

    def construct_optimizer(
        self,
        parameters: Iterable[torch.nn.Parameter],
    ):
        return self.optimizer_class(
            parameters,
            **self.optimizer_params,
        )

# optimizer_config = GeneralOptimizerConfig(
#     lr=0.001,
#     optimizer_class=torch.optim.Adam,
#     optimizer_params={
#         'lr': 0.001,
#         'betas': (0.9, 0.999),
#         'eps': 1e-8,
#         'weight_decay': 0.0,
#     }
# )

