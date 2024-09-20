from abc import ABC, abstractmethod
from typing import Literal, Generic
import contextlib
from ..protocol import TBatch
import torch.nn as nn


class OutputHeadBaseConfig(ABC, Generic[TBatch]):
    """
    Base class for the configuration of the output head
    """
    
    head_name: str
    """The name of the output head"""
    target_name: str
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0   
    """The coefficient of the loss function"""
    reduction: Literal["mean", "sum", "none"] = "sum"
    """
    The reduction method
    For example, the total_energy is the sum of the energy of each atom
    """
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    
    @abstractmethod
    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls: type[nn.Module],
    ) -> nn.Module: ...
    
    @abstractmethod
    def is_classification(self) -> bool: ...
    
    @contextlib.contextmanager
    def model_forward_context(self, data: TBatch):
        yield

    def supports_inference_mode(self) -> bool:
        return True