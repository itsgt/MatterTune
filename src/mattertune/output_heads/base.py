from abc import ABC, abstractmethod
from typing import Generic, Literal
import contextlib
from pydantic import BaseModel
from mattertune.data_structures import TMatterTuneBatch
from mattertune.finetune.loss import LossConfig
import torch.nn as nn


class OutputHeadBaseConfig(BaseModel, ABC, Generic[TMatterTuneBatch]):
    """
    Base class for the configuration of the output head
    """
    model_config = {
        'arbitrary_types_allowed': True
    }
    pred_type: Literal["scalar", "vector", "tensor", "classification"]
    """The prediction type of the output head"""
    target_name: str
    """The name of the target output by this head"""
    loss: LossConfig
    """The loss configuration for the target."""
    loss_coefficient: float = 1.0   
    """The coefficient of the loss function"""
    freeze: bool = False
    """Whether to freeze the output head"""
    
    @abstractmethod
    def construct_output_head(
        self,
    ) -> nn.Module: ...
    
    @abstractmethod
    def is_classification(self) -> bool: 
        return False
    
    def get_num_classes(self) -> int:
        return 0
    
    @contextlib.contextmanager
    def model_forward_context(self, data: TMatterTuneBatch):
        """
        Model forward context manager.
        Make preparations before the forward pass for the output head.
        For example, set auto_grad to True for pos if using Gradient Force Head.
        """
        yield

    def supports_inference_mode(self) -> bool:
        return True