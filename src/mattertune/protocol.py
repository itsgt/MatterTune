from abc import abstractmethod, ABC
# from collections.abc import MutableMapping
from typing import (
    Any,
    Generic,
    Protocol,
    Literal,
    runtime_checkable,
)
import torch
import torch.nn as nn
from typing_extensions import TypeVar, override
import jaxtyping as jt
import contextlib
from mattertune.finetune.data_module import RawData
from mattertune.finetune.loss import LossConfig


@runtime_checkable
class DataProtocol(Protocol):
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes"]
    pos: jt.Float[torch.Tensor, "num_nodes", 3]
    num_atoms: jt.Int[torch.Tensor, "1"]
    cell_displacement: jt.Float[torch.Tensor, "num_nodes", 3, 3]|None
    cell: jt.Float[torch.Tensor, "num_nodes", 3, 3]|None


TData = TypeVar("TData", bound=DataProtocol, infer_variance=True)


@runtime_checkable
class BatchProtocol(Protocol):
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes_in_batch"]
    pos: jt.Float[torch.Tensor, "num_nodes_in_batch", 3]
    num_atoms: jt.Int[torch.Tensor, "num_graphs_in_batch"]
    batch: jt.Int[torch.Tensor, "num_nodes_in_batch"]
    cell_displacement: jt.Float[torch.Tensor, "num_graphs_in_batch", 3, 3]|None
    cell: jt.Float[torch.Tensor, "num_graphs_in_batch", 3, 3]|None
    
    @abstractmethod
    def __len__(self) -> int: ...


TBatch = TypeVar("TBatch", bound=BatchProtocol, infer_variance=True)


@runtime_checkable
class BackBoneBaseOutputProtocol(Protocol):
    """
    The protocol of the output of the backbone model
    """
    pass

BackBoneBaseOutput = TypeVar("BackBoneBaseOutput", bound=BackBoneBaseOutputProtocol, infer_variance=True)


class BackBoneBaseModule(ABC, Generic[TBatch, BackBoneBaseOutput], nn.Module):
    """
    The base class of Backbone Model heritates from torch.nn.Module
    Wrap the pretrained model and define the output
    """
    def __init__(
        self,
        **kwargs,
    ):
        super(BackBoneBaseModule, self).__init__()
        
    @abstractmethod
    def forward(self, batch: TBatch) -> BackBoneBaseOutput:
        pass
    
    @classmethod
    @abstractmethod
    def load_backbone(
        cls,
        path: str,
        **kwargs,
    ) -> "BackBoneBaseModule[TBatch, BackBoneBaseOutput]": ...
    

class BackBoneBaseConfig(ABC, Generic[TBatch, BackBoneBaseOutput]):
    """
    The base class of Backbone Model Configuration
    """
    backbone_cls: type[BackBoneBaseModule[TBatch, BackBoneBaseOutput]]
    """The class of the backbone model"""
    freeze: bool = False
    """Whether to freeze the backbone model"""
    @abstractmethod
    def construct_backbone(
        self,
        **kwargs,
    ) -> BackBoneBaseModule[TBatch, BackBoneBaseOutput]: ...


class OutputHeadBaseConfig(ABC, Generic[TBatch]):
    """
    Base class for the configuration of the output head
    """
    
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
    
    @abstractmethod
    def get_num_classes(self) -> int:
        return 0
    
    @contextlib.contextmanager
    def model_forward_context(self, data: TBatch):
        """
        Model forward context manager.
        Make preparations before the forward pass for the output head.
        For example, set auto_grad to True for pos if using Gradient Force Head.
        """
        yield

    def supports_inference_mode(self) -> bool:
        return True