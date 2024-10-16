from abc import abstractmethod, ABC
from typing import (
    Any,
    Generic,
    Protocol,
    runtime_checkable,
)
import torch
import torch.nn as nn
from typing_extensions import TypeVar, override
import jaxtyping as jt
from pydantic import BaseModel


@runtime_checkable
class DataProtocol(Protocol):
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes"]
    pos: jt.Float[torch.Tensor, "num_nodes 3"]
    num_atoms: jt.Int[torch.Tensor, "1"]
    cell_displacement: jt.Float[torch.Tensor, "1 3 3"]|None
    cell: jt.Float[torch.Tensor, "1 3 3"]|None
    
    @abstractmethod
    def __getattr__(self, name: str) -> Any: ...
    
    @abstractmethod
    def __setattr__(self, key: str, value) -> None: ...
    
    @abstractmethod
    def __delattr__(self, key: str) -> None: ...
    
    @abstractmethod
    def __hasattr__(self, key: str) -> bool: ...
    

TData = TypeVar("TData", bound=DataProtocol, infer_variance=True)


@runtime_checkable
class BatchProtocol(Protocol):
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes_in_batch"]
    pos: jt.Float[torch.Tensor, "num_nodes_in_batch 3"]
    num_atoms: jt.Int[torch.Tensor, "num_graphs_in_batch"]
    batch: jt.Int[torch.Tensor, "num_nodes_in_batch"]
    cell_displacement: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]|None
    cell: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]|None
    
    @abstractmethod
    def __len__(self) -> int: ...
    
    @abstractmethod
    def __getattr__(self, name: str) -> Any: ...
    
    @abstractmethod
    def __setattr__(self, key: str, value) -> None: ...
    
    @abstractmethod
    def __delattr__(self, key: str) -> None: ...
    
    @abstractmethod
    def __hasattr__(self, key: str) -> bool: ...


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
        *args,
        **kwargs,
    ) -> "BackBoneBaseModule[TBatch, BackBoneBaseOutput]": ...
    
    @abstractmethod
    def process_batch_under_grad(
        self,
        batch: TBatch,
        training: bool
    ) -> TBatch: ...
    

class BackBoneBaseConfig(BaseModel, ABC, Generic[TBatch, BackBoneBaseOutput]):
    model_config = {
        'arbitrary_types_allowed': True
    }
    freeze: bool = False
    """Whether to freeze the backbone model"""
    @abstractmethod
    def construct_backbone(
        self,
        *args,
        **kwargs,
    ) -> BackBoneBaseModule[TBatch, BackBoneBaseOutput]: ...