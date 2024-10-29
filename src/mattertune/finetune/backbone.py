from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn
from typing import Generic
from mattertune.data_structures import TMatterTuneData, TMatterTuneBatch
from pydantic import BaseModel
from ase import Atoms


class BackBoneBaseModule(ABC, nn.Module, Generic[TMatterTuneData, TMatterTuneBatch]):
    """
    The base class of Backbone Model heritates from torch.nn.Module
    Wrap the pretrained model
    """
    def __init__(
        self,
        **kwargs,
    ):
        super(BackBoneBaseModule, self).__init__()
        
    @abstractmethod
    def forward(self, batch: TMatterTuneBatch) -> Any: ...
    
    @classmethod
    @abstractmethod
    def load_backbone(
        cls,
        path: str,
        *args,
        **kwargs,
    ) -> "BackBoneBaseModule": ...
    
    @abstractmethod
    def process_raw(
        self,
        atoms: Atoms,
        idx: int,
        labels: dict[str, torch.Tensor],
        inference: bool,
    ) -> TMatterTuneBatch: ...
    
    @abstractmethod
    def process_batch_under_grad(
        self,
        batch: TMatterTuneBatch,
        training: bool
    ) -> TMatterTuneBatch: ...
    
    @classmethod
    @abstractmethod
    def collate_fn(
        cls,
        data_list: list[TMatterTuneData],
    ) -> TMatterTuneBatch: ...
    

class BackBoneBaseConfig(BaseModel, ABC):
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
    ) -> BackBoneBaseModule: ...