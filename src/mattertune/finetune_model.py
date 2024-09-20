from abc import abstractmethod
from typing import Generic, TypeVar, Protocol, runtime_checkable
import torch
import torch.nn as nn
import pytorch_lightning as pl
from .protocol import TData, TBatch
from .output_head.base import OutputHeadBaseConfig
from torchtyping import TensorType

@runtime_checkable
class BackBoneBaseOutputProtocol(Protocol):
    edge_index_src: TensorType["num_edges_in_batch"]
    edge_index_dst: TensorType["num_edges_in_batch"]
    edge_vectors: TensorType["num_edges_in_batch", 3]
    edge_lengths: TensorType["num_edges_in_batch"]
    node_hidden_features: TensorType["num_nodes_in_batch", "node_hidden_dim"]
    edge_hidden_features: TensorType["num_edges_in_batch", "edge_hidden_dim"]
    energy_features: TensorType["num_nodes_in_batch", "energy_feature_dim"]
    force_features: TensorType["num_nodes_in_batch", "force_feature_dim"]

BackBoneBaseOutput = TypeVar("BackBoneBaseOutput", bound=BackBoneBaseOutputProtocol, infer_variance=True)

class BackBoneBaseModel(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
    """
    The base class of Backbone Model heritates from torch.nn.Module
    Wrap the pretrained model and define the output
    """
    def __init__(
        self,
        **kwargs,
    ):
        super(BackBoneBaseModel, self).__init__()
        
    @abstractmethod
    def forward(self, batch: TBatch) -> BackBoneBaseOutput:
        pass


class FineTunebaseModel(pl.LightningModule):
    """
    The base class of Finetune Model heritates from pytorch_lightning.LightningModule
    Two main components:
    - backbone: BackBoneBaseModel loaded from the pretrained model
    - output_head: defined by the user in finetune task
    """
    def __init__(
        self, 
        backbone: BackBoneBaseModel,
        output_heads_config: list[OutputHeadBaseConfig],
        **kwargs,
    ):
        super(FineTunebaseModel, self).__init__()
        self.backbone = backbone
        self.output_heads_config = output_heads_config
    
    