from typing import Literal, Generic
from typing_extensions import override
import torch
import torch.nn as nn
from einops import rearrange
from .base import OutputHeadBaseConfig
from ..modules.loss import LossConfig, L2MAELossConfig
from .layers.mlp import MLP
from .utils.scatter_polyfill import scatter
from ..protocol import TData, TBatch
from ..finetune_model import BackBoneBaseOutput


class DirectForceOutputheadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectForceTarget
    Compute force directly from the output layer
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    head_name: str = "DirectForceOutputHead"
    """The name of the output head"""
    target_name: str = "direct_forces"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    reduction: Literal["mean", "sum", "none"] = "sum"
    """
    The reduction method
    """
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    ## New parameters:
    loss: LossConfig = L2MAELossConfig()
    """The loss function to use for the target"""
    
    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls,
    ):
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return NodeVectorOutputHead(
            self,
            hidden_dim,
            activation_cls,
        )
        
class NodeVectorOutputHead(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
    """
    Compute force components directly from the backbone output's edge features
    Without using gradients
    """
    @override
    def __init__(
        self,
        head_config: DirectForceOutputheadConfig,
        hidden_dim: int,
        activation_cls,
    ):
        super(NodeVectorOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        
        self.out_mlp = MLP(
            ([self.hidden_dim] * self.head_config.num_layers) + [1],
            activation_cls=activation_cls,
        )
    
    @override
    def forward(
        self,
        *,
        batch_data: TBatch,
        backbone_output: BackBoneBaseOutput,
        output_head_results: dict[str, torch.Tensor],
    ):
        force_feature:torch.Tensor = backbone_output.force_features
        edge_vectors:torch.Tensor = backbone_output.edge_vectors
        edge_index_dst:torch.Tensor = backbone_output.edge_index_dst
        natoms_in_batch = batch_data.pos.shape[0]
        forces_scale = self.out_mlp(force_feature)
        forces = forces_scale * edge_vectors
        forces = scatter(
            forces,
            edge_index_dst,
            dim=0,
            dim_size=natoms_in_batch,
            reduce=self.head_config.reduction,
        )
        assert forces.shape == (natoms_in_batch, 3), f"forces.shape={forces.shape} != [num_nodes_in_batch, 3]"
        output_head_results[self.head_config.target_name] = forces
        return forces