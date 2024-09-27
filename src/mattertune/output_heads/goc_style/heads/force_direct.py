from typing import Literal, Generic
from typing_extensions import override
import torch
import torch.nn as nn
from mattertune.protocol import TBatch, OutputHeadBaseConfig
from mattertune.finetune.loss import LossConfig, L2MAELossConfig
from mattertune.output_heads.layers.mlp import MLP
from mattertune.output_heads.layers.activation import get_activation_cls
from mattertune.output_heads.goc_style.heads.utils.scatter_polyfill import scatter
from mattertune.output_heads.goc_style.backbone_module import GOCBackBoneOutput


class DirectForceOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectForceTarget
    Compute force directly from the output layer
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "vector"
    """The prediction type of the output head"""
    target_name: str = "direct_forces"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    hidden_dim: int
    """The input hidden dim of output head"""
    reduction: Literal["mean", "sum"] = "sum"
    """The reduction method for the output"""
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    loss: LossConfig = L2MAELossConfig()
    """The loss function to use for the target"""
    activation: str
    """Activation function to use for the output layer"""
    
    @override
    def is_classification(self) -> bool:
        return False

    @override
    def construct_output_head(
        self,
    ):
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return DirectForceOutputHead(
            self,
            self.hidden_dim,
            get_activation_cls(self.activation),
        )
        
class DirectForceOutputHead(nn.Module, Generic[TBatch]):
    """
    Compute force components directly from the backbone output's edge features
    Without using gradients
    """
    @override
    def __init__(
        self,
        head_config: DirectForceOutputHeadConfig,
        hidden_dim: int,
        activation_cls,
    ):
        super(DirectForceOutputHead, self).__init__()
        
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
        backbone_output: GOCBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ):
        force_feature:torch.Tensor = backbone_output["force_features"]
        edge_vectors:torch.Tensor = backbone_output["edge_vectors"]
        edge_index_dst:torch.Tensor = backbone_output["edge_index_dst"]
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