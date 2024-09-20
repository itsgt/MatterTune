from typing import Literal, Generic
from typing_extensions import override
import torch
import torch.nn as nn
from einops import rearrange
from .base import OutputHeadBaseConfig
from ..modules.loss import LossConfig, MAELossConfig
from .layers.mlp import MLP
from .utils.scatter_polyfill import scatter
from ..protocol import TData, TBatch
from ..finetune_model import BackBoneBaseOutput


class DirectScalerOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectScalerOutputHead
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    head_name: str = "DirectScalerOutputHead"
    """The name of the output head"""
    target_name: str = "direct_scaler"
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
    ## New parameters:
    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""

    @override
    def is_classification(self) -> bool:
        return False

    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls: type[nn.Module],
    ) -> nn.Module:
        """
        Construct the output head and return it.
        """
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return DirectScalerOutputHead(
            self,
            hidden_dim=hidden_dim,
            activation_cls=activation_cls,
        )
        
class DirectScalerOutputHead(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
    """
    The output head of the direct graph scaler.
    """
    @override
    def __init__(
        self,
        head_config: DirectScalerOutputHeadConfig,
        hidden_dim: int,
        activation_cls: type[nn.Module],
    ):
        super(DirectScalerOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        self.num_mlps = self.head_config.num_layers
        self.activation_cls = activation_cls
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
    ) -> torch.Tensor:
        node_features = backbone_output.node_hidden_features ## [num_nodes_in_batch, hidden_dim]
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).item() + 1)
        predicted_scaler = self.out_mlp(node_features) ## [num_nodes_in_batch, 1]
        if self.head_config.reduction == "none":
            predicted_scaler = rearrange(predicted_scaler, "n 1 -> n")
            output_head_results[self.head_config.target_name] = predicted_scaler
            return predicted_scaler
        else:
            scaler = scatter(
                predicted_scaler,
                batch_idx,
                dim=0,
                dim_size=num_graphs,
                reduce=self.head_config.reduction,
            ) ## [batch_size, 1]
            scaler = rearrange(scaler, "b 1 -> b")
            output_head_results[self.head_config.target_name] = scaler
            return scaler
        

class DirectEnergyOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectScalerOutputHead
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    head_name: str = "DirectScalerOutputHead"
    """The name of the output head"""
    target_name: str = "direct_scaler"
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
    ## New parameters:
    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""

    @override
    def is_classification(self) -> bool:
        return False

    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls: type[nn.Module],
    ) -> nn.Module:
        """
        Construct the output head and return it.
        """
        assert self.reduction != "none", "The reduction can't be none for DirectEnergyOutputHead, choose 'mean' or 'sum'"
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return DirectEnergyOutputHead(
            self,
            hidden_dim=hidden_dim,
            activation_cls=activation_cls,
        )
        
class DirectEnergyOutputHead(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
    """
    The output head of the direct graph scaler.
    """
    @override
    def __init__(
        self,
        head_config: DirectEnergyOutputHeadConfig,
        hidden_dim: int,
        activation_cls: type[nn.Module],
    ):
        super(DirectEnergyOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        self.num_mlps = self.head_config.num_layers
        self.activation_cls = activation_cls
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
    ) -> torch.Tensor:
        energy_features = backbone_output.energy_features ## [num_nodes_in_batch, hidden_dim]
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).item() + 1)
        predicted_scaler = self.out_mlp(energy_features) ## [num_nodes_in_batch, 1]
        scaler = scatter(
            predicted_scaler,
            batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.head_config.reduction,
        ) ## [batch_size, 1]
        assert scaler.shape == (num_graphs, 1), f"energy_scaler.shape={scaler.shape} != [(batch_size, 1)]"
        scaler = rearrange(scaler, "b 1 -> b")
        output_head_results[self.head_config.target_name] = scaler
        return scaler