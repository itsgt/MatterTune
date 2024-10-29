from __future__ import annotations

from typing import Generic, Literal

import torch
import torch.nn as nn
from einops import rearrange
from typing_extensions import override

from mattertune.finetune.loss import LossConfig, MAELossConfig
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput
from mattertune.output_heads.goc_style.heads.utils.scatter_polyfill import scatter
from mattertune.output_heads.layers.activation import get_activation_cls
from mattertune.output_heads.layers.mlp import MLP
from mattertune.protocol import TBatch


class DirectScalerOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectScalerOutputHead
    """

    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "scalar"
    """The prediction type of the output head"""
    target_name: str = "direct_scaler"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    hidden_dim: int
    """The input hidden dim of output head"""
    reduction: Literal["mean", "sum", "none"] = "sum"
    """The reduction method. For example, the total_energy is the sum of the energy of each atom"""
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""
    activation: str
    """Activation function to use for the output layer"""

    @override
    def is_classification(self) -> bool:
        return False

    def construct_output_head(
        self,
    ) -> nn.Module:
        """
        Construct the output head and return it.
        """
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return DirectScalerOutputHead(
            self,
            hidden_dim=self.hidden_dim,
            activation_cls=get_activation_cls(self.activation),
        )


class DirectScalerOutputHead(nn.Module, Generic[TBatch]):
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
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        node_features = backbone_output[
            "node_hidden_features"
        ]  ## [num_nodes_in_batch, hidden_dim]
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        predicted_scaler = self.out_mlp(node_features)  ## [num_nodes_in_batch, 1]
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
            )  ## [batch_size, 1]
            scaler = rearrange(scaler, "b 1 -> b")
            output_head_results[self.head_config.target_name] = scaler
            return scaler


class DirectEnergyOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectScalerOutputHead
    """

    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "scalar"
    """The prediction type of the output head"""
    target_name: str = "direct_scaler"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    hidden_dim: int
    """The input hidden dim of output head"""
    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""
    reduction: Literal["mean", "sum", "none"] = "sum"
    """
    The reduction method
    For example, the total_energy is the sum of the energy of each atom
    """
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    activation: str
    """Activation function to use for the output layer"""

    @override
    def is_classification(self) -> bool:
        return False

    def construct_output_head(
        self,
    ) -> nn.Module:
        """
        Construct the output head and return it.
        """
        assert (
            self.reduction != "none"
        ), "The reduction can't be none for DirectEnergyOutputHead, choose 'mean' or 'sum'"
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return DirectEnergyOutputHead(
            self,
            hidden_dim=self.hidden_dim,
            activation_cls=get_activation_cls(self.activation),
        )


class DirectEnergyOutputHead(nn.Module, Generic[TBatch]):
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
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        energy_features = backbone_output[
            "energy_features"
        ]  ## [num_nodes_in_batch, hidden_dim]
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu() + 1)
        predicted_scaler = self.out_mlp(energy_features)  ## [num_nodes_in_batch, 1]
        scaler = scatter(
            predicted_scaler,
            batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.head_config.reduction,
        )  ## [batch_size, 1]
        assert scaler.shape == (
            num_graphs,
            1,
        ), f"energy_scaler.shape={scaler.shape} != [(batch_size, 1)]"
        scaler = rearrange(scaler, "b 1 -> b")
        output_head_results[self.head_config.target_name] = scaler
        return scaler
