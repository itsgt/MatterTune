from typing import Literal, Generic
from typing_extensions import override, assert_never
import torch
import torch.nn as nn
from .base import OutputHeadBaseConfig
from ..modules.loss import LossConfig, MAELossConfig
from .layers.rank2block import Rank2DecompositionEdgeBlock
from ..protocol import TData, TBatch
from ..finetune_model import BackBoneBaseOutput


class DirectStressOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectStressOutputHead
    Compute stress directly without using gradients
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    head_name: str = "DirectStressOutputHead"
    """The name of the output head"""
    target_name: str = "direct_stress"
    """The name of the target output by this head"""
    reduction: Literal["mean", "sum", "none"] = "sum"
    """The reduction method"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    num_layers: int = 2
    """Number of layers in the output layer."""
    ## New parameters:
    loss: LossConfig = MAELossConfig()
    """The loss function to use for the target"""
    
    @override
    def is_classification(self) -> bool:
        return False
    
    @property
    def extensive(self):
        match self.reduction:
            case "sum":
                return True
            case "mean":
                return False
            case "none":
                raise ValueError("for stress, reduction cannot be none")
            case _:
                assert_never(self.reduction)
    
    @override
    def construct_output_head(
        self,
        hidden_dim: int|None,
        activation_cls: type[nn.Module],
    ):
        if hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectStressOutputHead")
        return DirectStressOutputHead(
            self,
            hidden_dim=hidden_dim,
        )

class DirectStressOutputHead(nn.Module, Generic[TBatch, BackBoneBaseOutput]):
    @override
    def __init__(
        self,
        target_config: DirectStressOutputHeadConfig,
        hidden_dim: int,
    ):
        super().__init__()

        self.target_config = target_config
        del target_config

        self.block = Rank2DecompositionEdgeBlock(
            hidden_dim,
            edge_level=True,
            extensive=self.target_config.extensive,
            num_layers=self.target_config.num_layers,
        )

    @override
    def forward(
        self, 
        *,
        batch_data: TBatch,
        backbone_output: BackBoneBaseOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        force_features = backbone_output.force_features
        edge_vectors = backbone_output.edge_vectors
        edge_index_dst = backbone_output.edge_index_dst
        batch_idx = batch_data.batch
        batch_size = int(torch.max(batch_idx).item() + 1)
        stress = self.block(
            force_features,
            edge_vectors,
            edge_index_dst,
            batch_idx,
            batch_size,
        )
        assert stress.shape == (batch_size, 3, 3), f"stress.shape={stress.shape} != [batch_size, 3, 3]"
        output_head_results[self.target_config.target_name] = stress
        return stress