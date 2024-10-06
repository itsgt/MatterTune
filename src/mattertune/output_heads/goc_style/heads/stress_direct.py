from typing import Literal, Generic
from typing_extensions import override, assert_never
import torch
import torch.nn as nn
from mattertune.protocol import TBatch
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.finetune.loss import LossConfig, MAELossConfig
from mattertune.output_heads.goc_style.heads.utils.rank2block import Rank2DecompositionEdgeBlock
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput


class DirectStressOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the DirectStressOutputHead
    Compute stress directly without using gradients
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "tensor"
    """The prediction type of the output head"""
    target_name: str = "direct_stress"
    """The name of the target output by this head"""
    ## New parameters:
    hidden_dim: int
    """The input hidden dim of output head"""
    reduction: Literal["mean", "sum"] = "sum"
    """The reduction method"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    num_layers: int = 2
    """Number of layers in the output layer."""
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
            case _:
                assert_never(self.reduction)
    
    @override
    def construct_output_head(
        self,
    ):
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectStressOutputHead")
        return DirectStressOutputHead(
            self,
            hidden_dim=self.hidden_dim,
        )

class DirectStressOutputHead(nn.Module, Generic[TBatch]):
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
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        force_features = backbone_output["force_features"]
        edge_vectors = backbone_output["edge_vectors"]
        edge_index_dst = backbone_output["edge_index_dst"]
        batch_idx = batch_data.batch
        batch_size = int(torch.max(batch_idx).detach().cpu().item() + 1)
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