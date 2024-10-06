from typing import Literal, Generic
from typing_extensions import override
import contextlib
import torch
import torch.nn as nn
from einops import rearrange
from mattertune.protocol import TBatch
from mattertune.output_heads.base import OutputHeadBaseConfig
from mattertune.finetune.loss import LossConfig, MAELossConfig
from mattertune.output_heads.layers.mlp import MLP
from mattertune.output_heads.layers.activation import get_activation_cls
from mattertune.output_heads.goc_style.heads.utils.scatter_polyfill import scatter
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput


class GlobalScalerOutputHeadConfig(OutputHeadBaseConfig):
    r"""
    Configuration of the GlobalScalerOutputHead
    Compute a global scalar output from the backbone output
    """

    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "scalar"
    """The prediction type of the output head"""
    target_name: str = "global_scaler"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    hidden_dim: int
    """The input hidden dim of output head"""
    reduction: Literal["mean", "sum"] = "sum"
    """The reduction method for the output"""
    loss: LossConfig = MAELossConfig()
    """The loss function to use for the target"""
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
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
            raise ValueError("hidden_dim must be provided for GlobalScalerOutputHead")
        return GlobalScalerOutputHead(
            self,
            self.hidden_dim,
            get_activation_cls(self.activation),
        )
        
        
class GlobalBinaryClassificationOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the GlobalBinaryClassificationOutputHead
    Compute a global binary classification output from the backbone output
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "classification"
    """The prediction type of the output head"""
    target_name: str = "global_binary_classification"
    """The name of the target output by this head"""
    hidden_dim: int|None = None
    """The hidden dimension of the output head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    reduction: Literal["mean", "sum"] = "sum"
    """The reduction method for the output"""
    pos_weight: float | None = None
    """The positive weight for the target"""
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    activation: str
    """Activation function to use for the output layer"""
    
    @override
    def is_classification(self) -> bool:
        return True
    
    @override
    def construct_output_head(
        self,
    ):
        if self.pos_weight is not None:
            assert self.pos_weight > 0, f"pos_weight of GlobalBinaryClassificationOutputHead must be positive, found {self.pos_weight}"
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for GlobalBinaryClassificationOutputHead")
        return GlobalBinaryClassificationOutputHead(
            self,
            self.hidden_dim,
            get_activation_cls(self.activation),
        )
        
    @override
    def get_num_classes(self) -> int:
        return 2
        

class GlobalMultiClassificationOutputHeadConfig(OutputHeadBaseConfig, Generic[TBatch]):
    """
    Configuration of the GlobalMultiClassificationOutputHead
    Compute a global multi classification output from the backbone output
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "classification"
    """The prediction type of the output head"""
    target_name: str = "global_multi_classification"
    """The name of the target output by this head"""
    hidden_dim: int|None = None
    """The hidden dimension of the output head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    reduction: Literal["mean", "sum"] = "sum"
    """The reduction method for the output"""
    num_classes: int
    """Number of classes in the classification"""
    class_weights: list[float] | None = None
    """The class weights for the target"""
    dropout: float | None = None
    """The dropout probability to use before the output layer"""
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    activation: str

    
    @override
    def is_classification(self) -> bool:
        return True
    
    @override
    def construct_output_head(
        self,
    ):
        if self.class_weights is not None:
            assert len(self.class_weights) == self.num_classes, f"Number of class weights must match the number of classes, found {len(self.class_weights)} != {self.num_classes}"
            for weight in self.class_weights:
                assert weight > 0, f"Class weights must be positive, found {weight}"
        if self.dropout is not None:
            assert 0.0 <= self.dropout < 1.0, f"Dropout probability must be between 0 and 1, found {self.dropout}"
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for GlobalMultiClassificationOutputHead")
        return GlobalMultiClassificationOutputHead(
            self,
            self.hidden_dim,
            get_activation_cls(self.activation),
        )
    
    @override
    def get_num_classes(self) -> int:
        return self.num_classes
        

class GlobalScalerOutputHead(nn.Module, Generic[TBatch]):
    r"""
    Compute a global scalar output from the backbone output
    """

    @override
    def __init__(
        self,
        head_config: GlobalScalerOutputHeadConfig,
        hidden_dim: int,
        activation_cls,
    ):
        super(GlobalScalerOutputHead, self).__init__()
        
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
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ):
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        node_features = backbone_output["node_hidden_features"]
        predicted_scaler = self.out_mlp(node_features)
        scaler = scatter(
            predicted_scaler,
            batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.head_config.reduction,
        )
        assert scaler.shape == (num_graphs, 1), f"scaler.shape={scaler.shape} != [num_graphs, 1]"
        scaler = rearrange(scaler, "b 1 -> b")
        output_head_results[self.head_config.target_name] = scaler
        return scaler
    

class GlobalBinaryClassificationOutputHead(nn.Module, Generic[TBatch]):
    """
    Compute a global binary classification output from the backbone output
    """

    @override
    def __init__(
        self,
        head_config: GlobalBinaryClassificationOutputHeadConfig,
        hidden_dim: int,
        activation_cls,
    ):
        super(GlobalBinaryClassificationOutputHead, self).__init__()
        
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
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ):
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        node_features = backbone_output["node_hidden_features"]
        predicted_logits = self.out_mlp(node_features)
        logits = scatter(
            predicted_logits,
            batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.head_config.reduction,
        )
        assert logits.shape == (num_graphs, 1), f"logits.shape={logits.shape} != [num_graphs, 1]"
        logits = rearrange(logits, "b 1 -> b")
        output_head_results[self.head_config.target_name] = logits
        return logits
    

class GlobalMultiClassificationOutputHead(nn.Module, Generic[TBatch]):
    """
    Compute a global multi classification output from the backbone output
    """

    @override
    def __init__(
        self,
        head_config: GlobalMultiClassificationOutputHeadConfig,
        hidden_dim: int,
        activation_cls,
    ):
        super(GlobalMultiClassificationOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        
        self.out_mlp = MLP(
            ([self.hidden_dim] * self.head_config.num_layers) + [self.head_config.num_classes],
            activation_cls=activation_cls,
        )
        
        self.register_buffer(
            f"{self.head_config.target_name}_class_weights",
            torch.tensor(self.head_config.class_weights, dtype=torch.float32),
            persistent=False,
        )
        
    @override
    def forward(
        self,
        *,
        batch_data: TBatch,
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ):
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        node_features = backbone_output["node_hidden_features"]
        predicted_logits = self.out_mlp(node_features)
        logits = scatter(
            predicted_logits,
            batch_idx,
            dim=0,
            dim_size=num_graphs,
            reduce=self.head_config.reduction,
        )
        assert logits.shape == (num_graphs, self.head_config.num_classes), f"logits.shape={logits.shape} != [num_graphs, {self.head_config.num_classes}]"
        output_head_results[self.head_config.target_name] = logits
        return logits