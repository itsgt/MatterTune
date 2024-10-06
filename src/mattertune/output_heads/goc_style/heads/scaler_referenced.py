from typing import Annotated, Literal, TypeAlias, Generic
from typing_extensions import override
from abc import ABC, abstractmethod
from pydantic import Field
import pandas as pd
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
from pydantic import BaseModel


class ReferenceInitializationConfigBase(ABC, BaseModel):
    @abstractmethod
    def initialize(
        self, max_atomic_number: int
    ) -> torch.Tensor: ...


class ZerosReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["zeros"] = "zeros"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> torch.Tensor:
        return torch.zeros((max_atomic_number + 1, 1))


class RandomReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["random"] = "random"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> torch.Tensor:
        return torch.randn((max_atomic_number + 1, 1))


## TODO: Probably not work
class MPElementalReferenceInitializationConfig(ReferenceInitializationConfigBase):
    name: Literal["mp_elemental"] = "mp_elemental"

    @override
    def initialize(
        self, max_atomic_number: int
    ) -> torch.Tensor:
        from matbench_discovery.data import DATA_FILES
        from pymatgen.core import Element
        from pymatgen.entries.computed_entries import ComputedEntry

        raise NotImplementedError("This method has a bug, it will be fixed in the future.")
        references = torch.zeros((max_atomic_number + 1, 1))

        for elem_str, entry in (
            pd.read_json(DATA_FILES.mp_elemental_ref_entries, typ="series")
            .map(ComputedEntry.from_dict)
            .to_dict()
            .items()
        ):
            references[Element(elem_str).Z] = round(entry.energy_per_atom, 4)

        return references


ReferenceInitializationConfig: TypeAlias = Annotated[
    ZerosReferenceInitializationConfig
    | RandomReferenceInitializationConfig
    | MPElementalReferenceInitializationConfig,
    Field(discriminator="name"),
]

class ReferencedScalerOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the ReferencedScalerOutputHead
    For example:
    e_i = f(x_i) + r_{Z_i}, where f is the output layer and r is the reference.
    """
    ## Paramerters heritated from OutputHeadBaseConfig:
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "scalar"
    """The prediction type of the output head"""
    target_name: str = "referenced_scaler"
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
    max_atomic_number: int
    """The max atomic number in the dataset."""
    initialization: ReferenceInitializationConfig
    """The initialization configuration for the references."""
    trainable_references: bool = True
    """Whether to train the references. If False, the references must be initialized."""
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
        return ReferencedScalerOutputHead(
            self,
            hidden_dim=self.hidden_dim,
            activation_cls=get_activation_cls(self.activation),
        )
        
class ReferencedScalerOutputHead(nn.Module, Generic[TBatch]):
    """
    The output head of the direct graph scaler.
    """
    def __init__(
        self,
        head_config: ReferencedScalerOutputHeadConfig,
        hidden_dim: int,
        activation_cls: type[nn.Module],
    ):
        super(ReferencedScalerOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        self.activation_cls = activation_cls
        self.out_mlp = MLP(
            ([self.hidden_dim] * self.head_config.num_layers) + [1],
            activation_cls=activation_cls,
        )
        self.references = nn.Embedding.from_pretrained(
            head_config.initialization.initialize(head_config.max_atomic_number),
            freeze=not head_config.trainable_references,
        )
        assert (
            self.references.weight.shape == (head_config.max_atomic_number + 1, 1)
        ), f"{self.references.weight.shape=} != {(head_config.max_atomic_number + 1, 1)}"

    def forward(
        self, 
        *,
        batch_data: TBatch,
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        node_features = backbone_output["node_hidden_features"] ## [num_nodes_in_batch, hidden_dim]
        atomic_numbers = batch_data.atomic_numbers
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        predicted_scaler = self.out_mlp(node_features) ## [num_nodes_in_batch, 1]
        predicted_reference = self.references(atomic_numbers) ## [num_nodes_in_batch, 1]
        predicted_scaler = predicted_scaler + predicted_reference
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
        

class ReferencedEnergyOutputHeadConfig(OutputHeadBaseConfig):
    """
    Configuration of the ReferencedScalerOutputHead
    For example:
    e_i = f(x_i) + r_{Z_i}, where f is the output layer and r is the reference.
    """
    ## Paramerters heritated from OutputHeadBaseConfig: 
    pred_type: Literal["scalar", "vector", "tensor", "classification"] = "scalar"
    """The prediction type of the output head"""
    target_name: str = "referenced_scaler"
    """The name of the target output by this head"""
    loss_coefficient: float = 1.0
    """The coefficient of the loss function"""
    ## New parameters:
    reduction: Literal["mean", "sum", "none"] = "sum"
    """
    The reduction method
    For example, the total_energy is the sum of the energy of each atom
    """
    num_layers: int = 5
    """Number of layers in the output layer."""
    output_init: Literal["HeOrthogonal", "zeros", "grid", "loggrid"] = "HeOrthogonal"
    """Initialization method for the output layer."""
    hidden_dim: int
    """The input hidden dim of output head"""
    loss: LossConfig = MAELossConfig()
    """The loss configuration for the target."""
    max_atomic_number: int
    """The max atomic number in the dataset."""
    initialization: ReferenceInitializationConfig
    """The initialization configuration for the references."""
    trainable_references: bool = True
    """Whether to train the references. If False, the references must be initialized."""
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
        assert self.reduction != "none", "The reduction can't be none for ReferencedEnergyOutputHead, choose 'mean' or 'sum'"
        if self.hidden_dim is None:
            raise ValueError("hidden_dim must be provided for DirectScalerOutputHead")
        return ReferencedEnergyOutputHead(
            self,
            hidden_dim=self.hidden_dim,
            activation_cls=get_activation_cls(self.activation),
        )
        
class ReferencedEnergyOutputHead(nn.Module, Generic[TBatch]):
    """
    The output head of the direct graph scaler.
    """
    def __init__(
        self,
        head_config: ReferencedEnergyOutputHeadConfig,
        hidden_dim: int,
        activation_cls: type[nn.Module],
    ):
        super(ReferencedEnergyOutputHead, self).__init__()
        
        self.head_config = head_config
        self.hidden_dim = hidden_dim
        self.activation_cls = activation_cls
        self.out_mlp = MLP(
            ([self.hidden_dim] * self.head_config.num_layers) + [1],
            activation_cls=activation_cls,
        )
        self.references = nn.Embedding.from_pretrained(
            head_config.initialization.initialize(head_config.max_atomic_number),
            freeze=not head_config.trainable_references,
        )
        assert (
            self.references.weight.shape == (head_config.max_atomic_number + 1, 1)
        ), f"{self.references.weight.shape=} != {(head_config.max_atomic_number + 1, 1)}"

    def forward(
        self, 
        *,
        batch_data: TBatch,
        backbone_output: GOCStyleBackBoneOutput,
        output_head_results: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        energy_features = backbone_output["energy_features"] ## [num_nodes_in_batch, hidden_dim]
        atomic_numbers = batch_data.atomic_numbers
        batch_idx = batch_data.batch
        num_graphs = int(torch.max(batch_idx).detach().cpu().item() + 1)
        predicted_scaler = self.out_mlp(energy_features) ## [num_nodes_in_batch, 1]
        predicted_reference = self.references(atomic_numbers) ## [num_nodes_in_batch, 1]
        predicted_scaler = predicted_scaler + predicted_reference
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