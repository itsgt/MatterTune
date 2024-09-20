from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    MutableMapping,
    Protocol,
    TypedDict,
    runtime_checkable,
    Optional,
)

import nshtrainer as nt
import nshutils.typecheck as tc
import torch
from typing_extensions import TypeVar, override
from torchtyping import TensorType
from .data import RawData


@runtime_checkable
class DataProtocol(Protocol):
    atomic_numbers: TensorType["num_nodes"]
    pos: TensorType["num_nodes", 3]
    cell_displacement: TensorType[3, 3]|None
    cell: TensorType[3, 3]|None


TData = TypeVar("TData", bound=DataProtocol, infer_variance=True)


@runtime_checkable
class BatchProtocol(Protocol):
    atomic_numbers: torch.Tensor # [num_nodes_in_batch]
    pos: torch.Tensor # [num_nodes_in_batch, 3]
    batch: torch.Tensor # [num_nodes_in_batch]
    cell_displacement: torch.Tensor|None # [batch_size, 3, 3]
    cell: torch.Tensor|None # [batch_size, 3, 3]
    
    @abstractmethod
    def __len__(self) -> int: ...


TBatch = TypeVar("TBatch", bound=BatchProtocol, infer_variance=True)


# ModelPredictions: TypeAlias = dict[str, torch.Tensor]
class ModelPredictions(TypedDict):
    energy: torch.Tensor
    forces: torch.Tensor


class MatterTuneBaseModuleConfig(nt.BaseConfig):
    pass


TConfig = TypeVar("TConfig", bound=MatterTuneBaseModuleConfig, infer_variance=True)


class MatterTuneBaseModule(
    nt.LightningModuleBase[TConfig],
    Generic[TConfig, TData, TBatch],
):
    @override
    def __init__(self, hparams: TConfig | MutableMapping[str, Any]):
        super().__init__(hparams)

    @abstractmethod
    def forward(self, batch: TBatch) -> ModelPredictions: ...

    @abstractmethod
    def loss(
        self, predictions: ModelPredictions, batch: TBatch
    ) -> tc.Float[torch.Tensor, ""]: ...

    @abstractmethod
    def data_transform(self, data: RawData) -> TData: ...

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch: ...

    # @abstractmethod
    # def metrics(
    #     self, predictions: ModelPredictions, batch: TBatch
    # ) -> dict[str, tc.Float[torch.Tensor, ""]]: ...