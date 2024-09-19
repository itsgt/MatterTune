from abc import ABC, abstractmethod
from typing import (
    Any,
    Generic,
    MutableMapping,
    Protocol,
    TypedDict,
    runtime_checkable,
)

import nshtrainer as nt
import nshutils.typecheck as tc
import torch
from typing_extensions import TypeVar, override
from .data import RawData


@runtime_checkable
class Data(Protocol):
    pass


TData = TypeVar("TData", bound=Data, infer_variance=True)


@runtime_checkable
class Batch(Protocol):
    pass


TBatch = TypeVar("TBatch", bound=Batch, infer_variance=True)


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