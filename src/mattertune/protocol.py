from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeAlias, TypeVar, runtime_checkable

import nshutils.typecheck as tc
import torch
from typing_extensions import override

from .data import RawData


@runtime_checkable
class Data(Protocol):
    pass


TData = TypeVar("TData", bound=Data)


@runtime_checkable
class Batch(Protocol):
    pass


TBatch = TypeVar("TBatch", bound=Batch)


ModelPredictions: TypeAlias = dict[str, torch.Tensor]


class ModelBase(ABC, Generic[TData, TBatch]):
    @override
    def __init__(self):
        super().__init__()

    @abstractmethod
    def data_transform(self, data: RawData) -> TData: ...

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch: ...

    @abstractmethod
    def forward(self, batch: TBatch) -> ModelPredictions: ...

    @abstractmethod
    def loss(
        self, predictions: ModelPredictions, batch: TBatch
    ) -> tc.Float[torch.Tensor, ""]: ...

    @abstractmethod
    def metrics(
        self, predictions: ModelPredictions, batch: TBatch
    ) -> dict[str, tc.Float[torch.Tensor, ""]]: ...
