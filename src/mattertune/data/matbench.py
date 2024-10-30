from __future__ import annotations

from typing import final

import ase
from torch.utils.data import Dataset
from typing_extensions import override

from .base import DatasetProtocol


class MatbenchDatasetConfig:
    pass


@final
class MatbenchDataset(DatasetProtocol, Dataset[ase.Atoms]):
    def __init__(self) -> None:
        super().__init__()

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        raise NotImplementedError

    @override
    def __len__(self) -> int:
        raise NotImplementedError
