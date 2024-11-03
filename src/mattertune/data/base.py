from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import nshconfig as C
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ase import Atoms


class DatasetBase(Dataset[Atoms], ABC):
    pass


class DatasetConfigBase(C.Config, ABC):
    @abstractmethod
    def create_dataset(self) -> DatasetBase: ...
