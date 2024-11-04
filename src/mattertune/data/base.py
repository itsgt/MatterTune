from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

import nshconfig as C
from torch.utils.data import Dataset
from typing_extensions import override

from ase import Atoms


class DatasetConfigBase(C.Config, ABC):
    @classmethod
    @abstractmethod
    def dataset_cls(cls) -> type[DatasetBase]: ...


TConfig = TypeVar("TConfig", bound=DatasetConfigBase, covariant=True)


class DatasetBase(Dataset[Atoms], ABC, Generic[TConfig]):
    @override
    def __init__(self, config: TConfig):
        super().__init__()

        self.config = config

    @classmethod
    def ensure_dependencies(cls):
        """
        Ensure that all dependencies are installed.

        This method should raise an exception if any dependencies are missing,
            with a message indicating which dependencies are missing and
            how to install them.
        """
        pass
