from __future__ import annotations

from pathlib import Path
from typing import Literal

import ase
from ase import Atoms
from ase.io import read
import nshconfig as C
from torch.utils.data import Dataset
from typing_extensions import override

from .base import DatasetProtocol


class XYZDatasetConfig(C.Config):
    type: Literal["xyz"] = "xyz"
    """Discriminator for the XYZ dataset."""

    src: Path
    """The path to the XYZ dataset."""

    def create_dataset(self):
        return XYZDataset(self)
    

class XYZDataset(DatasetProtocol, Dataset[ase.Atoms]):
    def __init__(self, config: XYZDatasetConfig):
        super().__init__()

        self.config = config
        self.atoms_list:list[Atoms] = read(str(config.src), index=":")

    @override
    def __getitem__(self, idx: int) -> ase.Atoms:
        return self.atoms_list[idx]

    @override
    def __len__(self) -> int:
        return len(self.atoms_list)