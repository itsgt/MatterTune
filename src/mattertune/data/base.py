from __future__ import annotations

from typing import Protocol, runtime_checkable

import ase


@runtime_checkable
class DatasetProtocol(Protocol):
    def __getitem__(self, idx: int, /) -> ase.Atoms: ...

    def __len__(self) -> int: ...
