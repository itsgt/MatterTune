from __future__ import annotations

from typing import Annotated, TypeAlias

from ..registry import data_registry
from .base import DatasetConfigBase
from .omat24 import OMAT24Dataset as OMAT24Dataset
from .omat24 import OMAT24DatasetConfig as OMAT24DatasetConfig

DatasetConfig: TypeAlias = Annotated[
    DatasetConfigBase,
    data_registry.RegistryResolution(),
]
