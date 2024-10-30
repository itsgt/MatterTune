from __future__ import annotations

from typing import Annotated, TypeAlias

import nshconfig as C

from .omat24 import OMAT24Dataset as OMAT24Dataset
from .omat24 import OMAT24DatasetConfig as OMAT24DatasetConfig

DatasetConfig: TypeAlias = Annotated[
    OMAT24DatasetConfig,
    C.Field(
        description="The configuration for the dataset.",
        discriminator="type",
    ),
]
