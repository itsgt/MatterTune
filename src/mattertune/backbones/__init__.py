from __future__ import annotations

from typing import Annotated, TypeAlias

import nshconfig as C

from .jmp import JMPBackboneConfig as JMPBackboneConfig
from .jmp import JMPBackboneModule as JMPBackboneModule

BackboneConfig: TypeAlias = Annotated[
    JMPBackboneConfig,
    C.Field(
        description="The configuration for the backbone.",
        discriminator="type",
    ),
]
