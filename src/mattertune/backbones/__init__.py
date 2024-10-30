from __future__ import annotations

from typing import Annotated, TypeAlias

from pydantic import Field

from .jmp import JMPBackboneConfig as JMPBackboneConfig
from .jmp import JMPBackboneModule as JMPBackboneModule

BackboneConfig: TypeAlias = Annotated[
    JMPBackboneConfig,
    Field(
        description="The configuration for the backbone.",
        discriminator="type",
    ),
]
