from __future__ import annotations

from typing import Annotated, TypeAlias

import nshconfig as C

# from .jmp import JMPBackboneConfig as JMPBackboneConfig
# from .jmp import JMPBackboneModule as JMPBackboneModule
from .m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
from .m3gnet import M3GNetBackboneModule as M3GNetBackboneModule

BackboneConfig: TypeAlias = Annotated[
    # JMPBackboneConfig,
    M3GNetBackboneConfig,
    C.Field(
        description="The configuration for the backbone.",
        discriminator="type",
    ),
]
