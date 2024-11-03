from __future__ import annotations

from typing import Annotated, TypeAlias

from ..finetune.base import FinetuneModuleBaseConfig
from ..registry import backbone_registry
from .jmp import JMPBackboneConfig as JMPBackboneConfig
from .jmp import JMPBackboneModule as JMPBackboneModule
from .m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
from .m3gnet import M3GNetBackboneModule as M3GNetBackboneModule

ModelConfig: TypeAlias = Annotated[
    FinetuneModuleBaseConfig,
    backbone_registry.RegistryResolution(),
]
