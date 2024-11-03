from __future__ import annotations

import nshconfig as C

from .finetune.base import FinetuneModuleBaseConfig

backbone_registry = C.Registry(
    FinetuneModuleBaseConfig,
    discriminator_field="name",
)
