from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.backbones.jmp.model import CutoffsConfig as CutoffsConfig
    from mattertune.backbones.jmp.model import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.backbones.jmp.model import JMPBackboneConfig as JMPBackboneConfig
    from mattertune.backbones.jmp.model import (
        JMPGraphComputerConfig as JMPGraphComputerConfig,
    )
    from mattertune.backbones.jmp.model import MaxNeighborsConfig as MaxNeighborsConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "CutoffsConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).CutoffsConfig
        if name == "FinetuneModuleBaseConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).FinetuneModuleBaseConfig
        if name == "JMPBackboneConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).JMPBackboneConfig
        if name == "JMPGraphComputerConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).JMPGraphComputerConfig
        if name == "MaxNeighborsConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).MaxNeighborsConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
