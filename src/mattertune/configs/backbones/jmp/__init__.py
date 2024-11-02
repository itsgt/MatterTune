from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.backbones.jmp import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.backbones.jmp import GraphComputerConfig as GraphComputerConfig
    from mattertune.backbones.jmp import JMPBackboneConfig as JMPBackboneConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "FinetuneModuleBaseConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp"
            ).FinetuneModuleBaseConfig
        if name == "GraphComputerConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp"
            ).GraphComputerConfig
        if name == "JMPBackboneConfig":
            return importlib.import_module("mattertune.backbones.jmp").JMPBackboneConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
