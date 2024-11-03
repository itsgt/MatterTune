from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.backbones.m3gnet import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.backbones.m3gnet import GraphComputerConfig as GraphComputerConfig
    from mattertune.backbones.m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "FinetuneModuleBaseConfig":
            return importlib.import_module(
                "mattertune.backbones.m3gnet"
            ).FinetuneModuleBaseConfig
        if name == "GraphComputerConfig":
            return importlib.import_module(
                "mattertune.backbones.m3gnet"
            ).GraphComputerConfig
        if name == "M3GNetBackboneConfig":
            return importlib.import_module(
                "mattertune.backbones.m3gnet"
            ).M3GNetBackboneConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
