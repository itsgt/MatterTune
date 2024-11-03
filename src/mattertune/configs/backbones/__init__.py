from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.backbones import BackboneConfig as BackboneConfig
    from mattertune.backbones import JMPBackboneConfig as JMPBackboneConfig
    from mattertune.backbones.jmp import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.backbones.jmp import GraphComputerConfig as GraphComputerConfig
    from mattertune.backbones.m3gnet import M3GNetBackboneConfig as M3GNetBackboneConfig
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
            return importlib.import_module("mattertune.backbones").JMPBackboneConfig
        if name == "BackboneConfig":
            return importlib.import_module("mattertune.backbones").BackboneConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import jmp as jmp
