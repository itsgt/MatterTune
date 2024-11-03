from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.backbones import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.backbones import JMPBackboneConfig as JMPBackboneConfig
    from mattertune.backbones import M3GNetBackboneConfig as M3GNetBackboneConfig
    from mattertune.backbones.jmp.model import CutoffsConfig as CutoffsConfig
    from mattertune.backbones.jmp.model import (
        JMPGraphComputerConfig as JMPGraphComputerConfig,
    )
    from mattertune.backbones.jmp.model import MaxNeighborsConfig as MaxNeighborsConfig
    from mattertune.backbones.m3gnet import GraphComputerConfig as GraphComputerConfig
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
                "mattertune.backbones"
            ).FinetuneModuleBaseConfig
        if name == "GraphComputerConfig":
            return importlib.import_module(
                "mattertune.backbones.m3gnet"
            ).GraphComputerConfig
        if name == "JMPBackboneConfig":
            return importlib.import_module("mattertune.backbones").JMPBackboneConfig
        if name == "JMPGraphComputerConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).JMPGraphComputerConfig
        if name == "M3GNetBackboneConfig":
            return importlib.import_module("mattertune.backbones").M3GNetBackboneConfig
        if name == "MaxNeighborsConfig":
            return importlib.import_module(
                "mattertune.backbones.jmp.model"
            ).MaxNeighborsConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import jmp as jmp
from . import m3gnet as m3gnet
