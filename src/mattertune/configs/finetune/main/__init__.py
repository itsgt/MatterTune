from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.main import BackboneConfig as BackboneConfig
    from mattertune.finetune.main import DatasetConfig as DatasetConfig
    from mattertune.finetune.main import MatterTunerConfig as MatterTunerConfig
    from mattertune.finetune.main import PerSplitDataConfig as PerSplitDataConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "MatterTunerConfig":
            return importlib.import_module("mattertune.finetune.main").MatterTunerConfig
        if name == "PerSplitDataConfig":
            return importlib.import_module(
                "mattertune.finetune.main"
            ).PerSplitDataConfig
        if name == "BackboneConfig":
            return importlib.import_module("mattertune.finetune.main").BackboneConfig
        if name == "DatasetConfig":
            return importlib.import_module("mattertune.finetune.main").DatasetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
