from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.lr_scheduler import (
        CosineAnnealingLRConfig as CosineAnnealingLRConfig,
    )
    from mattertune.finetune.lr_scheduler import ExponentialConfig as ExponentialConfig
    from mattertune.finetune.lr_scheduler import LRSchedulerConfig as LRSchedulerConfig
    from mattertune.finetune.lr_scheduler import MultiStepLRConfig as MultiStepLRConfig
    from mattertune.finetune.lr_scheduler import (
        ReduceOnPlateauConfig as ReduceOnPlateauConfig,
    )
    from mattertune.finetune.lr_scheduler import StepLRConfig as StepLRConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "CosineAnnealingLRConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).CosineAnnealingLRConfig
        if name == "ExponentialConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).ExponentialConfig
        if name == "MultiStepLRConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).MultiStepLRConfig
        if name == "ReduceOnPlateauConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).ReduceOnPlateauConfig
        if name == "StepLRConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).StepLRConfig
        if name == "LRSchedulerConfig":
            return importlib.import_module(
                "mattertune.finetune.lr_scheduler"
            ).LRSchedulerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
