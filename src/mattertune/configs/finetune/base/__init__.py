from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.base import (
        FinetuneModuleBaseConfig as FinetuneModuleBaseConfig,
    )
    from mattertune.finetune.base import LRSchedulerConfig as LRSchedulerConfig
    from mattertune.finetune.base import OptimizerConfig as OptimizerConfig
    from mattertune.finetune.base import PropertyConfig as PropertyConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "FinetuneModuleBaseConfig":
            return importlib.import_module(
                "mattertune.finetune.base"
            ).FinetuneModuleBaseConfig
        if name == "LRSchedulerConfig":
            return importlib.import_module("mattertune.finetune.base").LRSchedulerConfig
        if name == "OptimizerConfig":
            return importlib.import_module("mattertune.finetune.base").OptimizerConfig
        if name == "PropertyConfig":
            return importlib.import_module("mattertune.finetune.base").PropertyConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
