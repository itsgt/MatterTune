from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.optimizer import AdamConfig as AdamConfig
    from mattertune.finetune.optimizer import AdamWConfig as AdamWConfig
    from mattertune.finetune.optimizer import OptimizerConfig as OptimizerConfig
    from mattertune.finetune.optimizer import SGDConfig as SGDConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "AdamConfig":
            return importlib.import_module("mattertune.finetune.optimizer").AdamConfig
        if name == "AdamWConfig":
            return importlib.import_module("mattertune.finetune.optimizer").AdamWConfig
        if name == "SGDConfig":
            return importlib.import_module("mattertune.finetune.optimizer").SGDConfig
        if name == "OptimizerConfig":
            return importlib.import_module(
                "mattertune.finetune.optimizer"
            ).OptimizerConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
