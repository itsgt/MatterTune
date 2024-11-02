from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.loss import HuberLossConfig as HuberLossConfig
    from mattertune.finetune.loss import L2MAELossConfig as L2MAELossConfig
    from mattertune.finetune.loss import LossConfig as LossConfig
    from mattertune.finetune.loss import MAELossConfig as MAELossConfig
    from mattertune.finetune.loss import MSELossConfig as MSELossConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "HuberLossConfig":
            return importlib.import_module("mattertune.finetune.loss").HuberLossConfig
        if name == "L2MAELossConfig":
            return importlib.import_module("mattertune.finetune.loss").L2MAELossConfig
        if name == "MAELossConfig":
            return importlib.import_module("mattertune.finetune.loss").MAELossConfig
        if name == "MSELossConfig":
            return importlib.import_module("mattertune.finetune.loss").MSELossConfig
        if name == "LossConfig":
            return importlib.import_module("mattertune.finetune.loss").LossConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
