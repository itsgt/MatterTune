from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.finetune.properties import (
        EnergyPropertyConfig as EnergyPropertyConfig,
    )
    from mattertune.finetune.properties import (
        ForcesPropertyConfig as ForcesPropertyConfig,
    )
    from mattertune.finetune.properties import (
        GraphPropertyConfig as GraphPropertyConfig,
    )
    from mattertune.finetune.properties import LossConfig as LossConfig
    from mattertune.finetune.properties import PropertyConfig as PropertyConfig
    from mattertune.finetune.properties import PropertyConfigBase as PropertyConfigBase
    from mattertune.finetune.properties import (
        StressesPropertyConfig as StressesPropertyConfig,
    )
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "EnergyPropertyConfig":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).EnergyPropertyConfig
        if name == "ForcesPropertyConfig":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).ForcesPropertyConfig
        if name == "GraphPropertyConfig":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).GraphPropertyConfig
        if name == "PropertyConfigBase":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).PropertyConfigBase
        if name == "StressesPropertyConfig":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).StressesPropertyConfig
        if name == "LossConfig":
            return importlib.import_module("mattertune.finetune.properties").LossConfig
        if name == "PropertyConfig":
            return importlib.import_module(
                "mattertune.finetune.properties"
            ).PropertyConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
