from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.data.omat24 import OMAT24DatasetConfig as OMAT24DatasetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "OMAT24DatasetConfig":
            return importlib.import_module("mattertune.data.omat24").OMAT24DatasetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
