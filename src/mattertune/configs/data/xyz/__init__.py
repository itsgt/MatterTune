from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.data.xyz import XYZDatasetConfig as XYZDatasetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "XYZDatasetConfig":
            return importlib.import_module("mattertune.data.xyz").XYZDatasetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
