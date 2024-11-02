from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.wrappers.potential import PropertyConfig as PropertyConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "PropertyConfig":
            return importlib.import_module(
                "mattertune.wrappers.potential"
            ).PropertyConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import potential as potential
