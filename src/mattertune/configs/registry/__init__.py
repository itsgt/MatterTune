from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.registry import FinetuneModuleBaseConfig as FinetuneModuleBaseConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "FinetuneModuleBaseConfig":
            return importlib.import_module(
                "mattertune.registry"
            ).FinetuneModuleBaseConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Submodule exports
