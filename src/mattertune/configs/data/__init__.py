from __future__ import annotations

__codegen__ = True

from typing import TYPE_CHECKING

# Config/alias imports

if TYPE_CHECKING:
    from mattertune.data import DatasetConfig as DatasetConfig
    from mattertune.data import OMAT24DatasetConfig as OMAT24DatasetConfig
else:

    def __getattr__(name):
        import importlib

        if name in globals():
            return globals()[name]
        if name == "OMAT24DatasetConfig":
            return importlib.import_module("mattertune.data").OMAT24DatasetConfig
        if name == "DatasetConfig":
            return importlib.import_module("mattertune.data").DatasetConfig
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# Submodule exports
from . import omat24 as omat24
