from __future__ import annotations

import contextlib
from typing import Any


@contextlib.contextmanager
def optional_import_error_message(pip_package_name: str, /):
    try:
        yield
    except ImportError as e:
        raise ImportError(
            f"The `{pip_package_name}` package is not installed. Please install it by running "
            f"`pip install {pip_package_name}`."
        ) from e


def with_defaults(d: dict[str, Any], /, **kwargs: Any) -> dict[str, Any]:
    return {**kwargs, **d}
