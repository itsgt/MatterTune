from __future__ import annotations

import typing_extensions as typ

if typ.TYPE_CHECKING:
    from mattertune.data.json_data import JSONDatasetConfig


__codegen__ = True


# Schema entries
class JSONDatasetConfigTypedDict(typ.TypedDict, total=False):
    type: typ.Literal["json"]
    """Discriminator for the JSON dataset."""

    src: typ.Required[str | str]
    """The path to the JSON dataset."""


@typ.overload
def CreateJSONDatasetConfig(dict: JSONDatasetConfigTypedDict, /) -> JSONDatasetConfig: ...


@typ.overload
def CreateJSONDatasetConfig(
    **dict: typ.Unpack[JSONDatasetConfigTypedDict],
) -> JSONDatasetConfig: ...


def CreateJSONDatasetConfig(*args, **kwargs):
    from mattertune.data.json_data import JSONDatasetConfig

    dict = args[0] if args else kwargs
    return JSONDatasetConfig.model_validate(dict)