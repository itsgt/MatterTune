from __future__ import annotations

from pydantic import BaseModel

from ..backbones import BackboneConfig
from ..data import DatasetConfig


class PerSplitDataConfig(BaseModel):
    train: DatasetConfig
    """The configuration for the training data."""

    validation: DatasetConfig
    """The configuration for the validation data."""


class TunerConfig(BaseModel):
    data: PerSplitDataConfig
    """The configuration for the data."""

    model: BackboneConfig
    """The configuration for the model."""


class MatterTuner:
    pass
