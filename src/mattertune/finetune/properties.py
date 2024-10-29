from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field
from typing_extensions import override

from .loss import LossConfig

if TYPE_CHECKING:
    from .metrics import MetricBase

log = logging.getLogger(__name__)


class PropertyConfigBase(BaseModel, ABC):
    loss: LossConfig
    """The loss function to use when training the model on this property."""

    loss_coefficient: float = 1.0
    """The coefficient to apply to this property's loss function when training the model."""

    @abstractmethod
    @classmethod
    def metric_cls(cls) -> type[MetricBase]: ...


class GraphPropertyConfig(PropertyConfigBase):
    type: Literal["graph_property"] = "graph_property"

    @override
    @classmethod
    def metric_cls(cls):
        from .metrics import GraphPropertyMetrics

        return GraphPropertyMetrics


class EnergyPropertyConfig(PropertyConfigBase):
    type: Literal["energy"] = "energy"

    @override
    @classmethod
    def metric_cls(cls):
        from .metrics import GraphPropertyMetrics

        return GraphPropertyMetrics


class ForcePropertyConfig(PropertyConfigBase):
    type: Literal["force"] = "force"

    @override
    @classmethod
    def metric_cls(cls):
        from .metrics import GraphPropertyMetrics

        return GraphPropertyMetrics


class StressPropertyConfig(PropertyConfigBase):
    type: Literal["stress"] = "stress"

    @override
    @classmethod
    def metric_cls(cls):
        from .metrics import GraphPropertyMetrics

        return GraphPropertyMetrics


PropertyConfig: TypeAlias = Annotated[
    GraphPropertyConfig
    | EnergyPropertyConfig
    | ForcePropertyConfig
    | StressPropertyConfig,
    Field(discriminator="type"),
]
