from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import numpy as np
import torch
from pydantic import BaseModel, Field
from typing_extensions import override

from .loss import LossConfig

if TYPE_CHECKING:
    from ase import Atoms

    from .metrics import MetricBase

log = logging.getLogger(__name__)


class PropertyConfigBase(BaseModel, ABC):
    loss: LossConfig
    """The loss function to use when training the model on this property."""

    loss_coefficient: float = 1.0
    """The coefficient to apply to this property's loss function when training the model."""

    @abstractmethod
    def from_ase_atoms(self, atoms: Atoms) -> int | float | np.ndarray | torch.Tensor:
        """Extract the property value from an ASE Atoms object."""

    @classmethod
    def metric_cls(cls) -> type[MetricBase]:
        from .metrics import PropertyMetrics

        return PropertyMetrics


class GraphPropertyConfig(PropertyConfigBase):
    type: Literal["graph_property"] = "graph_property"

    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model.
    This is also the key that will be used to access the property in the ASE Atoms object."""

    @override
    def from_ase_atoms(self, atoms: Atoms) -> np.ndarray:
        return atoms.get_properties([self.name])


class EnergyPropertyConfig(PropertyConfigBase):
    type: Literal["energy"] = "energy"

    name: str = "energy"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""


class ForcesPropertyConfig(PropertyConfigBase):
    type: Literal["forces"] = "forces"

    name: str = "forces"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""


class StressesPropertyConfig(PropertyConfigBase):
    type: Literal["stresses"] = "stresses"

    name: str = "stresses"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""


PropertyConfig: TypeAlias = Annotated[
    GraphPropertyConfig
    | EnergyPropertyConfig
    | ForcesPropertyConfig
    | StressesPropertyConfig,
    Field(discriminator="type"),
]
