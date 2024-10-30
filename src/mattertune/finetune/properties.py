from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal, TypeAlias

import nshconfig as C
import numpy as np
import torch
from ase import Atoms
from typing_extensions import assert_never, override

from .loss import LossConfig

if TYPE_CHECKING:
    from ase import Atoms

    from .metrics import MetricBase

log = logging.getLogger(__name__)

DType: TypeAlias = Literal["int", "float"]
"""The type of the property values."""


class PropertyConfigBase(C.Config, ABC):
    name: str
    """The name of the property.
    This is the key that will be used to access the property in the output of the model.
    This is also the key that will be used to access the property in the ASE Atoms object."""

    dtype: DType
    """The type of the property values."""

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

    def _torch_dtype(self) -> torch.dtype:
        """Internal helper to convert the dtype to a torch dtype."""
        match self.dtype:
            case "int":
                return torch.long
            case "float":
                return torch.float
            case _:
                assert_never(self.dtype)

    def _from_ase_atoms_to_torch(self, atoms: Atoms) -> torch.Tensor:
        """Internal helper to convert the property value from an ASE Atoms object to a torch tensor."""
        value = self.from_ase_atoms(atoms)
        return torch.tensor(value, dtype=self._torch_dtype())


class GraphPropertyConfig(PropertyConfigBase):
    type: Literal["graph_property"] = "graph_property"

    @override
    def from_ase_atoms(self, atoms):
        return atoms.info[self.name]


class EnergyPropertyConfig(PropertyConfigBase):
    type: Literal["energy"] = "energy"

    name: str = "energy"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: DType = "float"
    """The type of the property values."""

    @override
    def from_ase_atoms(self, atoms):
        return atoms.get_total_energy()


class ForcesPropertyConfig(PropertyConfigBase):
    type: Literal["forces"] = "forces"

    name: str = "forces"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: DType = "float"
    """The type of the property values."""

    conservative: bool
    """
    Whether the forces are energy conserving.
    This is used by the backbone to decide the type of output head to use for
        this property. Conservative force predictions are computed by taking the
        negative gradient of the energy with respect to the atomic positions, whereas
        non-conservative forces may be computed by other means.
    """

    @override
    def from_ase_atoms(self, atoms):
        return atoms.get_forces()


class StressesPropertyConfig(PropertyConfigBase):
    type: Literal["stresses"] = "stresses"

    name: str = "stresses"
    """The name of the property.
    This is the key that will be used to access the property in the output of the model."""

    dtype: DType = "float"
    """The type of the property values."""

    conservative: bool
    """
    Similar to the `conservative` parameter in `ForcesPropertyConfig`, this parameter
        specifies whether the stresses should be computed in a conservative manner.
    """

    @override
    def from_ase_atoms(self, atoms):
        return atoms.get_stress()


PropertyConfig: TypeAlias = Annotated[
    GraphPropertyConfig
    | EnergyPropertyConfig
    | ForcesPropertyConfig
    | StressesPropertyConfig,
    C.Field(
        description="The configuration for the property.",
        discriminator="type",
    ),
]
