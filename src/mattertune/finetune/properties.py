from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Literal

import nshconfig as C
import numpy as np
import torch
from ase import Atoms
from typing_extensions import TypeAliasType, assert_never, override

from .loss import LossConfig

if TYPE_CHECKING:
    from .metrics import MetricBase

log = logging.getLogger(__name__)

DType = TypeAliasType("DType", Literal["float"])
"""The type of the property values."""

ASECalculatorPropertyName = TypeAliasType(
    "ASECalculatorPropertyName",
    Literal[
        "energy",
        "forces",
        "stress",
        "dipole",
        "charges",
        "magmom",
        "magmoms",
    ],
)


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
            case "float":
                return torch.float
            case _:
                assert_never(self.dtype)

    def _numpy_dtype(self):
        """Internal helper to convert the dtype to a numpy dtype."""
        match self.dtype:
            case "float":
                return np.float32
            case _:
                assert_never(self.dtype)

    def _from_ase_atoms_to_torch(self, atoms: Atoms) -> torch.Tensor:
        """Internal helper to convert the property value from an ASE Atoms object to a torch tensor."""
        value = self.from_ase_atoms(atoms)
        return torch.tensor(value, dtype=self._torch_dtype())

    @abstractmethod
    def ase_calculator_property_name(self) -> ASECalculatorPropertyName | None:
        """
        If this property can be calculated by an ASE calculator, returns
        the name of the property that the ASE calculator uses. Otherwise,
        returns None.

        This should only return non-None for properties that are supported by
        the ASE calculator interface, i.e.:
        - 'energy'
        - 'forces'
        - 'stress'
        - 'dipole'
        - 'charges'
        - 'magmom'
        - 'magmoms'

        Note that this does not refer to the new experimental custom property
        prediction support feature in ASE, but rather the built-in properties
        that ASE can calculate in the ``ase.calculators.calculator.Calculator``
        class.
        """

    @abstractmethod
    def property_type(self) -> Literal["system", "atom"]: ...

    def prepare_value_for_ase_calculator(self, value: float | np.ndarray):
        """Convert the property value to a format that can be used by the ASE calculator."""
        return value


class GraphPropertyConfig(PropertyConfigBase):
    type: Literal["graph_property"] = "graph_property"

    reduction: Literal["mean", "sum", "max"]
    """The reduction to use for the output.
    - "sum": Sum the property values for all atoms in the system.
    This is optimal for extensive properties (e.g. energy).
    - "mean": Take the mean of the property values for all atoms in the system.
    This is optimal for intensive properties (e.g. density).
    - "max": Take the maximum of the property values for all atoms in the system.
    This is optimal for properties like the `last phdos peak` of Matbench's phonons dataset.
    """

    @override
    def from_ase_atoms(self, atoms):
        return atoms.info[self.name]

    @override
    def ase_calculator_property_name(self):
        return None

    @override
    def property_type(self):
        return "system"


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

    @override
    def ase_calculator_property_name(self):
        return "energy"

    @override
    def prepare_value_for_ase_calculator(self, value):
        if isinstance(value, np.ndarray):
            return value.item()
        return value

    @override
    def property_type(self):
        return "system"


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

    @override
    def ase_calculator_property_name(self):
        return "forces"

    @override
    def property_type(self):
        return "atom"


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

    @override
    def ase_calculator_property_name(self):
        return "stress"

    @override
    def prepare_value_for_ase_calculator(self, value):
        assert isinstance(value, np.ndarray), "Stress must be a numpy array."

        from ase.constraints import full_3x3_to_voigt_6_stress

        return full_3x3_to_voigt_6_stress(value.reshape((3, 3)))

    @override
    def property_type(self):
        return "system"


class AtomInvariantVectorPropertyConfig(PropertyConfigBase):
    """Configuration class for atom-level vector properties that are invariant to rotations.

    This class handles properties that are associated with individual atoms and remain unchanged
    under molecular rotations (rotational invariance). Examples include atomic spectra or other
    per-atom vector quantities that don't transform under rotation.

    This is distinct from equivariant properties like forces, which transform with the rotation
    of the molecule.
    """

    type: Literal["atom_invariant_vector"] = "atom_invariant_vector"

    size: C.PositiveInt
    """The size of the vector property associated with each atom.

    This parameter specifies the dimensionality of the vector property for each atom
    in the system. For example, if representing atomic spectra, this would be the
    number of spectral components per atom.
    """

    @override
    def from_ase_atoms(self, atoms):
        return atoms.info[self.name]

    @override
    def ase_calculator_property_name(self):
        return None

    @override
    def property_type(self):
        return "atom"


class MLPAtomDensityHeadConfig(C.Config):
    type: Literal["mlp"] = "mlp"

    # hidden_sizes: list[C.PositiveInt]
    # """The sizes of the hidden layers in the MLP."""
    # TODO: Implement this later

    final_activation: Literal["none", "softplus", "relu"] = "softplus"
    """The activation function to use for the final layer of the MLP.

    Ideally, this should be a non-negative activation function, as the output of the
    MLP is always a non-negative quantity.
    """


class MDNAtomDensityHeadConfig(C.Config):
    type: Literal["mdn"] = "mdn"

    # hidden_sizes: list[C.PositiveInt]
    # """The sizes of the hidden layers in the MLP used for the MDN head."""
    # TODO: Implement this later

    num_components: C.PositiveInt
    """The number of components in the mixture density network."""


AtomDensityHeadConfig = TypeAliasType(
    "AtomDensityHeadConfig",
    Annotated[
        MLPAtomDensityHeadConfig | MDNAtomDensityHeadConfig,
        C.Field(discriminator="type"),
    ],
)


class AtomDensityPropertyConfig(PropertyConfigBase):
    """
    Configuration class for atom-level density properties.

    This class handles properties that are associated with individual atoms and are
    density-like in nature. Examples include RDF spectra.
    """

    type: Literal["atom_density"] = "atom_density"

    num_bins: C.PositiveInt
    """The number of bins to use for the density histogram."""

    bin_range: tuple[float, float]
    """The range of the bins to use for the density histogram."""

    output_head: AtomDensityHeadConfig
    """The configuration for the output head to use for this property.

    - MLPAtomDensityHeadConfig: Configuration for an MLP output head.
    - MDNAtomDensityHeadConfig: Configuration for a mixture density network output head.
    """

    @override
    def from_ase_atoms(self, atoms):
        return atoms.info[self.name]

    @override
    def ase_calculator_property_name(self):
        return None

    @override
    def property_type(self):
        return "atom"


PropertyConfig = TypeAliasType(
    "PropertyConfig",
    Annotated[
        GraphPropertyConfig
        | EnergyPropertyConfig
        | ForcesPropertyConfig
        | StressesPropertyConfig
        | AtomInvariantVectorPropertyConfig
        | AtomDensityPropertyConfig,
        C.Field(
            description="The configuration for the property.",
            discriminator="type",
        ),
    ],
)
