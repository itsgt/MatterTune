from ase.calculators.calculator import Calculator
from ase import Atoms
from ase.constraints import full_3x3_to_voigt_6_stress
from ase.outputs import _defineprop, all_outputs
from mattertune.potential import MatterTunePotential
from typing_extensions import override
import torch

_defineprop("band_gap", "num_atoms, 3")


class MatterTuneASECalculator(Calculator):
    implemented_properties: list[str] = []
    """
    Wrap a MatterTunePotential object to be used as an ASE calculator.
    """
    def __init__(
        self,
        *,
        potential: MatterTunePotential,
        stress_coeff: float, ## Coefficient to scale the stress predictions, make sure to convert to eV/Angstrom^3
        target_name_map: dict[str, str]|None = None, ## Mapping from output head target_name to property name
    ):
        super().__init__()
        self.potential = potential
        self.stress_coeff = stress_coeff
        self.implemented_properties = self.potential.get_supported_properties()
        if target_name_map is not None:
            self.target_name_map = target_name_map
        else:
            self.target_name_map = {prop: prop for prop in self.implemented_properties}
        self.property_name_map = {v: k for k, v in self.target_name_map.items()} ## Reverse mapping from property name to output head target_name
        self.implemented_properties = [self.target_name_map[prop] for prop in self.implemented_properties]
        
        for prop in self.implemented_properties:
            _defineprop(prop, float, shape=("?"))
        
    
    @override
    def calculate(
        self,
        atoms: Atoms|None = None,
        properties: list|None = None,
        system_changes: list|None = None,
    ):
        all_changes = ['positions', 'numbers', 'cell', 'pbc',
               'initial_charges', 'initial_magmoms']
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        
        assert atoms is not None, "atoms must be provided for calculation"
        assert set(properties).issubset(set(self.implemented_properties)), f"looking for {properties}, but only {self.implemented_properties} are implemented, \
            please check the supported properties of the model or set target_name_map of the calculator"
        
        predictions = self.potential.predict([atoms])
        
        for prop in properties:
            target_name = self.property_name_map[prop]
            prop_value = predictions[target_name].detach().to(dtype=torch.float32).cpu().numpy()
            if prop_value.reshape(-1).shape == (1,):
                prop_value = prop_value[0]
            if prop == "stress":
                if prop_value.shape == (6,):
                    prop_value = self.stress_coeff * prop_value
                elif prop_value.shape == (3, 3):
                    prop_value = self.stress_coeff * full_3x3_to_voigt_6_stress(prop_value)
                else:
                    raise ValueError(f"Invalid stress shape {prop_value.shape}")
            self.results[prop] = prop_value
    
    @override
    def calculate_properties(self, atoms, properties: str|list[str]):
        if isinstance(properties, str):
            properties = [properties]
        self.calculate(atoms=atoms, properties=properties)
        predicted_properties = {prop: self.results[prop] for prop in properties}
        if len(predicted_properties) == 1:
            return predicted_properties[properties[0]]
        else:
            return predicted_properties