from abc import ABC, abstractmethod
from typing import Generic
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.calculators.calculator import Calculator, all_changes, PropertyNotImplementedError
from ase.constraints import full_3x3_to_voigt_6_stress
from mattertune.finetune import FinetuneModuleBase
from mattertune.protocol import TBatch

class AseCalculator(Calculator, ABC, Generic[TBatch]):
    """ASE calculator that uses a MatterTune finetune model to predict properties."""
    implemented_properties: list[str] = ["energy", "forces", "stress"]
    """
    Implemented properties that can be calculated by the calculator.
    Default is ["energy", "forces", "stress"].
    """
    
    def __init__(
        self,
        model: FinetuneModuleBase,
        stress_coeff: float = 1.0, ## Coefficient to scale the stress predictions, make sure to convert to eV/Angstrom^3
        implemented_properties: list[str]|None = None,
        target_name_map: dict[str, str]|None = None,
        write_to_atoms_info: bool = False,
    ):
        super().__init__()
        self.model = model
        self.stress_coeff = stress_coeff
        if implemented_properties is not None:
            self.implemented_properties = implemented_properties
        if target_name_map is not None:
            self.target_name_map = target_name_map
        else:
            self.target_name_map = {prop: prop for prop in self.implemented_properties}
        self.write_to_atoms_info = write_to_atoms_info
        
    @abstractmethod
    def process_ASE_atoms(self, atoms: Atoms) -> TBatch:
        """
        Convert ASE atoms object to a batch of inputs for the model with batch_size=1.
        """
        pass
        
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
        batch = self.process_ASE_atoms(atoms)
        output_head_results = self.model(batch)
        mapped_results = {}
        for target, result in output_head_results.items():
            result = result.detach().cpu().numpy()
            if result.shape == (1,):
                result = result[0]
            if target == "stress":
                result = full_3x3_to_voigt_6_stress(result) * self.stress_coeff
            mapped_results[self.target_name_map[target]] = result
        self.results.update(mapped_results)
        if self.write_to_atoms_info:
            for target, result in mapped_results.items():
                atoms.info[target] = result