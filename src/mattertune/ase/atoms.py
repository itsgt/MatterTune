from ase import Atoms
from mattertune.ase.calculator import MatterTuneASECalculator


h2o = Atoms('H2O', positions=[[0, 0, 0], [0, 0, 1.1], [0, 1.1, 0]])
energy = h2o.get_potential_energy()


class MatterTuneAtoms(Atoms):
    """
    Class that extends ASE Atoms class to support customized property prediction.
    """
    def get_property(self, property_name: str):
        """
        Get the property of the atoms object.
        """
        if self._calc is None:
            raise RuntimeError('MatterTuneAtoms object has no calculator.')
        if not isinstance(self._calc, MatterTuneASECalculator):
            raise RuntimeError('MatterTuneAtoms object has a calculator that is not a MatterTuneASECalculator, which\
                does not support customized property prediction.')
        property = self._calc.get_property(self, property_name)
        return property
