import torch
from ase import Atoms
from abc import ABC, abstractmethod
from typing_extensions import override
from pydantic import BaseModel


class RawDataProviderBaseConfig(BaseModel, ABC):
    """
    Base class for raw data provider config.
    """
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    @abstractmethod
    def build_provider(self) -> "RawDataProviderBase": ...

class RawDataProviderBase(ABC):
    """
    Base class for raw data provider.
    """
    @abstractmethod
    def get_train_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]: ...
    
    @abstractmethod
    def get_val_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]: ...
    
    @abstractmethod
    def get_test_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]: ...


from ase.io import read
import numpy as np

class RawASEAtomsDataProviderConfig(RawDataProviderBaseConfig):
    """
    Config for RawASEAtomsDataProvider. Build raw data provider from ase.Atoms list.
    """
    atoms_list: list[Atoms]
    labels: list[dict[str, torch.Tensor]]
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    
    def build_provider(self) -> "RawDataProviderBase":
        return RawASEAtomsDataProvider(
            atoms_list=self.atoms_list,
            labels=self.labels,
            val_split=self.val_split,
            test_split=self.test_split,
            shuffle=self.shuffle,
        )

class RawASEAtomsDataProvider(RawDataProviderBase):
    """
    Raw data provider from ase.Atoms list.
    """
    def __init__(
        self,
        atoms_list: list[Atoms],
        labels: list[dict[str, torch.Tensor]],
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
    ):
        self.train_indices = np.arange(len(atoms_list))
        if shuffle:
            np.random.shuffle(self.train_indices)
        self.train_indices = self.train_indices[int(len(atoms_list) * (val_split + test_split)):]
        self.val_indices = np.arange(len(atoms_list))[:int(len(atoms_list) * val_split)]
        self.test_indices = np.arange(len(atoms_list))[int(len(atoms_list) * val_split):int(len(atoms_list) * (val_split + test_split))]
        self.train_atoms_list = [atoms_list[i] for i in self.train_indices]
        self.train_labels = [labels[i] for i in self.train_indices]
        self.val_atoms_list = [atoms_list[i] for i in self.val_indices]
        self.val_labels = [labels[i] for i in self.val_indices]
        self.test_atoms_list = [atoms_list[i] for i in self.test_indices]
        self.test_labels = [labels[i] for i in self.test_indices]
        
    @override
    def get_train_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.train_atoms_list, self.train_labels
    
    @override
    def get_val_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.val_atoms_list, self.val_labels
    
    @override
    def get_test_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.test_atoms_list, self.test_labels
    

class RawEFSDataProviderFromXYZConfig(RawDataProviderBaseConfig):
    """
    Config for RawEFSDataProviderFromXYZ.
    """
    file_path: str|list[str]
    val_split: float = 0.1
    test_split: float = 0.1
    shuffle: bool = True
    include_forces: bool = True
    include_stress: bool = True
    
    def build_provider(self) -> "RawDataProviderBase":
        return RawEFSDataProviderFromXYZ(
            file_path=self.file_path,
            val_split=self.val_split,
            test_split=self.test_split,
            shuffle=self.shuffle,
            include_forces=self.include_forces,
            include_stress=self.include_stress,
        )

class RawEFSDataProviderFromXYZ(RawDataProviderBase):
    """
    Read raw data from xyz file.
    And split all data into train, val, test.
    Only support efs labels.
    Get the labels from atoms.get_poteintial_energy(), atoms.get_forces(), and atoms.get_stress()
    """
    def __init__(
        self,
        file_path: str|list[str],
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        include_forces: bool = True,
        include_stress: bool = True,
    ):
        self.include_forces = include_forces
        self.include_stress = include_stress
        
        atoms_list = []
        if isinstance(file_path, str):
            file_path = [file_path]
        for xyz_file in file_path:
            atoms_list += read(xyz_file, ":")
        indices = np.arange(len(atoms_list))
        if shuffle:
            np.random.shuffle(indices)
        self.train_indices = indices[int(len(indices) * (val_split + test_split)):]
        self.val_indices = indices[:int(len(indices) * val_split)]
        self.test_indices = indices[int(len(indices) * val_split):int(len(indices) * (val_split + test_split))]
        self.train_atoms_list = [atoms_list[i] for i in self.train_indices]
        self.train_labels = self.get_labels(self.train_atoms_list)
        self.val_atoms_list = [atoms_list[i] for i in self.val_indices]
        self.val_labels = self.get_labels(self.val_atoms_list)
        self.test_atoms_list = [atoms_list[i] for i in self.test_indices]
        self.test_labels = self.get_labels(self.test_atoms_list)
        
    def get_labels(self, atoms_list: list[Atoms]) -> list[dict[str, torch.Tensor]]:
        labels = []
        for atoms in atoms_list:
            label = {}
            label["energy"] = torch.tensor([atoms.get_potential_energy()], dtype=torch.float)
            if self.include_forces:
                label["forces"] = torch.tensor(atoms.get_forces(), dtype=torch.float)
            if self.include_stress:
                label["stress"] = torch.tensor(atoms.get_stress(voigt=False), dtype=torch.float)
            labels.append(label)
        return labels
    
    @override
    def get_train_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.train_atoms_list, self.train_labels
    
    @override
    def get_val_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.val_atoms_list, self.val_labels
    
    @override
    def get_test_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
        return self.test_atoms_list, self.test_labels
        
# from pymatgen.core.structure import Structure
# from pymatgen.io.ase import AseAtomsAdaptor

# class RawStructureDataProviderConfig(RawDataProviderBaseConfig):
#     """
#     Config for RawStructureDataProvider. Build raw data provider from pymatgen.Structure list.
#     """
#     structures: list[Structure]
#     labels: list[dict[str, torch.Tensor]]
#     val_split: float = 0.1
#     test_split: float = 0.1
#     shuffle: bool = True
    
#     def build_provider(self) -> "RawDataProviderBase":
#         return RawStructureDataProvider(
#             structures=self.structures,
#             labels=self.labels,
#             val_split=self.val_split,
#             test_split=self.test_split,
#             shuffle=self.shuffle,
#         )

# class RawStructureDataProvider(RawDataProviderBase):
#     """
#     Raw data provider from pymatgen.Structure list.
#     """
#     def __init__(
#         self,
#         structures: list[Structure],
#         labels: list[dict[str, torch.Tensor]],
#         val_split: float = 0.1,
#         test_split: float = 0.1,
#         shuffle: bool = True,
#     ):
#         adaptor = AseAtomsAdaptor()
#         atoms_list = [adaptor.get_atoms(structure) for structure in structures]
#         self.train_indices = np.arange(len(atoms_list))
#         if shuffle:
#             np.random.shuffle(self.train_indices)
#         self.train_indices = self.train_indices[int(len(atoms_list) * (val_split + test_split)):]
#         self.val_indices = np.arange(len(atoms_list))[:int(len(atoms_list) * val_split)]
#         self.test_indices = np.arange(len(atoms_list))[int(len(atoms_list) * val_split):int(len(atoms_list) * (val_split + test_split))]
#         self.train_atoms_list = [atoms_list[i] for i in self.train_indices]
#         self.train_labels = [labels[i] for i in self.train_indices]
#         self.val_atoms_list = [atoms_list[i] for i in self.val_indices]
#         self.val_labels = [labels[i] for i in self.val_indices]
#         self.test_atoms_list = [atoms_list[i] for i in self.test_indices]
#         self.test_labels = [labels[i] for i in self.test_indices]
        
#     @override
#     def get_train_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
#         return self.train_atoms_list, self.train_labels
    
#     @override
#     def get_val_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
#         return self.val_atoms_list, self.val_labels
    
#     @override
#     def get_test_data(self) -> tuple[list[Atoms], list[dict[str, torch.Tensor]]]:
#         return self.test_atoms_list, self.test_labels