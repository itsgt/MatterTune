from torch.utils.data import Dataset, DataLoader, DistributedSampler
from ase import Atoms
from ase.io import read
from typing_extensions import override
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from mattertune.finetune.data_module import MatterTuneDataModuleBase, MatterTuneDatasetBase, ReferenceConfig
import torch
import numpy as np
import random

class JMPDataset(MatterTuneDatasetBase):
    """
    Custom Dataset for JMP FineTune task.
    """
    def __init__(
        self,
        xyz_path: str,
        indices: list[int],
        ignore_data_errors: bool = True,
        training: bool = True,
        references: dict[str, ReferenceConfig] = {},
    ):
        self.xyz_path = xyz_path
        self.indices = indices
        self.ignore_data_errors = ignore_data_errors
        self.training = training
        self.references = references

        # Instead of loading all data, we just read the total number of samples
        # and store the file path. We will load each sample in __getitem__
        self.atoms_list: list[Atoms] = read(self.xyz_path, index=":")
        # Filter atoms_list based on indices
        self.atoms_list = [self.atoms_list[i] for i in self.indices]
        
        ## Compute atom references
        def get_chemical_composition(atoms: Atoms) -> np.ndarray:
            chemical_numbers = np.array(atoms.get_atomic_numbers()) - 1
            return np.bincount(chemical_numbers, minlength=120)
        self.atom_references: dict[str, np.ndarray] = {}
        for key, reference_config in self.references.items():
            self.atom_references[key] = reference_config.compute_references(
                np.array([get_chemical_composition(atoms) for atoms in self.atoms_list]),
                np.array([atoms.get_total_energy() for atoms in self.atoms_list]),
            )

    def __len__(self):
        return len(self.atoms_list)

    def __getitem__(self, idx):
        try:
            raw_data: Atoms = self.atoms_list[idx]
            data_dict = {
                "idx": idx,
                "atomic_numbers": torch.tensor(raw_data.get_atomic_numbers()).long(),
                "pos": torch.tensor(raw_data.get_positions()).float(),
                "num_atoms": torch.tensor([len(raw_data)]).long(),
                "cell_displacement": None,
                "cell": torch.tensor(np.array(raw_data.get_cell()), dtype=torch.float).unsqueeze(dim=0),
                "natoms": torch.tensor(len(raw_data)).long(),
                "tags": 2 * torch.ones(len(raw_data), dtype=torch.long),
                "fixed": torch.zeros_like(torch.tensor(np.array(raw_data.get_atomic_numbers())), dtype=torch.bool),
                "force": torch.tensor(np.array(raw_data.get_forces()), dtype=torch.float),
                "stress": torch.tensor(np.array(raw_data.get_stress(voigt=False)), dtype=torch.float).unsqueeze(dim=0),
                "energy": torch.tensor(raw_data.get_total_energy(), dtype=torch.float),
                "cell_displacement": torch.zeros((len(raw_data), 3, 3), dtype=torch.float),
            }
            for key, reference in self.atom_references.items():
                if (unreferenced_value := data_dict.get(key)) is None:
                    raise ValueError(f"Missing key {key} in data_dict")

                data_dict[key] = (
                    unreferenced_value - reference[data_dict["atomic_numbers"]].sum().item()
                )

            data = Data.from_dict(data_dict)
            # Now process the data to generate graphs
            return data
        except Exception as e:
            if self.ignore_data_errors:
                print(f"Error processing sample {idx}: {e}")
                return None
            else:
                raise e


class JMPDataModule(MatterTuneDataModuleBase):
    """
    DataModule for JMP FineTune task using small batch data loading.
    """
    def __init__(
        self,
        batch_size: int,
        xyz_path: str,
        num_workers: int = 0,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        ignore_data_errors: bool = True,
        references: dict[str, ReferenceConfig] = {},
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            shuffle=shuffle,
            ignore_data_errors=ignore_data_errors,
        )
        self.xyz_path = xyz_path
        self.references = references

    def setup(self, stage: str|None = None) -> None:
        # Set random seed to ensure consistency across different GPUs
        torch.manual_seed(42)
        random.seed(42)

        # Instead of loading all data, we just get the total number of samples
        atoms_list = read(self.xyz_path, index=":")
        total_size = len(atoms_list)
        indices = list(range(total_size))
        if self.shuffle:
            random.shuffle(indices)

        train_end = int(total_size * (1 - self.val_split - self.test_split))
        val_end = int(total_size * (1 - self.test_split))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        self.train_dataset = JMPDataset(
            xyz_path=self.xyz_path,
            indices=train_indices,
            ignore_data_errors=self.ignore_data_errors,
            training=True,
            references=self.references,
        )

        self.val_dataset = JMPDataset(
            xyz_path=self.xyz_path,
            indices=val_indices,
            ignore_data_errors=self.ignore_data_errors,
            training=False,
            references=self.references,
        )

        self.test_dataset = JMPDataset(
            xyz_path=self.xyz_path,
            indices=test_indices,
            ignore_data_errors=self.ignore_data_errors,
            training=False,
            references=self.references,
        )
        
    @override
    def collate_fn(self, data_list: list[BaseData]) -> Batch:
        # Remove None data (if any errors occurred and ignore_data_errors=True)
        data_list = [data for data in data_list if data is not None]
        return Batch.from_data_list(data_list)

    @override
    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=True)

    @override
    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    @override
    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)