from abc import ABC, abstractmethod
from mattertune.protocol import TBatch, TData
from typing import Generic, TypeAlias, Annotated, Literal
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pytorch_lightning as pl
from pydantic import Field, BaseModel
import jaxtyping as jt
import numpy as np
from typing_extensions import final, override


class BaseReferenceConfig(BaseModel, ABC):
    @abstractmethod
    def compute_references(
        self,
        compositions: jt.Int[np.ndarray, "dataset_size n_atomic_numbers"],
        energies: jt.Float[np.ndarray, "dataset_size"],
    ) -> jt.Float[np.ndarray, "n_atomic_numbers"]: ...


@final
class LinearReferenceConfig(BaseReferenceConfig):
    name: Literal["linear_reference"] = "linear_reference"

    @override
    def compute_references(self, compositions, energies):
        from sklearn.linear_model import LinearRegression

        c = compositions
        y = energies
        num_chem_species = c.shape[1]

        # tweak to fine tune training from many-element to small element
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        full_coeff = np.zeros(num_chem_species)
        coef_reduced = LinearRegression(fit_intercept=False).fit(c_reduced, y).coef_
        full_coeff[~zero_indices] = coef_reduced

        return full_coeff


@final
class RidgeReferenceConfig(BaseReferenceConfig):
    name: Literal["ridge_reference"] = "ridge_reference"

    alpha: float

    @override
    def compute_references(self, compositions, energies):
        from sklearn.linear_model import Ridge

        c = compositions
        y = energies
        num_chem_species = c.shape[1]

        # tweak to fine tune training from many-element to small element
        zero_indices = np.all(c == 0, axis=0)
        c_reduced = c[:, ~zero_indices]
        full_coeff = np.zeros(num_chem_species)
        coef_reduced = (
            Ridge(alpha=self.alpha, fit_intercept=False).fit(c_reduced, y).coef_
        )
        full_coeff[~zero_indices] = coef_reduced

        return full_coeff


ReferenceConfig: TypeAlias = Annotated[
    LinearReferenceConfig | RidgeReferenceConfig,
    Field(discriminator="name"),
]


class MatterTuneDatasetBase(Dataset, ABC, Generic[TData]):
    """
    Base class for MatterTune dataset
    """
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx) -> TData:
        pass
    
    
    
class MatterTuneDataModuleBase(pl.LightningDataModule, Generic[TData, TBatch]):
    """
    The base class for MatterTune data modules.
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        ignore_data_errors: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.ignore_data_errors = ignore_data_errors

    @abstractmethod
    def setup(self, stage: str) -> None:
        """
        Setup train, validation, and test datasets
        Split the data into train, validation, and test datasets
        """
        pass
    
    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch:
        """
        Collate function for the DataLoader
        """
        pass
    
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        sampler = DistributedSampler(dataset) if self.trainer and self.trainer.num_devices > 1 else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(sampler is None) and shuffle,
            num_workers=self.num_workers,
            sampler=sampler,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )
    
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """
        DataLoader for training dataset
        """
        pass
    
    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """
        DataLoader for validation dataset
        """
        pass

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """
        DataLoader for test dataset
        """
        pass