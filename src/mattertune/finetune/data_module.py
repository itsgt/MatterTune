from typing import Any, TypeAlias, Generic
from ..protocol import TData, TBatch
from abc import abstractmethod
import random
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

RawData: TypeAlias = Any

class ListDataset(Dataset, Generic[TData]):
    def __init__(self, data_list: list[TData]):
        self.data_list = data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> TData:
        return self.data_list[idx]


class MatterTuneBaseDataModule(
    pl.LightningDataModule,
    Generic[TData, TBatch],
):
    """
    The base class for MatterTune data modules.
    Three methods must be implemented for using this class:
    - load_raw(): load structured data from dir or file
    - process_raw(): process raw data into TData 
    - collate_fn(): collate a list of TData into a TBatch
    """
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        val_split: float = 0.2,
        test_split: float = 0.1,
        shuffle: bool = True,
        **kwargs: Any,  # Additional parameters can be added as needed
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.kwargs = kwargs

        # Initialize placeholders for datasets
        self.raw_data: list[RawData]|None = None
        self.data: list[TData]|None = None
        self.train_data: ListDataset|None = None
        self.val_data: ListDataset|None = None
        self.test_data: ListDataset|None = None

    @abstractmethod
    def load_raw(self, **kwargs: Any) -> list[RawData]:
        """
        Load raw data from somewhere.
        """
        pass

    @abstractmethod
    def process_raw(self, raw_data_list: list[RawData], **kwargs: Any) -> list[TData]:
        """
        Process raw data into TData.
        """
        pass

    @abstractmethod
    def collate_fn(self, data_list: list[TData]) -> TBatch:
        """
        Collate a list of TData into a TBatch.
        """
        pass
    
    @abstractmethod
    def prepare_data(self) -> None:
        """
        This method is called only from a single process in distributed settings.
        Use it to download data and do any data preparation that should be done only once.
        """
        # Load and process data here if needed
        # # Example: Download dataset if not already present
        # if not os.path.exists(self.data_dir):
        #     download_dataset(self.data_dir)

        # # Example: Extract data if not already done
        # if not os.path.exists(self.extracted_data_dir):
        #     extract_dataset(self.data_dir, self.extracted_data_dir)

        # # Optionally, perform any heavy, shared preprocessing and save the results
        # if not os.path.exists(self.preprocessed_data_file):
        #     raw_data = self.load_raw(**self.kwargs)
        #     preprocessed_data = self.shared_preprocessing(raw_data)
        #     save_preprocessed_data(preprocessed_data, self.preprocessed_data_file)
        pass

    def setup(self, stage: str|None = None) -> None:
        """
        Split the data into train, validation, and test datasets.
        This method is called on every GPU in distributed settings.
        """
        if self.raw_data is None:
            self.raw_data = self.load_raw(**self.kwargs)
        
        if self.data is None:
            self.data = self.process_raw(self.raw_data, **self.kwargs)

        if self.train_data is None or self.val_data is None or self.test_data is None:
            # Implement default splitting
            total_size = len(self.data)
            indices = list(range(total_size))
            if self.shuffle:
                random.shuffle(indices)

            test_size = int(total_size * self.test_split)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size - test_size

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

            # Create datasets using ListDataset
            self.train_data = ListDataset([self.data[i] for i in train_indices])
            self.val_data = ListDataset([self.data[i] for i in val_indices])
            self.test_data = ListDataset([self.data[i] for i in test_indices])

    def train_dataloader(self) -> DataLoader:
        if self.train_data is None:
            raise ValueError("train_data is not set.")
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_data is None:
            raise ValueError("val_data is not set.")
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_data is None:
            raise ValueError("test_data is not set.")
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        
        
## TODO: Load data and predict