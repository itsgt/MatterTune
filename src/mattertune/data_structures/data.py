from typing import Any, Protocol, runtime_checkable, Generic
from typing_extensions import TypeVar
from abc import abstractmethod, ABC
import jaxtyping as jt
import torch
from torch.utils.data import Dataset


@runtime_checkable
class MatterTuneDataProtocol(Protocol):
    r"""
    Base class for single strcucture objects used for forward pass.
    Three conponents are required:
    - idx: the index of the strcucture in the dataset, should be a torch.Tensor of shape (1,)
    - backbone_input: the input to the backbone model, can be any type
    - labels: the labels of the strcucture, should be a dictionary of torch.Tensor
    """
    idx: jt.Int[torch.Tensor, "1"]
    backbone_input: Any
    labels: dict[str, torch.Tensor]
    
    @property
    def num_atoms(self) -> jt.Int[torch.Tensor, "1"]:
        r"""
        Get the number of atoms in the strcucture.
        """
        raise NotImplementedError
    
    @property
    def atomic_numbers(self) -> jt.Int[torch.Tensor, "num_atoms"]:
        r"""
        Get the atomic numbers of the atoms in the strcucture.
        """
        raise NotImplementedError
    
    @property
    def positions(self) -> jt.Float[torch.Tensor, "num_atoms 3"]:
        r"""
        Get the positions of the atoms in the strcucture.
        """
        raise NotImplementedError
    
    @positions.setter
    def positions(self, value: jt.Float[torch.Tensor, "num_atoms 3"]):
        r"""
        Set the positions of the atoms in the strcucture.
        """
        raise NotImplementedError
    
    @property
    def cell(self) -> jt.Float[torch.Tensor, "3 3"]:
        r"""
        Get the cell of the strcucture.
        """
        raise NotImplementedError
    
    @cell.setter
    def cell(self, value: jt.Float[torch.Tensor, "3 3"]):
        r"""
        Set the cell of the strcucture.
        """
        raise NotImplementedError
    
    @property
    def strain(self) -> jt.Float[torch.Tensor, "3 3"]:
        r"""
        Get the strain/cell_displacement of the strcucture.
        """
        raise NotImplementedError
    
    @strain.setter
    def strain(self, value: jt.Float[torch.Tensor, "3 3"]):
        r"""
        Set the strain/cell_displacement of the strcucture.
        """
        raise NotImplementedError
    
TMatterTuneData = TypeVar("TMatterTuneData", bound=MatterTuneDataProtocol, infer_variance=True)
    
@runtime_checkable
class MatterTuneBatchProtocol(Protocol):
    r"""
    Base class for batched strcucture objects used for forward pass.
    Three conponents are required:
    - idx: the index of the strcucture in the dataset
    - backbone_input: the input to the backbone model, can be any type
    - labels: the labels of the strcuctures, should be a dictionary of torch.Tensor
    """
    idx: jt.Int[torch.Tensor, "num_graphs_in_batch"]
    backbone_input: Any
    labels: dict[str, torch.Tensor]
    
    @property
    def batch(self) -> jt.Int[torch.Tensor, "num_atoms_in_batch"]:
        r"""
        Get the structure index of the atoms in the batch.
        """
        raise NotImplementedError
    
    @property
    def num_atoms(self) -> jt.Int[torch.Tensor, "num_graphs_in_batch"]:
        r"""
        Get the number of atoms in the strcucture.
        """
        raise NotImplementedError
    
    @property
    def atomic_numbers(self) -> jt.Int[torch.Tensor, "num_atoms_in_batch"]:
        r"""
        Get the atomic numbers of the atoms in the strcuctures.
        """
        raise NotImplementedError
    
    @property
    def positions(self) -> jt.Float[torch.Tensor, "num_atoms_in_batch 3"]:
        r"""
        Get the positions of the atoms in the strcuctures.
        """
        raise NotImplementedError
    
    @positions.setter
    def positions(self, value: jt.Float[torch.Tensor, "num_atoms_in_batch 3"]):
        r"""
        Set the positions of the atoms in the strcuctures.
        """
        raise NotImplementedError
    
    @property
    def cell(self) -> jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]:
        r"""
        Get the cell of the strcuctures.
        """
        raise NotImplementedError
    
    @cell.setter
    def cell(self, value: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]):
        r"""
        Set the cell of the strcuctures.
        """
        raise NotImplementedError
    
    @property
    def strain(self) -> jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]:
        r"""
        Get the strain/cell_displacement of the strcuctures.
        """
        raise NotImplementedError
    
    @strain.setter
    def strain(self, value: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]):
        r"""
        Set the strain/cell_displacement of the strcuctures.
        """
        raise NotImplementedError
    
TMatterTuneBatch = TypeVar("TMatterTuneBatch", bound=MatterTuneBatchProtocol, infer_variance=True)

    
class MatterTuneDataSetBase(Dataset, Generic[TMatterTuneData]):
    r"""
    Base dataset class for MatterTuneDataBase and its subclasses.
    """
    def __init__(
        self,
        data_list: list[TMatterTuneData],
    ):
        super().__init__()
        self.data_list = data_list
        
    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx) -> TMatterTuneData:
        return self.data_list[idx]
    