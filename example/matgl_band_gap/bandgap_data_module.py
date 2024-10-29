"""
Based on the MGLDataset and MGLDataLoader implemented in Matgl
Add some customizations to make it compatible with MatterTune
"""

from __future__ import annotations

import random

import dgl
import jaxtyping as jt
import matgl
import numpy as np
import torch
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from matgl.graph.converters import GraphConverter
from pymatgen.core.structure import Structure
from tqdm import trange
from typing_extensions import override

from mattertune.finetune.data_module import (
    MatterTuneDataModuleBase,
    MatterTuneDatasetBase,
)


class DataTypedDict(dict):
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes"]
    pos: jt.Float[torch.Tensor, "num_nodes 3"]
    num_atoms: jt.Int[torch.Tensor, "1"]
    cell_displacement: jt.Float[torch.Tensor, "1 3 3"]
    cell: jt.Float[torch.Tensor, "1 3 3"]
    graph: dgl.DGLGraph
    line_graph: dgl.DGLGraph | None
    lattice: torch.Tensor
    state_attr: torch.Tensor
    band_gap: torch.Tensor

    def __getattr__(self, key: str):
        # Handle special cases where Python expects certain attributes to exist
        if key in self:
            return self[key]
        # If it's a built-in attribute, use the default behavior
        try:
            return super().__getattr__(key)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"{key} not found in the data.")

    def __hasattr__(self, key: str):
        return key in self


## TODO: DGL.Batch
class BatchTypedDict(dict):
    batch: jt.Int[torch.Tensor, "num_graphs"]
    atomic_numbers: jt.Int[torch.Tensor, "num_nodes_in_batch"]
    pos: jt.Float[torch.Tensor, "num_nodes_in_batch 3"]
    num_atoms: jt.Int[torch.Tensor, "num_graphs_in_batch"]
    cell_displacement: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]
    cell: jt.Float[torch.Tensor, "num_graphs_in_batch 3 3"]
    graph: dgl.DGLGraph
    line_graph: dgl.DGLGraph | None
    lattice: torch.Tensor
    state_attr: torch.Tensor
    band_gap: torch.Tensor

    def __getattr__(self, key: str):
        # Handle special cases where Python expects certain attributes to exist
        if key in self:
            return self[key]
        # If it's a built-in attribute, use the default behavior
        try:
            return super().__getattr__(key)
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{key}'"
            )

    def __setattr__(self, key: str, value):
        self[key] = value

    def __delattr__(self, key: str):
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"{key} not found in the data.")

    def __hasattr__(self, key: str):
        return key in self


class MatglDataset(MatterTuneDatasetBase):
    """
    Custom Dataset for Matgl FineTune task.
    """

    def __init__(
        self,
        structures: list[Structure],
        labels: dict[str, list],
        threebody_cutoff: float,
        converter: GraphConverter | None = None,
        include_line_graph: bool = False,
        directed_line_graph: bool = False,
        graph_labels: list[int | float] | None = None,
    ):
        super().__init__()
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures
        self.labels = labels
        self.threebody_cutoff = threebody_cutoff
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels
        self.process()

    def process(self):
        """Convert Pymatgen structure into dgl graphs."""
        num_graphs = len(self.structures)  # type: ignore
        graphs, lattices, line_graphs, state_attrs = [], [], [], []

        for idx in trange(num_graphs):
            structure = self.structures[idx]  # type: ignore
            graph, lattice, state_attr = self.converter.get_graph(structure)  # type: ignore
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)
            graph.ndata["pos"] = torch.tensor(structure.cart_coords)
            graph.edata["pbc_offshift"] = torch.matmul(
                graph.edata["pbc_offset"], lattice[0]
            )
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.edata["bond_vec"] = bond_vec
            graph.edata["bond_dist"] = bond_dist
            if self.include_line_graph:
                line_graph = create_line_graph(
                    graph, self.threebody_cutoff, directed=self.directed_line_graph
                )  # type: ignore
                for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                    line_graph.ndata.pop(name)
                line_graphs.append(line_graph)
            graph.ndata.pop("pos")
            graph.edata.pop("pbc_offshift")
        if self.graph_labels is not None:
            state_attrs = torch.tensor(self.graph_labels).long()
        else:
            state_attrs = torch.tensor(np.array(state_attrs), dtype=matgl.float_th)

        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        if self.include_line_graph:
            self.line_graphs = line_graphs
            return self.graphs, self.lattices, self.line_graphs, self.state_attr
        return self.graphs, self.lattices, self.state_attr

    @override
    def __getitem__(self, idx: int) -> DataTypedDict:
        """Get graph and label with idx."""
        atomic_numbers = torch.tensor(
            self.structures[idx].atomic_numbers, dtype=matgl.int_th
        )
        pos = torch.tensor(self.structures[idx].cart_coords, dtype=matgl.float_th)
        num_atoms = torch.tensor([len(self.structures[idx])], dtype=matgl.int_th)
        cell = torch.tensor(
            self.structures[idx].lattice.matrix, dtype=matgl.float_th
        ).reshape(1, 3, 3)
        cell_displacement = torch.zeros((1, 3, 3), dtype=matgl.float_th)

        graph = self.graphs[idx]
        lattice = self.lattices[idx]
        state_attr = self.state_attr[idx]
        labels = {k: v[idx] for k, v in self.labels.items()}
        if self.include_line_graph:
            line_graph = self.line_graphs[idx]
        else:
            line_graph = None

        data = DataTypedDict(
            atomic_numbers=atomic_numbers,
            pos=pos,
            num_atoms=num_atoms,
            cell_displacement=cell_displacement,
            cell=cell,
            graph=graph,
            line_graph=line_graph,
            lattice=lattice,
            state_attr=state_attr,
            band_gap=torch.tensor([labels.get("band_gap", 0.0)], dtype=matgl.float_th),
        )
        return data

    @override
    def __len__(self):
        """Get size of dataset."""
        return len(self.graphs)


class MatglDataModule(MatterTuneDataModuleBase):
    """
    Custom DataModule for Matgl FineTune task.
    """

    def __init__(
        self,
        batch_size: int,
        converter: GraphConverter,
        structures: list[Structure] = [],
        labels: dict[str, list] = {},
        test_structures: list[Structure] = [],
        test_labels: dict[str, list] = {},
        threebody_cutoff: float = 4.0,
        include_line_graph: bool = False,
        directed_line_graph: bool = False,
        graph_labels: list[int | float] | None = None,
        num_workers: int = 0,
        val_split: float = 0.1,
        test_split: float = 0.1,
        shuffle: bool = True,
        ignore_data_errors: bool = True,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            val_split=val_split,
            test_split=test_split,
            shuffle=shuffle,
            ignore_data_errors=ignore_data_errors,
        )
        self.threebody_cutoff = threebody_cutoff
        self.converter = converter
        self.include_line_graph = include_line_graph
        self.directed_line_graph = directed_line_graph
        self.graph_labels = graph_labels

        if len(test_structures) == 0:
            ## Split data into train, val, and test
            total_size = len(structures)
            indices = list(range(total_size))
            if self.shuffle:
                random.shuffle(indices)

            train_end = int(total_size * (1 - self.val_split - self.test_split))
            val_end = int(total_size * (1 - self.test_split))

            self.train_structures = [structures[i] for i in indices[:train_end]]
            self.val_structures = [structures[i] for i in indices[train_end:val_end]]
            self.test_structures = [structures[i] for i in indices[val_end:]]

            self.train_labels = {
                k: [v[i] for i in indices[:train_end]] for k, v in labels.items()
            }
            self.val_labels = {
                k: [v[i] for i in indices[train_end:val_end]] for k, v in labels.items()
            }
            self.test_labels = {
                k: [v[i] for i in indices[val_end:]] for k, v in labels.items()
            }

        else:
            total_size = len(structures)
            indices = list(range(total_size))
            if self.shuffle:
                random.shuffle(indices)

            train_end = int(total_size * (1 - self.val_split))

            self.train_structures = [structures[i] for i in indices[:train_end]]
            self.val_structures = [structures[i] for i in indices[train_end:]]
            self.test_structures = test_structures

            self.train_labels = {
                k: [v[i] for i in indices[:train_end]] for k, v in labels.items()
            }
            self.val_labels = {
                k: [v[i] for i in indices[train_end:]] for k, v in labels.items()
            }
            self.test_labels = test_labels

    @override
    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = MatglDataset(
            structures=self.train_structures,
            labels=self.train_labels,
            threebody_cutoff=self.threebody_cutoff,
            converter=self.converter,
            include_line_graph=self.include_line_graph,
            directed_line_graph=self.directed_line_graph,
            graph_labels=self.graph_labels,
        )
        self.val_dataset = MatglDataset(
            structures=self.val_structures,
            labels=self.val_labels,
            threebody_cutoff=self.threebody_cutoff,
            converter=self.converter,
            include_line_graph=self.include_line_graph,
            directed_line_graph=self.directed_line_graph,
            graph_labels=self.graph_labels,
        )
        self.test_dataset = MatglDataset(
            structures=self.test_structures,
            labels=self.test_labels,
            threebody_cutoff=self.threebody_cutoff,
            converter=self.converter,
            include_line_graph=self.include_line_graph,
            directed_line_graph=self.directed_line_graph,
            graph_labels=self.graph_labels,
        )

    @override
    def collate_fn(self, data_list: list[DataTypedDict]) -> BatchTypedDict:
        atomic_numbers = torch.cat([data["atomic_numbers"] for data in data_list])
        pos = torch.cat([data["pos"] for data in data_list])
        num_atoms = torch.cat([data["num_atoms"] for data in data_list])
        cell_displacement = torch.cat([data["cell_displacement"] for data in data_list])
        cell = torch.cat([data["cell"] for data in data_list])
        graph = dgl.batch([data["graph"] for data in data_list])
        if self.include_line_graph:
            line_graph = dgl.batch([data["line_graph"] for data in data_list])
        else:
            line_graph = None
        lattice = torch.cat([data["lattice"] for data in data_list])
        state_attr = torch.cat([data["state_attr"] for data in data_list])
        band_gap = torch.cat([data["band_gap"] for data in data_list])
        batch_idx = torch.cat(
            [
                torch.full((len(data["atomic_numbers"]),), i, dtype=matgl.int_th)
                for i, data in enumerate(data_list)
            ]
        )
        batch_idx = batch_idx.to(torch.int64)
        batch = BatchTypedDict(
            batch=batch_idx,
            atomic_numbers=atomic_numbers,
            pos=pos,
            num_atoms=num_atoms,
            cell_displacement=cell_displacement,
            cell=cell,
            graph=graph,
            line_graph=line_graph,
            lattice=lattice,
            state_attr=state_attr,
            band_gap=band_gap,
        )
        return batch

    @override
    def train_dataloader(self):
        return self._create_dataloader(self.train_dataset, shuffle=self.shuffle)

    @override
    def val_dataloader(self):
        return self._create_dataloader(self.val_dataset, shuffle=False)

    @override
    def test_dataloader(self):
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def get_predict_dataloader(self, structures: list[Structure]):
        dataset = MatglDataset(
            structures=structures,
            labels={},
            threebody_cutoff=self.threebody_cutoff,
            converter=self.converter,
            include_line_graph=self.include_line_graph,
            directed_line_graph=self.directed_line_graph,
            graph_labels=self.graph_labels,
        )
        return self._create_dataloader(dataset, shuffle=False)
