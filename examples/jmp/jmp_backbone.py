import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
from typing_extensions import override, cast
from ase import Atoms
import numpy as np
from jmppeft.models.gemnet.backbone import GOCBackboneOutput
from mattertune.data_structures.data import TMatterTuneBatch
from mattertune.output_heads.goc_style.backbone_output import GOCStyleBackBoneOutput
from mattertune.finetune.backbone import BackBoneBaseModule, BackBoneBaseConfig
from load_pretrain import load_pretrain
from jmppeft.utils.goc_graph import generate_graph, MaxNeighbors, Cutoffs, subselect_graph, tag_mask, Graph
from torch_geometric.utils import dropout_edge
from functools import partial
from dataclasses import dataclass


@dataclass
class JMPData():
    r"""
    Data class for JMP, which follows the MatterTuneDataProtocol.
    """
    idx: torch.Tensor
    backbone_input: BaseData
    labels: dict[str, torch.Tensor]
    
    @property
    def num_atoms(self) -> torch.Tensor:
        return self.backbone_input.num_atoms
    
    @property
    def atomic_numbers(self) -> torch.Tensor:
        return self.backbone_input.atomic_numbers
    
    @property
    def positions(self) -> torch.Tensor:
        return self.backbone_input.pos
    
    @positions.setter
    def positions(self, value: torch.Tensor):
        self.backbone_input.pos = value
    
    @property
    def cell(self) -> torch.Tensor:
        return self.backbone_input.cell

    @cell.setter
    def cell(self, value: torch.Tensor):
        self.backbone_input.cell = value
    
    @property
    def strain(self) -> torch.Tensor:
        return self.backbone_input.strain

    @strain.setter
    def strain(self, value: torch.Tensor):
        self.backbone_input.strain = value
        
@dataclass
class JMPBatch():
    r"""
    Batch class for JMP, which follows the MatterTuneBatchProtocol.
    """
    idx: torch.Tensor
    backbone_input: Batch
    labels: dict[str, torch.Tensor]
    
    @property
    def batch(self) -> torch.Tensor:
        return self.backbone_input.batch
    
    @property
    def num_atoms(self) -> torch.Tensor:
        return self.backbone_input.num_atoms
    
    @property
    def atomic_numbers(self) -> torch.Tensor:
        return self.backbone_input.atomic_numbers
    
    @property
    def positions(self) -> torch.Tensor:
        return self.backbone_input.pos
    
    @positions.setter
    def positions(self, value: torch.Tensor):
        self.backbone_input.pos = value
    
    @property
    def cell(self) -> torch.Tensor:
        return self.backbone_input.cell

    @cell.setter
    def cell(self, value: torch.Tensor):
        self.backbone_input.cell = value
    
    @property
    def strain(self) -> torch.Tensor:
        return self.backbone_input.strain

    @strain.setter
    def strain(self, value: torch.Tensor):
        self.backbone_input.strain = value
    


class JMPBackbone(BackBoneBaseModule, nn.Module):
    r"""
    Wrapper for the backbone model of JMP
    Paper: https://openreview.net/forum?id=PfPnugdxup
    """
    def __init__(
        self,
        embedding: nn.Module,
        backbone: nn.Module,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        qint_tags: list,
        edge_dropout: float|None = None,
        pbc: bool = True,
        per_graph_radius_graph: bool = False,
    ):
        nn.Module.__init__(self)
        super().__init__()
        self.embedding = embedding
        self.backbone = backbone
        self.cutoffs = cutoffs
        self.max_neighbors = max_neighbors
        self.qint_tags = qint_tags
        self.edge_dropout = edge_dropout
        self.pbc = pbc
        self.per_graph_radius_graph = per_graph_radius_graph
        
    @override
    def forward(self, batch: JMPBatch) -> GOCStyleBackBoneOutput:
        
        backbone_input = batch.backbone_input
        atomic_numbers = backbone_input.atomic_numbers.long() - 1
        h = self.embedding(atomic_numbers)
        out = cast(GOCBackboneOutput, self.backbone(backbone_input, h=h))
        
        output = {
            "edge_index_src": out["idx_s"],
            "edge_index_dst": out["idx_t"],
            "edge_vectors": out["V_st"],
            "edge_lengths": out["D_st"],
            "node_hidden_features": out["energy"],
            "edge_hidden_features": out["forces"],
            "energy_features": out["energy"],
            "force_features": out["forces"],
        }
        return cast(GOCStyleBackBoneOutput, output)
    
    @override
    def process_raw(self, atoms: Atoms, idx: int, labels: dict[str, torch.Tensor], inference: bool) -> JMPData:
        data_dict = {
            "atomic_numbers": torch.tensor(atoms.get_atomic_numbers()).long(),
            "pos": torch.tensor(atoms.get_positions()).float(),
            "num_atoms": torch.tensor([len(atoms)]).long(),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=torch.float).unsqueeze(dim=0),
            "strain": torch.zeros_like(torch.tensor(np.array(atoms.get_cell())), dtype=torch.float),
            "natoms": torch.tensor(len(atoms)).long(),
            "tags": 2 * torch.ones(len(atoms), dtype=torch.long),
            "fixed": torch.zeros_like(torch.tensor(np.array(atoms.get_atomic_numbers())), dtype=torch.bool),
        }
        if "stress" in labels:
            labels["stress"] = labels["stress"].reshape(1, 3, 3)
        data = Data.from_dict(data_dict)
        return JMPData(torch.tensor([idx], dtype=torch.long), data, labels)

    @override
    def process_batch_under_grad(self, batch: JMPBatch, training: bool) -> JMPBatch:
        
        backbone_input = batch.backbone_input
        aint_graph = generate_graph(
            backbone_input,
            cutoff=self.cutoffs.aint,
            max_neighbors=self.max_neighbors.aint,
            pbc=self.pbc,
            per_graph=self.per_graph_radius_graph,
        )
        aint_graph = self.process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            backbone_input,
            aint_graph,
            cutoff_orig=self.cutoffs.aint,
            max_neighbors_orig=self.max_neighbors.aint,
        )
        main_graph = subselect(self.cutoffs.main, self.max_neighbors.main)
        aeaint_graph = subselect(self.cutoffs.aeaint, self.max_neighbors.aeaint)
        qint_graph = subselect(self.cutoffs.qint, self.max_neighbors.qint)

        # We can't do this at the data level: This is because the batch collate_fn doesn't know
        # that it needs to increment the "id_swap" indices as it collates the data.
        # So we do this at the graph level (which is done in the GemNetOC `get_graphs_and_indices` method).
        # main_graph = symmetrize_edges(main_graph, num_atoms=data.pos.shape[0])
        qint_graph = tag_mask(backbone_input, qint_graph, tags=self.qint_tags)

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(backbone_input, f"{graph_type}_{key}", value)

        batch.backbone_input = backbone_input
        return batch
    
    def process_aint_graph(self, graph: Graph, *, training: bool):
        if self.edge_dropout:
            graph["edge_index"], mask = dropout_edge(
                graph["edge_index"],
                p=self.edge_dropout,
                training=training,
            )
            graph["distance"] = graph["distance"][mask]
            graph["vector"] = graph["vector"][mask]
            graph["cell_offset"] = graph["cell_offset"][mask]

            if "id_swap_edge_index" in graph:
                graph["id_swap_edge_index"] = graph["id_swap_edge_index"][mask]

        return graph
    
    @classmethod
    @override
    def collate_fn(
        cls,
        data_list: list[JMPData],
    ) -> JMPBatch:
        idx = torch.cat([d.idx for d in data_list], dim=0)
        data = Batch.from_data_list([d.backbone_input for d in data_list])
        labels = {key: torch.cat([d.labels[key] for d in data_list], dim=0) for key in data_list[0].labels}
        return JMPBatch(idx, data, labels)
        
    @classmethod
    @override
    def load_backbone(
        cls,
        path: str,
        type: str,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        qint_tags: list,
        edge_dropout: float|None = None,
        per_graph_radius_graph: bool = False,
    ):
        backbone, embedding = load_pretrain(
            model_type=type,
            ckpt_path=path,
        )
        return JMPBackbone(
            embedding = embedding,
            backbone = backbone,
            cutoffs = cutoffs,
            max_neighbors = max_neighbors,
            qint_tags = qint_tags,
            edge_dropout = edge_dropout,
            per_graph_radius_graph = per_graph_radius_graph,
        )

class JMPBackboneConfig(BackBoneBaseConfig):
    freeze: bool = False
    ckpt_path: str
    type: str
    cutoffs: Cutoffs
    max_neighbors: MaxNeighbors
    qint_tags: list
    edge_dropout: float|None = None
    per_graph_radius_graph: bool = False,
    
    @override
    def construct_backbone(self) -> JMPBackbone:
        return JMPBackbone.load_backbone(
            self.ckpt_path, 
            self.type, 
            self.cutoffs, 
            self.max_neighbors, 
            self.qint_tags, 
            self.edge_dropout, 
            self.per_graph_radius_graph
        )