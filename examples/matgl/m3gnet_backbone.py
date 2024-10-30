import torch
import torch.nn as nn
import dgl
from typing_extensions import override
from pathlib import Path
import matgl
from matgl.models import M3GNet
from matgl.utils.io import _get_file_paths
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.compute import compute_pair_vector_and_distance, create_line_graph
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np
from mattertune.data_structures.data import TMatterTuneBatch
from mattertune.output_heads.goc_style.backbone_output import GOCStyleBackBoneOutput
from mattertune.finetune.backbone import BackBoneBaseModule, BackBoneBaseConfig
from functools import partial
from dataclasses import dataclass

@dataclass
class BackboneInput:
    g: dgl.DGLGraph
    l_g: dgl.DGLGraph|None
    lattice: torch.Tensor
    state_attr: torch.Tensor
    atomic_numbers: torch.Tensor
    strain: torch.Tensor

@dataclass
class MatGLData():
    r"""
    Data class for MatGL, which follows the MatterTuneDataProtocol.
    """
    idx: torch.Tensor
    backbone_input: BackboneInput
    labels: dict[str, torch.Tensor]
    
    @property
    def num_atoms(self) -> torch.Tensor:
        return self.backbone_input.g.batch_num_nodes()
    
    @property
    def atomic_numbers(self) -> torch.Tensor:
        return self.backbone_input.atomic_numbers
    
    @property
    def positions(self) -> torch.Tensor:
        return self.backbone_input.g.ndata["pos"]
    
    @positions.setter
    def positions(self, value: torch.Tensor):
        self.backbone_input.g.ndata["pos"] = value
    
    @property
    def cell(self) -> torch.Tensor:
        return self.backbone_input.lattice

    @cell.setter
    def cell(self, value: torch.Tensor):
        self.backbone_input.lattice = value
    
    @property
    def strain(self) -> torch.Tensor:
        return self.backbone_input.strain

    @strain.setter
    def strain(self, value: torch.Tensor):
        self.backbone_input.strain = value
        
@dataclass
class MatGLBatch():
    r"""
    Data class for MatGL, which follows the MatterTuneBatchProtocol.
    """
    idx: torch.Tensor
    backbone_input: BackboneInput
    labels: dict[str, torch.Tensor]
    
    @property
    def batch(self) -> torch.Tensor:
        num_nodes = self.backbone_input.g.batch_num_nodes()
        return torch.repeat_interleave(torch.arange(len(num_nodes)), num_nodes)
    
    @property
    def num_atoms(self) -> torch.Tensor:
        return self.backbone_input.g.batch_num_nodes()
    
    @property
    def atomic_numbers(self) -> torch.Tensor:
        return self.backbone_input.atomic_numbers
    
    @property
    def positions(self) -> torch.Tensor:
        return self.backbone_input.g.ndata["pos"]
    
    @positions.setter
    def positions(self, value: torch.Tensor):
        self.backbone_input.g.ndata["pos"] = value
    
    @property
    def cell(self) -> torch.Tensor:
        return self.backbone_input.lattice

    @cell.setter
    def cell(self, value: torch.Tensor):
        self.backbone_input.lattice = value
    
    @property
    def strain(self) -> torch.Tensor:
        return self.backbone_input.strain

    @strain.setter
    def strain(self, value: torch.Tensor):
        self.backbone_input.strain = value
        
        
class M3GNetBackbone(BackBoneBaseModule, nn.Module):
    """
    Wrapper for the M3GNet model.
    Original Paper: https://www.nature.com/articles/s43588-022-00349-3
    Used MatGL implementation: https://github.com/materialsvirtuallab/matgl
    """
    def __init__(
        self,
        model: M3GNet,
        converter: Structure2Graph,
        pre_compute_line_graph: bool = False,
        graph_labels: list[int|float] | None = None
    ):
        super().__init__()
        self.model = model
        self.converter = converter
        self.adaptor = AseAtomsAdaptor()
        self.pre_compute_line_graph = pre_compute_line_graph
        self.graph_labels = graph_labels
    
    @override
    def forward(
        self,
        batch: MatGLBatch,
    ) -> GOCStyleBackBoneOutput:
        g, l_g, state_attr = batch.backbone_input.g, batch.backbone_input.l_g, batch.backbone_input.state_attr
        _ = self.model(g, l_g, state_attr)
        output = GOCStyleBackBoneOutput(
            edeg_index_src=g.edges()[0],
            edge_index_dst=g.edges()[1],
            edge_vectors=g.edata["bond_vec"],
            edge_lengths=g.edata["bond_dist"],
            node_hidden_features=g.ndata["node_feat"],
            edge_hidden_features=g.edata["edge_feat"],
            energy_features=g.ndata["node_feat"],
            force_features=g.edata["edge_feat"],
        )
        return output
    
    @override
    def process_raw(self, atoms: Atoms, idx: int, labels: dict[str, torch.Tensor], inference: bool) -> MatGLData:
        atomic_numbers = torch.tensor(atoms.get_atomic_numbers()).long()
        structure = self.adaptor.get_structure(atoms)
        graph, lattice, state_attr = self.converter.get_graph(structure)
        graph.ndata["pos"] = torch.tensor(structure.cart_coords)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
        bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_vec"] = bond_vec
        graph.edata["bond_dist"] = bond_dist
        graph.ndata.pop("pos")
        graph.edata.pop("pbc_offshift")
        if self.pre_compute_line_graph:
            line_graph = create_line_graph(graph, self.threebody_cutoff, directed=self.directed_line_graph)  # type: ignore
            for name in ["bond_vec", "bond_dist", "pbc_offset"]:
                line_graph.ndata.pop(name)
        else:
            line_graph = None
        if self.graph_labels is not None:
            state_attr = torch.tensor([self.graph_labels], dtype=matgl.float_th)
        else:
            state_attr = torch.tensor(np.array(state_attr), dtype=matgl.float_th)
            
        if "stress" in labels:
            labels["stress"] = labels["stress"].reshape(1, 3, 3)
            
        return MatGLData(
            idx = torch.tensor([idx], dtype=torch.long),
            backbone_input = BackboneInput(
                g = graph,
                l_g = line_graph,
                lattice = lattice,
                state_attr = state_attr,
                atomic_numbers = atomic_numbers,
                strain = torch.zeros_like(lattice)
            ),
            labels = labels
        )
        
    @classmethod
    @override
    def collate_fn(
        cls,
        data_list: list[MatGLData],
    ) -> MatGLBatch:
        r"""
        Referred to the implementation of collate_fn_pes in MatGL.
        """
        idx = torch.cat([d.idx for d in data_list], dim=0)
        g_list = [d.backbone_input.g for d in data_list]
        l_g_list = [d.backbone_input.l_g for d in data_list]
        g = dgl.batch(g_list)
        l_g = dgl.batch(l_g_list) if l_g_list[0] is not None else None
        lattice = torch.stack([d.backbone_input.lattice for d in data_list])
        state_attr = torch.stack([d.backbone_input.state_attr for d in data_list])
        atomic_numbers = torch.cat([d.backbone_input.atomic_numbers for d in data_list], dim=0)
        strain = torch.stack([d.backbone_input.strain for d in data_list])
        labels = {key: torch.cat([d.labels[key] for d in data_list], dim=0) for key in data_list[0].labels}
        return MatGLBatch(
            idx = idx,
            backbone_input = BackboneInput(
                g = g,
                l_g = l_g,
                lattice = lattice,
                state_attr = state_attr,
                atomic_numbers = atomic_numbers,
                strain = strain
            ),
            labels = labels
        )
        
    @override
    def process_batch_under_grad(self, batch: MatGLBatch, training: bool) -> MatGLBatch:
        strain = batch.backbone_input.strain
        batch.backbone_input.lattice = batch.backbone_input.lattice @ (torch.eye(3, device=lattice.device) + strain)
        batch.backbone_input.g.edata["lattice"] = torch.repeat_interleave(lattice, batch.backbone_input.g.batch_num_edges(), dim=0)
        batch.backbone_input.g.edata["pbc_offshift"] = torch.matmul(batch.backbone_input.g.edata["pbc_offset"], lattice[0])
        batch.backbone_input.g.ndata["pos"] = (
            batch.backbone_input.g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, batch.backbone_input.g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        return batch
    
    @classmethod
    @override
    def load_backbone(
        cls,
        path: Path,
        element_types,
        cutoff: float,
        pre_compute_line_graph: bool = False,
        graph_labels: list[int|float] | None = None,
        **kwargs,
    ):
        path = Path(path)
        fpaths = _get_file_paths(path, **kwargs)
        model = M3GNet.load(fpaths, **kwargs)
        model.final_layer.requires_grad_(False)
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        return cls(model, converter, pre_compute_line_graph, graph_labels)