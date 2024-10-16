import torch
import torch.nn as nn
from pathlib import Path
from matgl.models import M3GNet
from matgl.graph.compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.io import _get_file_paths
from typing_extensions import override
import dgl
from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput
from mattertune.protocol import BackBoneBaseConfig, BackBoneBaseModule
from bandgap_data_module import BatchTypedDict


class M3GNetBackbone(BackBoneBaseModule, M3GNet):
    @classmethod
    @override
    def load_backbone(
        cls,
        path: Path,
        **kwargs,
    ):
        path = Path(path)
        fpaths = _get_file_paths(path, **kwargs)
        return cls.load(fpaths, **kwargs)
        
    
    @override
    def forward(
        self,
        batch: BatchTypedDict,
    ) -> GOCStyleBackBoneOutput:
        """
        Override the forward method of M3GNet to return the backbone output with specific style.
        """
        g = batch["graph"]
        state_attr = batch["state_attr"]
        l_g = batch["line_graph"]
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            l_g = ensure_line_graph_compatibility(g, l_g, self.threebody_cutoff)
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = self.basis_expansion(l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)
        fea_dict = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            },
        }
        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                g,
                l_g,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            fea_dict[f"gc_{i+1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }
        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat
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
    def process_batch_under_grad(self, batch: BatchTypedDict, training: bool) -> BatchTypedDict:
        g = batch["graph"]
        lat = batch["lattice"]
        state_attr = batch["state_attr"]
        l_g = batch["line_graph"]
        
        strain = batch["cell_displacement"]
        lattice = lat @ (torch.eye(3, device=lat.device) + strain)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if batch["pos"].requires_grad:
            g.ndata["pos"] = batch["pos"]
        batch["graph"] = g
        return batch
        

class M3GNetBackboneConfig(BackBoneBaseConfig):
    freeze: bool = False
    path: Path
    @override
    def construct_backbone(self) -> M3GNetBackbone:
        return M3GNetBackbone.load_backbone(self.path)