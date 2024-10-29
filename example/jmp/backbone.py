from __future__ import annotations

from functools import partial

import torch.nn as nn
from jmppeft.models.gemnet.backbone import GOCBackboneOutput
from jmppeft.utils.goc_graph import (
    Cutoffs,
    Graph,
    MaxNeighbors,
    generate_graph,
    subselect_graph,
    tag_mask,
)
from load_pretrain import load_pretrain
from torch_geometric.data.data import BaseData
from torch_geometric.utils import dropout_edge
from typing_extensions import cast, override

from mattertune.output_heads.goc_style.backbone_module import GOCStyleBackBoneOutput
from mattertune.protocol import BackBoneBaseConfig, BackBoneBaseModule


class JMPBackbone(BackBoneBaseModule, nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        backbone: nn.Module,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        qint_tags: list,
        edge_dropout: float | None = None,
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

    @classmethod
    @override
    def load_backbone(
        cls,
        path: str,
        type: str,
        cutoffs: Cutoffs,
        max_neighbors: MaxNeighbors,
        qint_tags: list,
        edge_dropout: float | None = None,
        per_graph_radius_graph: bool = False,
    ):
        backbone, embedding = load_pretrain(
            model_type=type,
            ckpt_path=path,
        )
        return JMPBackbone(
            embedding,
            backbone,
            cutoffs,
            max_neighbors,
            qint_tags,
            edge_dropout,
            per_graph_radius_graph,
        )

    @override
    def forward(self, batch: BaseData) -> GOCStyleBackBoneOutput:
        atomic_numbers = batch.atomic_numbers.long() - 1
        h = self.embedding(atomic_numbers)
        out = cast(GOCBackboneOutput, self.backbone(batch, h=h))

        output = {
            "edge_index_src": out["idx_s"],
            "edge_index_dst": out["idx_t"],
            "edge_vectors": out["V_st"],
            "edge_lengths": out["D_st"],
            "node_hidden_features": out["h"],
            "edge_hidden_features": out["forces"],
            "energy_features": out["energy"],
            "force_features": out["forces"],
        }
        return cast(GOCStyleBackBoneOutput, output)

    @override
    def process_batch_under_grad(self, batch: BaseData, training: bool) -> BaseData:
        aint_graph = generate_graph(
            batch,
            cutoff=self.cutoffs.aint,
            max_neighbors=self.max_neighbors.aint,
            pbc=self.pbc,
            per_graph=self.per_graph_radius_graph,
        )
        aint_graph = self.process_aint_graph(aint_graph, training=training)
        subselect = partial(
            subselect_graph,
            batch,
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
        qint_graph = tag_mask(batch, qint_graph, tags=self.qint_tags)

        graphs = {
            "main": main_graph,
            "a2a": aint_graph,
            "a2ee2a": aeaint_graph,
            "qint": qint_graph,
        }

        for graph_type, graph in graphs.items():
            for key, value in graph.items():
                setattr(batch, f"{graph_type}_{key}", value)

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


class JMPBackboneConfig(BackBoneBaseConfig):
    freeze: bool = False
    ckpt_path: str
    type: str
    cutoffs: Cutoffs
    max_neighbors: MaxNeighbors
    qint_tags: list
    edge_dropout: float | None = None
    per_graph_radius_graph: bool = (False,)

    @override
    def construct_backbone(self) -> JMPBackbone:
        return JMPBackbone.load_backbone(
            self.ckpt_path,
            self.type,
            self.cutoffs,
            self.max_neighbors,
            self.qint_tags,
            self.edge_dropout,
            self.per_graph_radius_graph,
        )
