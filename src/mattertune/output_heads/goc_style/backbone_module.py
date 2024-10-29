from __future__ import annotations

from typing import TypedDict

import jaxtyping as jt
import torch


class GOCStyleBackBoneOutput(TypedDict):
    edeg_index_src: jt.Int[torch.Tensor, "num_edges_in_batch"]
    edge_index_dst: jt.Int[torch.Tensor, "num_edges_in_batch"]
    edge_vectors: jt.Float[torch.Tensor, "num_edges_in_batch 3"]
    edge_lengths: jt.Float[torch.Tensor, "num_edges_in_batch"]
    node_hidden_features: jt.Float[torch.Tensor, "num_nodes_in_batch node_hidden_dim"]
    edge_hidden_features: jt.Float[torch.Tensor, "num_edges_in_batch edge_hidden_dim"]
    energy_features: jt.Float[torch.Tensor, "num_nodes_in_batch energy_feature_dim"]
    force_features: jt.Float[torch.Tensor, "num_edges_in_batch force_feature_dim"]
    ## TODO: global features
