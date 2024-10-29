from __future__ import annotations

import torch
import torch.nn as nn
from e3nn import o3
from einops import rearrange
from torch_scatter import scatter
from typing_extensions import override


class Rank2DecompositionEdgeBlock(nn.Module):
    r"""Prediction of rank 2 tensor
    Decompose rank 2 tensor with irreps
    since it is symmetric we need just irrep degree 0 and 2
    Parameters
    ----------
    emb_size : int
        size of edge embedding used to compute outer products
    num_layers : int
        number of layers of the MLP
    --------
    """

    change_mat: torch.Tensor  ## size must be [9, 9]

    def __init__(
        self,
        emb_size,
        edge_level,
        extensive=False,
        num_layers=2,
    ):
        super().__init__()
        assert self.change_mat.shape == (
            9,
            9,
        ), f"{self.change_mat.shape=} must be {(9, 9)}, found {self.change_mat.shape}"
        self.emb_size = emb_size
        self.edge_level = edge_level
        self.extensive = extensive
        self.scalar_nonlinearity = nn.SiLU()
        self.scalar_MLP = nn.ModuleList()
        self.irrep2_MLP = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.scalar_MLP.append(nn.Linear(emb_size, emb_size))
                self.irrep2_MLP.append(nn.Linear(emb_size, emb_size))
                self.scalar_MLP.append(self.scalar_nonlinearity)
                self.irrep2_MLP.append(self.scalar_nonlinearity)
            else:
                self.scalar_MLP.append(nn.Linear(emb_size, 1))
                self.irrep2_MLP.append(nn.Linear(emb_size, 1))

        # Change of basis obtained by stacking the C-G coefficients in the right way

        self.register_buffer(
            "change_mat",
            torch.transpose(
                torch.tensor(
                    [
                        [3 ** (-0.5), 0, 0, 0, 3 ** (-0.5), 0, 0, 0, 3 ** (-0.5)],
                        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0],
                        [0, 0, -(2 ** (-0.5)), 0, 0, 0, 2 ** (-0.5), 0, 0],
                        [0, 2 ** (-0.5), 0, -(2 ** (-0.5)), 0, 0, 0, 0, 0],
                        [0, 0, 0.5**0.5, 0, 0, 0, 0.5**0.5, 0, 0],
                        [0, 2 ** (-0.5), 0, 2 ** (-0.5), 0, 0, 0, 0, 0],
                        [
                            -(6 ** (-0.5)),
                            0,
                            0,
                            0,
                            2 * 6 ** (-0.5),
                            0,
                            0,
                            0,
                            -(6 ** (-0.5)),
                        ],
                        [0, 0, 0, 0, 0, 2 ** (-0.5), 0, 2 ** (-0.5), 0],
                        [-(2 ** (-0.5)), 0, 0, 0, 0, 0, 0, 0, 2 ** (-0.5)],
                    ]
                ).detach(),
                0,
                1,
            ),
            persistent=False,
        )

    @override
    def forward(
        self,
        edge_features: torch.Tensor,  ## [num_edges, edge_hidden_dim]
        edge_vectors: torch.Tensor,  ## [num_edges, 3]
        edge_index_dst: torch.Tensor,  ## [num_edges]
        batch_idx: torch.Tensor,  ## [num_edges]
        batch_size: int,
    ) -> torch.Tensor:
        """evaluate
        Parameters
        ----------
        x_edge : `torch.Tensor [num_edges, emb_size]`
            edge features
        edge_vec : `torch.Tensor [num_edges, 3]`
            euclidean vectors of the edges
        edge_index_dst : `torch.Tensor [num_edges]`
            destination node index of the edges
        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(..., 3, 3)``
        """
        # Calculate spherical harmonics of degree 2 of the points sampled
        sphere_irrep2 = o3.spherical_harmonics(
            2, edge_vectors, True
        ).detach()  # (nEdges, 5)

        if self.edge_level:
            # Irrep 0 prediction
            edge_scalar = edge_features
            for i, module in enumerate(self.scalar_MLP):
                edge_scalar = module(edge_scalar)

            # Irrep 2 prediction
            edge_irrep2 = edge_features  # (nEdges, 5, emb_size)
            for i, module in enumerate(self.irrep2_MLP):
                edge_irrep2 = module(edge_irrep2)
            edge_irrep2 = sphere_irrep2[:, :, None] * edge_irrep2[:, None, :]

            node_scalar = scatter(
                edge_scalar,
                edge_index_dst,
                dim=0,
                dim_size=batch_idx.shape[0],
                reduce="mean",
            )
            node_irrep2 = scatter(
                edge_irrep2,
                edge_index_dst,
                dim=0,
                dim_size=batch_idx.shape[0],
                reduce="mean",
            )
        else:
            raise NotImplementedError
            edge_irrep2 = (
                sphere_irrep2[:, :, None] * x_edge[:, None, :]
            )  # (nAtoms, 5, emb_size)

            node_scalar = scatter(x_edge, idx_t, dim=0, reduce="mean")
            node_irrep2 = scatter(edge_irrep2, idx_t, dim=0, reduce="mean")

            # Irrep 0 prediction
            for i, module in enumerate(self.scalar_MLP):
                if i == 0:
                    node_scalar = module(node_scalar)
                else:
                    node_scalar = module(node_scalar)

            # Irrep 2 prediction
            for i, module in enumerate(self.irrep2_MLP):
                if i == 0:
                    node_irrep2 = module(node_irrep2)
                else:
                    node_irrep2 = module(node_irrep2)

        if self.extensive:
            scalar = scatter(
                node_scalar.view(-1),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="sum",
            )
            irrep2 = scatter(
                node_irrep2.view(-1, 5),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="sum",
            )
        else:
            irrep2 = scatter(
                node_irrep2.view(-1, 5),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="mean",
            )
            scalar = scatter(
                node_scalar.view(-1),
                batch_idx,
                dim=0,
                dim_size=batch_size,
                reduce="mean",
            )

        # Change of basis to compute a rank 2 symmetric tensor

        vector = torch.zeros((batch_size, 3), device=scalar.device).detach()
        flatten_irreps = torch.cat([scalar.reshape(-1, 1), vector, irrep2], dim=1)
        stress = torch.einsum(
            "ab, cb->ca", self.change_mat.to(flatten_irreps.device), flatten_irreps
        )
        assert stress.shape == (
            batch_size,
            9,
        ), f"{stress.shape=} must be {(batch_size, 9)}, found {stress.shape}"
        assert torch.is_floating_point(
            stress
        ), f"{stress.dtype=} must be float, found {stress.dtype}"

        stress = rearrange(
            stress,
            "b (three1 three2) -> b three1 three2",
            three1=3,
            three2=3,
        )
        assert stress.shape == (
            batch_size,
            3,
            3,
        ), f"{stress.shape=} must be {(batch_size, 3, 3)}, found {stress.shape}"
        assert torch.allclose(
            stress, stress.permute(0, 2, 1)
        ), f"{stress=} must be symmetric, found not symmetric"

        return stress
