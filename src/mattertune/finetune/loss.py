from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Annotated, Literal, TypeAlias
import numpy as np
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torchtyping import TensorType
from typing_extensions import assert_never, override
from pydantic import Field, BaseModel
from typing import Generic
from mattertune.protocol import TBatch


Reduction: TypeAlias = Literal["mean", "sum", "none"]

class LossConfigBase(BaseModel, ABC, Generic[TBatch]):
    y_mult_coeff: float = 1.0
    
    def pre_compute(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ):
        y_pred = y_pred * self.y_mult_coeff
        y_true = y_true * self.y_mult_coeff
        return y_pred, y_true
    
    def compute(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        y_pred, y_true = self.pre_compute(y_pred, y_true)
        return self.compute_impl(y_pred, y_true, batch, reduction)
    
    @abstractmethod
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor: ...

    def reduce(
        self,
        loss: torch.Tensor,
        reduction: Reduction,
    ) -> torch.Tensor:
        match reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                assert_never(reduction)

class MAELossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mae"] = "mae"

    divide_by_natoms: bool = False
    """Whether to divide the target/pred  by the number of atoms."""

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
        natoms: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.divide_by_natoms:
            assert natoms is not None, "natoms is required"
            y_pred = y_pred / natoms
            y_true = y_true / natoms

        loss = F.l1_loss(y_pred, y_true, reduction="none")
        loss = self.reduce(loss, reduction)
        return loss


class MSELossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mse"] = "mse"

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        loss = F.mse_loss(y_pred, y_true, reduction="none")
        loss = self.reduce(loss, reduction)
        return loss


class HuberLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["huber"] = "huber"

    delta: float

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        loss = F.huber_loss(y_pred, y_true, delta=self.delta, reduction="none")
        loss = self.reduce(loss, reduction)
        return loss


class MACEHuberLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mace_huber"] = "mace_huber"

    delta: float

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        # Define the multiplication factors for each condition
        factors = self.delta * np.array([1.0, 0.7, 0.4, 0.1])

        # Apply multiplication factors based on conditions
        c1 = torch.norm(y_true, dim=-1) < 100
        c2 = (torch.norm(y_true, dim=-1) >= 100) & (torch.norm(y_true, dim=-1) < 200)
        c3 = (torch.norm(y_true, dim=-1) >= 200) & (torch.norm(y_true, dim=-1) < 300)
        c4 = ~(c1 | c2 | c3)

        se = torch.zeros_like(y_pred)

        se[c1] = F.huber_loss(
            y_true[c1], y_pred[c1], reduction="none", delta=factors[0]
        )
        se[c2] = F.huber_loss(
            y_true[c2], y_pred[c2], reduction="none", delta=factors[1]
        )
        se[c3] = F.huber_loss(
            y_true[c3], y_pred[c3], reduction="none", delta=factors[2]
        )
        se[c4] = F.huber_loss(
            y_true[c4], y_pred[c4], reduction="none", delta=factors[3]
        )

        # Reduce the loss
        loss = self.reduce(se, reduction)
        return loss


def _apply_focal_loss_per_atom(
    *,
    loss: torch.Tensor, # A float tensor with "natoms" dimension
    freq_ratios: torch.Tensor, # A float tensor with "atom_type" dimension
    atomic_numbers: torch.Tensor, # An int tensor with "natoms" (number of atoms) dimension
    gamma: float,  # A standard float value (not a tensor)
):
    # Apply the frequency factor to each sample & compute the mean frequency factor for each graph
    freq_factor = freq_ratios[atomic_numbers]
    assert freq_factor.shape == (loss.shape[0],), f"freq_factor should be in shape ({loss.shape[0]},), but here gets {freq_factor.shape}"
    assert torch.is_floating_point(freq_factor), f"freq_factor should be in float but here gets {freq_factor.dtype}"

    # Compute difficulty factor
    # We'll use the Huber loss value instead of MAE for the difficulty
    if loss.max() == 0:
        raise ValueError("loss.max() is 0")
    normalized_loss = loss / loss.max()
    difficulty = (1 - torch.exp(-normalized_loss)) ** gamma
    assert difficulty.shape == (loss.shape[0],), f"difficulty should be in shape ({loss.shape[0]},), but here gets {difficulty.shape}"
    assert torch.is_floating_point(difficulty), f"difficulty should be in float but here gets {difficulty.dtype}"
    # Combine factors
    focal_loss = loss * freq_factor * difficulty

    return focal_loss


class MACEHuberForceFocalLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mace_huber_force_focal"] = "mace_huber_force_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> torch.Tensor:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        # Define the multiplication factors for each condition
        factors = self.delta * np.array([1.0, 0.7, 0.4, 0.1])

        # Apply multiplication factors based on conditions
        c1 = torch.norm(y_true, dim=-1) < 100
        c2 = (torch.norm(y_true, dim=-1) >= 100) & (torch.norm(y_true, dim=-1) < 200)
        c3 = (torch.norm(y_true, dim=-1) >= 200) & (torch.norm(y_true, dim=-1) < 300)
        c4 = ~(c1 | c2 | c3)

        se = torch.zeros_like(y_pred)

        se[c1] = F.huber_loss(
            y_true[c1], y_pred[c1], reduction="none", delta=factors[0]
        )
        se[c2] = F.huber_loss(
            y_true[c2], y_pred[c2], reduction="none", delta=factors[1]
        )
        se[c3] = F.huber_loss(
            y_true[c3], y_pred[c3], reduction="none", delta=factors[2]
        )
        se[c4] = F.huber_loss(
            y_true[c4], y_pred[c4], reduction="none", delta=factors[3]
        )

        # Reduce over the 3 force components
        loss = se.mean(dim=-1)

        # Compute frequency factor for each sample based on its atom type
        if not hasattr(batch, "atomic_numbers"):
            raise ValueError("atomic_numbers is required")
        atomic_numbers = batch.atomic_numbers
        loss = _apply_focal_loss_per_atom(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=atomic_numbers,
            gamma=self.gamma,
        )

        # Reduce the loss
        loss = self.reduce(loss, reduction)
        return loss


class MACEHuberEnergyLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mace_huber_energy"] = "mace_huber_energy"

    delta: float

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        if not hasattr(batch, "num_atoms"):
            raise ValueError("natoms is required")
        num_atoms = batch.num_atoms.reshape(-1, 1)
        # First, divide the energy by the number of atoms
        y_pred = y_pred / num_atoms
        y_true = y_true / num_atoms

        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction="none", delta=self.delta)

        return self.reduce(loss, reduction)


def _apply_focal_loss_per_graph(
    *,
    loss: torch.Tensor,  # A float tensor with "batch_size" dimension
    freq_ratios: torch.Tensor,  # A float tensor with "atom_type" dimension
    atomic_numbers: torch.Tensor,  # An int tensor with "natoms" (number of atoms) dimension
    batch: torch.Tensor,  # An int tensor with "natoms" dimension
    gamma: float,
):
    # Apply the frequency factor to each sample & compute the mean frequency factor for each graph
    freq_factor = scatter(
        freq_ratios[atomic_numbers],
        batch,
        dim=0,
        dim_size=loss.shape[0],
        reduce="mean",
    )
    assert freq_factor.shape == (loss.shape[0],), f"freq_factor should be in shape ({loss.shape[0]},), but here gets {freq_factor.shape}"
    assert torch.is_floating_point(freq_factor), f"freq_factor should be in float but here gets {freq_factor.dtype}"

    # Compute difficulty factor
    # We'll use the Huber loss value instead of MAE for the difficulty
    normalized_loss = loss / loss.max()
    difficulty = (1 - torch.exp(-normalized_loss)) ** gamma
    assert difficulty.shape == (loss.shape[0],), f"difficulty should be in shape ({loss.shape[0]},), but here gets {difficulty.shape}"
    assert torch.is_floating_point(difficulty), f"difficulty should be in float but here gets {difficulty.dtype}"

    # Combine factors
    focal_loss = loss * freq_factor * difficulty

    return focal_loss


class MACEHuberEnergyFocalLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["mace_huber_energy_focal"] = "mace_huber_energy_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> torch.Tensor:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        if not hasattr(batch, "num_atoms"):
            raise ValueError("natoms is required")
        num_atoms = batch.num_atoms.reshape(-1, 1)
        if not hasattr(batch, "atomic_numbers"):
            raise ValueError("atomic_numbers is required")
        atomic_numbers = batch.atomic_numbers
        if not hasattr(batch, "batch"):
            raise ValueError("batch is required")
        batch = batch.batch
        assert y_pred.shape == y_true.shape, f"y_pred and y_true should have the same shape, but here gets {y_pred.shape} and {y_true.shape}"
        assert y_pred.shape[0] == num_atoms.shape[0], f"y_pred and natoms should have the same batch size, but here gets {y_pred.shape[0]} and {num_atoms.shape[0]}"
        assert torch.is_floating_point(y_pred), f"y_pred should be in float but here gets {y_pred.dtype}"
        assert torch.is_floating_point(y_true), f"y_true should be in float but here gets {y_true.dtype}"
        # First, divide the energy by the number of atoms
        y_pred = y_pred / num_atoms
        y_true = y_true / num_atoms

        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction="none", delta=self.delta)
        assert loss.shape == (y_pred.shape[0],), f"loss should be in shape ({y_pred.shape[0]},), but here gets {loss.shape}"
        assert torch.is_floating_point(loss), f"loss should be in float but here gets {loss.dtype}"

        # Compute frequency factor for each sample based on its atom type
        loss = _apply_focal_loss_per_graph(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=atomic_numbers,
            batch=batch,
            gamma=self.gamma,
        )
        assert loss.shape == (y_pred.shape[0],), f"loss should be in shape ({y_pred.shape[0]},), but here gets {loss.shape}"
        assert torch.is_floating_point(loss), f"loss should be in float but here gets {loss.dtype}"

        loss = self.reduce(loss, reduction)
        return loss


class HuberStressFocalLossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["huber_stress_focal"] = "huber_stress_focal"

    delta: float  # Huber loss delta

    freq_ratios_path: Path
    gamma: float = 2.0  # Focal loss gamma

    @cached_property
    def freq_ratios(self) -> torch.Tensor:
        freq_ratios = np.load(self.freq_ratios_path)
        return torch.from_numpy(freq_ratios).float()

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        if not hasattr(batch, "atomic_numbers"):
            raise ValueError("atomic_numbers is required")
        atomic_numbers = batch.atomic_numbers
        if not hasattr(batch, "batch"):
            raise ValueError("batch is required")
        batch = batch.batch
        # Compute the loss
        loss = F.huber_loss(y_pred, y_true, reduction="none", delta=self.delta)
        assert loss.shape == (y_pred.shape[0], 3, 3), f"loss should be in shape ({y_pred.shape[0]}, 3, 3), but here gets {loss.shape}"
        assert torch.is_floating_point(loss), f"loss should be in float but here gets {loss.dtype}"

        # Reduce over the 3x3 stress tensor
        loss = loss.mean(dim=(-2, -1))
        assert loss.shape == (y_pred.shape[0],), f"loss should be in shape ({y_pred.shape[0]},), but here gets {loss.shape}"
        assert torch.is_floating_point(loss), f"loss should be in float but here gets {loss.dtype}"
        
        # Compute frequency factor for each sample based on its atom type
        loss = _apply_focal_loss_per_graph(
            loss=loss,
            freq_ratios=self.freq_ratios.to(loss.device),
            atomic_numbers=atomic_numbers,
            batch=batch,
            gamma=self.gamma,
        )
        assert loss.shape == (y_pred.shape[0],), f"loss should be in shape ({y_pred.shape[0]},), but here gets {loss.shape}"
        assert torch.is_floating_point(loss), f"loss should be in float but here gets {loss.dtype}"

        # Reduce the loss
        loss = self.reduce(loss, reduction)
        return loss


class L2MAELossConfig(LossConfigBase, Generic[TBatch]):
    name: Literal["l2_mae"] = "l2_mae"

    p: int | float = 2

    @override
    def compute_impl(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        batch: TBatch,
        reduction: Reduction = "mean",
    ) -> torch.Tensor:
        loss = F.pairwise_distance(y_pred, y_true, p=self.p)
        return self.reduce(loss, reduction)


LossConfig: TypeAlias = Annotated[
    MAELossConfig
    | MSELossConfig
    | HuberLossConfig
    | MACEHuberLossConfig
    | MACEHuberEnergyLossConfig
    | MACEHuberForceFocalLossConfig
    | MACEHuberEnergyFocalLossConfig
    | HuberStressFocalLossConfig
    | L2MAELossConfig,
    Field(discriminator="name")
]
