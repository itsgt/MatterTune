from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from typing_extensions import assert_never, override


class LossConfigBase(BaseModel, ABC):
    @abstractmethod
    def compute_loss(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss value given the model output, ``prediction``,
            and the target label, ``label``.

        The loss value should be a scalar tensor.

        Args:
            prediction (torch.Tensor): The model output.
            label (torch.Tensor): The target label.

        Returns:
            torch.Tensor: The computed loss value.
        """


class MAELossConfig(LossConfigBase):
    name: Literal["mae"] = "mae"

    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

    @override
    def compute_loss(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return F.l1_loss(prediction, label, reduction=self.reduction)


class MSELossConfig(LossConfigBase):
    name: Literal["mse"] = "mse"

    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

    @override
    def compute_loss(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return F.mse_loss(prediction, label, reduction=self.reduction)


class HuberLossConfig(LossConfigBase):
    name: Literal["huber"] = "huber"

    delta: float = 1.0
    """The threshold value for the Huber loss function."""

    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

    @override
    def compute_loss(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return F.huber_loss(
            prediction, label, delta=self.delta, reduction=self.reduction
        )


class L2MAELossConfig(LossConfigBase):
    name: Literal["l2_mae"] = "l2_mae"

    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

    @override
    def compute_loss(
        self,
        prediction: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        return l2_mae_loss(prediction, label, reduction=self.reduction)


def l2_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    distances = F.pairwise_distance(output, target, p=2)
    match reduction:
        case "mean":
            return distances.mean()
        case "sum":
            return distances.sum()
        case "none":
            return distances
        case _:
            assert_never(reduction)


LossConfig: TypeAlias = Annotated[
    MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig,
    Field(discriminator="name"),
]
