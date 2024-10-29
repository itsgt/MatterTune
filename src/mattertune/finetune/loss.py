from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from typing_extensions import assert_never


class MAELossConfig(BaseModel):
    name: Literal["mae"] = "mae"


class MSELossConfig(BaseModel):
    name: Literal["mse"] = "mse"


class HuberLossConfig(BaseModel):
    name: Literal["huber"] = "huber"

    delta: float = 1.0


class L2MAELossConfig(BaseModel):
    name: Literal["l2_mae"] = "l2_mae"


def l2_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    distances = F.pairwise_distance(output, target, p=2)
    match reduction:
        case "mean":
            return distances.mean()
        case "sum":
            return distances.sum()
        case _:
            assert_never(reduction)


LossConfig: TypeAlias = Annotated[
    MAELossConfig | MSELossConfig | HuberLossConfig | L2MAELossConfig,
    Field(discriminator="name"),
]
