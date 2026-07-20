from __future__ import annotations

from typing import Annotated, Literal

import nshconfig as C
import torch
import torch.nn.functional as F
from typing_extensions import TypeAliasType, assert_never

def smoothness_loss(y):
    x = y / y.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return ((x[:, 2:] - 2*x[:, 1:-1] + x[:, :-2])**2).mean()

class MAELossConfig(C.Config):
    name: Literal["mae"] = "mae"
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

class MAEMaskedLossConfig(C.Config):
    name: Literal["mae_masked"] = "mae_masked"
    reduction: Literal["mean", "sum"] = "mean"
    natoms: int = 80
    mask: torch.Tensor = torch.tensor([True for _ in range(10)], 
        dtype = torch.bool)

class EXAFSLossConfig(C.Config):
    name: Literal["exafs"] = "exafs"
    reduction: Literal["mean", "sum"] = "mean"
    deltaR_only: bool = False
    offset_deltaR: float = 0.0
    offset_sigma2: float = 0.0
    offset_third:  float = 0.0
    offset_fourth: float = 0.0
    scale_deltaR: float = 1.0
    scale_sigma2: float = 1.0
    scale_third:  float = 1.0
    scale_fourth: float = 1.0
     
class MAEAtomAveragedLossConfig(C.Config):
    name: Literal["mae_atom_avg"] = "mae_atom_avg"
    reduction: Literal["mean", "sum"] = "mean"

class CosAtomAveragedLossConfig(C.Config):
    name: Literal["cos_atom_avg"] = "cos_atom_avg"
    reduction: Literal["mean", "sum"] = "mean"
    smoothness_λ: float = 1e-3

class MAEWeightedLossConfig(C.Config):
    name: Literal["mae_weighted"] = "mae_weighted"
    reduction: Literal["mean", "sum"] = "mean"
    w: torch.Tensor = torch.pow(torch.arange(1.0, 10.0, 0.01), 2) / 100
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

class MAEWithSTDLossConfig(C.Config):
    name: Literal["mae_with_std"] = "mae_with_std"
    λ: float = 1.0
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

class MAEWithDerivConfig(C.Config):
    name: Literal["mae_with_deriv"] = "mae_with_deriv"
    λ: float = 0.1
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

class MSELossConfig(C.Config):
    name: Literal["mse"] = "mse"
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """

class NoLossConfig(C.Config):
    name: Literal["none"] = "none"
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """


class HuberLossConfig(C.Config):
    name: Literal["huber"] = "huber"
    delta: float = 1.0
    """The threshold value for the Huber loss function."""
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """


class L2MAELossConfig(C.Config):
    name: Literal["l2_mae"] = "l2_mae"
    reduction: Literal["mean", "sum"] = "mean"
    """How to reduce the loss values across the batch.

    - ``"mean"``: The mean of the loss values.
    - ``"sum"``: The sum of the loss values.
    """


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

LossConfig = TypeAliasType(
    "LossConfig",
    Annotated[
        MAELossConfig | MAEWithSTDLossConfig | MAEWithDerivConfig | EXAFSLossConfig | MAEWeightedLossConfig | MAEMaskedLossConfig | MSELossConfig | NoLossConfig | HuberLossConfig | L2MAELossConfig | MAEAtomAveragedLossConfig | CosAtomAveragedLossConfig,
        C.Field(discriminator="name"),
    ],
)


def compute_loss(
    config: LossConfig,
    prediction: torch.Tensor,
    label: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the loss value given the model output, ``prediction``,
    and the target label, ``label``.

    The loss value should be a scalar tensor.

    Args:
        config: The loss configuration.
        prediction: The model output.
        label: The target label.

    Returns:
        The computed loss value.
    """
    try:
        prediction = prediction.reshape(label.shape)
    except RuntimeError:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match ground truth shape {label.shape}"
        )

    match config:
        case MAELossConfig():
            return F.l1_loss(prediction, label, reduction=config.reduction)

        case MAEMaskedLossConfig():
            mask = config.mask.repeat(int(prediction.shape[0] / config.natoms))
            return F.l1_loss(prediction[mask, :], label[mask, :], 
                reduction=config.reduction)
        
        case MAEAtomAveragedLossConfig():
            unique_labels, counts = torch.unique_consecutive(
                label[:, -1], return_counts=True)

            label_means = []
            pred_means = []

            count_so_far = 0
            for i in range(len(unique_labels)):
                mask = torch.max(torch.abs(label[count_so_far:(count_so_far + counts[i]), :-1]), axis = 1).values > 0
                label_mean = torch.mean(label[count_so_far:(count_so_far + counts[i]), :-1][mask, :], axis = 0)
                pred_mean = torch.mean(prediction[count_so_far:(count_so_far + counts[i]), :-1][mask, :], axis = 0)
                label_means.append(label_mean)
                pred_means.append(pred_mean)
                count_so_far += counts[i]
        
            return F.l1_loss(
                torch.stack(pred_means),
                torch.stack(label_means),
                reduction=config.reduction,
            )

        case CosAtomAveragedLossConfig():
            unique_labels, counts = torch.unique_consecutive(
                label[:, -1], return_counts=True)

            label_means = []
            pred_means = []

            count_so_far = 0
            for i in range(len(unique_labels)):
                mask = torch.max(torch.abs(label[count_so_far:(count_so_far + counts[i]), :-1]), axis = 1).values > 0
                label_mean = torch.mean(label[count_so_far:(count_so_far + counts[i]), :-1][mask, :], axis = 0)
                pred_mean = torch.mean(prediction[count_so_far:(count_so_far + counts[i]), :-1][mask, :], axis = 0)
                label_means.append(label_mean)
                pred_means.append(pred_mean)
                count_so_far += counts[i]
        
            cos_sim = F.cosine_similarity(
                torch.stack(pred_means),
                torch.stack(label_means),
                dim=1,
            )

            loss = 1.0 - cos_sim + config.smoothness_λ * smoothness_loss(torch.stack(pred_means))
            if config.reduction == "mean":
                return loss.mean()
            elif config.reduction == "sum":
                return loss.sum()
            else:  # "none"
                return loss

        case MAEWithDerivConfig():
            mae_loss = F.l1_loss(prediction, label, reduction=config.reduction)
            deriv_loss = F.l1_loss(prediction[:, 1:] - prediction[:, :-1], 
                label[:, 1:] - label[:, :-1], reduction=config.reduction)
            return mae_loss + config.λ * deriv_loss

        case MAEWithSTDLossConfig():
            mae_loss = F.l1_loss(prediction, label, reduction=config.reduction)
            std_loss = torch.mean(torch.abs(torch.std(prediction, dim = 1
                ) - torch.std(label, dim = 1)))
            return mae_loss + config.λ * std_loss

        case MAEWeightedLossConfig():
            r_w = config.w.to(prediction.device).repeat((prediction.shape[0], 1))
            return F.l1_loss(r_w * prediction, r_w * label, reduction=config.reduction)

        case MSELossConfig():
            return F.mse_loss(prediction, label, reduction=config.reduction)

        case HuberLossConfig():
            return F.huber_loss(
                prediction, label, delta=config.delta, reduction=config.reduction
            )

        case L2MAELossConfig():
            return l2_mae_loss(prediction, label, reduction=config.reduction)

        case NoLossConfig():
            return 0 * F.l1_loss(prediction, label, reduction=config.reduction)

        case _:
            assert_never(config)


def compute_loss_with_batch(
    config: LossConfig,
    prediction: torch.Tensor,
    label: torch.Tensor,
    batch,
) -> torch.Tensor:
    match config:
        case EXAFSLossConfig():
            edge_match = batch.system_features["edge_match"].int()
            N_edges = batch.system_features["N_edges"].int()
            struct_is = batch.system_features["struct_i"].int()
            edge_match_id = batch.system_features["edge_match_id"].int()
            edge_match_id_mapped = (edge_match_id[:, None] == struct_is[None, :]).nonzero()[:, 1]
            edge_offsets = torch.cumsum(N_edges, dim = 0) - N_edges
            edge_match = edge_match + edge_offsets[edge_match_id_mapped]

            errs = torch.linalg.norm(batch.edge_features["vectors"][edge_match] - 
                batch.system_features["edge_vec_check"], axis = 1)
            assert errs.max() < 0.02

            if config.deltaR_only:
                return F.mse_loss(config.offset_deltaR + config.scale_deltaR * prediction[edge_match, 0], batch.system_features["deltar"], reduction=config.reduction)
            if config.sigma2_only:
                return F.mse_loss(config.offset_sigma2 + config.scale_sigma2 * prediction[edge_match, 1], batch.system_features["sigma2"], reduction=config.reduction)
            else:
                return  ((1 / config.scale_deltaR ** 2) * F.mse_loss(config.offset_deltaR + config.scale_deltaR * prediction[edge_match, 0], batch.system_features["deltar"], reduction=config.reduction
                    ) + (1 / config.scale_sigma2 ** 2) * F.mse_loss(config.offset_sigma2 + config.scale_sigma2 * prediction[edge_match, 1], batch.system_features["sigma2"], reduction=config.reduction)
                    )# + (1 / config.scale_third ** 2) * F.mse_loss(config.offset_third + config.scale_third * prediction[edge_match, 2], batch.system_features["third"],  reduction=config.reduction
                    #) + (1 / config.scale_fourth ** 2) * F.mse_loss(config.offset_fourth + config.scale_fourth * prediction[edge_match, 3], batch.system_features["fourth"], reduction=config.reduction))
        case _:
            assert_never(config)
