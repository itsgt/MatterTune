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
    ws: list[float] = [0.9, 0.1, 0.1]
    exp_spectra: torch.Tensor = torch.tensor([0.])
    abs_inds: list[torch.Tensor] = [torch.tensor([0.])]
    ss_paths_info = None
     

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

def interp_linear_batch(x, xk, yk):
    x = np.asarray(x)
    xk = np.asarray(xk)
    yk = np.asarray(yk)

    m, n = yk.shape
    nk = x.size

    # interval indices
    idx = np.searchsorted(xk, x, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    x0 = xk[idx]
    x1 = xk[idx + 1]

    w = (x - x0) / (x1 - x0)   # (nk,)

    # broadcasted gather
    y0 = yk[:, idx]            # (m, nk)
    y1 = yk[:, idx + 1]        # (m, nk)

    return y0 + (y1 - y0) * w

# https://github.com/xraypy/xraylarch/blob/6a68e776c3b10625bcda556432f45a4ddb6b18d1/larch/xafs/feffdat.py#L632
def calc_chi_linear_vectorized(q, s02, deltar, sigma2, third, fourth,
    amp, pha, rep, lam, reff, degen, ei):
    reff   = reff[:,   None]
    degen  = degen[:,  None]
    deltar = deltar[:, None]
    sigma2 = sigma2[:, None]
    third  = third[:,  None]
    fourth = fourth[:, None]

    pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
    p  = np.sqrt(pp)

    cchi = np.exp(
        -2 * reff * p.imag
        - 2 * pp * (sigma2 - pp * fourth / 3.0)
        + 1j * (
            2 * q * reff
            + pha
            + 2 * p *(deltar - 2 * sigma2 / reff - 2 * pp * third / 3.0)
        )
    ) * degen * s02 * amp / (q * (reff + deltar) ** 2)
    return cchi.imag


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

    match config:
        case EXAFSLossConfig():
            tot_edge = 0
            for i, struct_i in label:
                n_edge = batch.n_edge[i]
                edge_vecs = batch.edge_features['vectors'][tot_edge:(tot_edge + n_edge)]
                edge_preds = prediction[tot_edge:(tot_edge + n_edge)]
                edge_Reffs = torch.linalg.norm(edge_vecs, dim = 0)
                receivers = batch.receivers[tot_edge:(tot_edge + n_edge)]
                senders = batch.senders[tot_edge:(tot_edge + n_edge)]
                scatterer_Zs = batch.node_features['atomic numbers'][receivers]
                for abs_i in config.abs_inds[struct_i]:
                    abs_Reffs = edge_Reffs[senders == abs_i]
                    abs_Zs = scatterer_Zs[senders == abs_i]
                    abs_preds = edge_preds[senders == abs_i]




            raise ValueError(batch)
            raise ValueError(f'Edge Feat. Vectors: size {batch.edge_features["vectors"].size()} values {str(batch.edge_features["vectors"])}\n' + 
                f'Edge Feat. Unit Shifts: size {batch.edge_features["unit_shifts"].size()} values {str(batch.edge_features["unit_shifts"])}\n' + 
                f'Senders: size {batch.senders.size()} values {str(batch.senders)}\n' + 
                f'Receivers: size {batch.receivers.size()} values {str(batch.receivers)}\n' + 
                f'Prediction: size {prediction.size()} values {str(prediction)}\n'
            ) 

        case _:
            assert_never(config)
