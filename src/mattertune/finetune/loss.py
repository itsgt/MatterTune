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

def interp_soft_adaptive(x, xk, yk, beta = 1.0):
    dxk = xk[1:] - xk[:-1]                     # (n-1,)
    dx = x.unsqueeze(-1) - xk.unsqueeze(0)     # (nk, n)
    h = ((torch.cat([dxk[:1], dxk]) + torch.cat([dxk, dxk[-1:]])) / 2).unsqueeze(0)
    r = dx / h
    w = torch.exp(-beta * r ** 2)
    w = w / w.sum(dim = -1, keepdim = True)
    y = (w.unsqueeze(0) * yk.unsqueeze(1)).sum(dim=-1)
    return y

ETOK = 0.262465831

# https://github.com/xraypy/xraylarch/blob/6a68e776c3b10625bcda556432f45a4ddb6b18d1/larch/xafs/feffdat.py#L632
def calc_chi_batch(q, deltar, sigma2, third, fourth, amp, pha, rep, lam, reff, ei):
    q = q.unsqueeze(0)                # (1, Nk)
    deltar = deltar[:, None]          # (Np, 1)
    sigma2 = sigma2[:, None]
    third  = third[:, None]
    fourth = fourth[:, None]
    reff   = reff[:, None]

    
    pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
    p = torch.sqrt(pp)

    cchi = torch.exp(
        -2 * reff * p.imag
        - 2 * pp * (sigma2 - pp * fourth / 3.0)
        + 1j * (
            2 * q * reff
            + pha
            + 2 * p * (deltar - 2 * sigma2 / reff - 2 * pp * third / 3.0)
        )
    ) * amp / (q * (reff + deltar) ** 2)
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
    match config:
        case EXAFSLossConfig():
            tot_edge = 0
            k1s = torch.arange(0.0, 16.00001, 0.05, device = prediction.device)
            k2s = k1s ** 2
            k3s = k1s ** 3
            sl = 60
            sr = -10
            k_feff = batch.system_features["feff_k"][:(len(batch.system_features["feff_k"]) // len(label))]
            
            ΔE0_max = k2s[sl] - 0.05
            ΔE0_min = k2s[sr] - 16 ** 2
            mid_ΔE0_range = 0.5 * (ΔE0_max + ΔE0_min)
            half_ΔE0_range = 0.5 * (ΔE0_max - ΔE0_min)
            sim_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            exp_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            edge_inds = torch.arange(batch.edge_features["vectors"].size(dim = 0), device = prediction.device)
            for i, struct_i_flt in enumerate(label):
                n_edge = batch.n_edge[i]
                struct_i = struct_i_flt.int()
                struct_edge_inds = edge_inds[tot_edge:(tot_edge + n_edge)]
                unique_abs_inds = torch.unique(batch.system_features["abs_inds"][tot_edge:(tot_edge + n_edge)], sorted = True)
                chi = torch.zeros((len(unique_abs_inds), len(k1s[sl:sr])), device = prediction.device)
                edge_preds = prediction[tot_edge:(tot_edge + n_edge)]
                for j, abs_i in enumerate(unique_abs_inds):
                    abs_mask = batch.system_features["abs_inds"][tot_edge:(tot_edge + n_edge)] == abs_i
                    abs_preds = edge_preds[abs_mask]
                    abs_edge_inds = struct_edge_inds[abs_mask]

                    ΔE0 = mid_ΔE0_range + half_ΔE0_range * torch.tanh(torch.mean(abs_preds[:, 0]) / half_ΔE0_range)
                    
                    q = torch.sqrt(k2s[sl:sr] - ΔE0)
                    pinds = batch.system_features["path_inds"][abs_edge_inds]
                    valid = pinds >= 0

                    pinds = pinds[valid].int()
                    ss_preds = abs_preds[valid]

                    amp = interp_soft_adaptive(q, k_feff, batch.system_features["amp"][pinds])
                    pha = interp_soft_adaptive(q, k_feff, batch.system_features["pha"][pinds])
                    rep = interp_soft_adaptive(q, k_feff, batch.system_features["rep"][pinds])
                    lam = interp_soft_adaptive(q, k_feff, batch.system_features["lam"][pinds])

                    deltar = 0.2 * torch.tanh(ss_preds[:, 1])
                    sigma2 = 0.05 * torch.sigmoid(ss_preds[:, 2])
                    third  = 0.01 * torch.tanh(ss_preds[:, 3])
                    fourth = torch.zeros_like(deltar)
                    Reff = batch.system_features["Reffs"][pinds]

                    chi_paths = calc_chi_batch(
                        q, deltar, sigma2, third, fourth,
                        amp, pha, rep, lam,
                        Reff
                    )   # (N_paths, Nk)
                    chi[j] = torch.sum(chi_paths, dim = 0)

                tot_edge += n_edge
            
                mean_sim_chi = torch.mean(chi, dim = 0) 
                sim_chis[i] = mean_sim_chi / torch.linalg.norm(mean_sim_chi)
                exp_chis[i] = config.exp_spectra[struct_i][sl:sr] / torch.linalg.norm(config.exp_spectra[struct_i][sl:sr])
            return F.mse_loss(sim_chis, exp_chis, reduction=config.reduction)

        case _:
            assert_never(config)
