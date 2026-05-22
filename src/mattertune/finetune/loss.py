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
    ss_paths_info: list[list[dict[str, torch.Tensor]]] = [[{"edge": torch.tensor([0.0],)}]]
     

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
    y = (w * yk.unsqueeze(0)).sum(dim = -1)
    return y

ETOK = 0.262465831

# https://github.com/xraypy/xraylarch/blob/6a68e776c3b10625bcda556432f45a4ddb6b18d1/larch/xafs/feffdat.py#L632
def calc_chi(q, deltar, sigma2, third, fourth, amp, pha, rep, lam, reff, ei):
    pp = (rep + 1j / lam) ** 2 + 1j * ei * ETOK
    check(pp.real, "pp.real")
    check(pp.imag, "pp.imag")

    p = torch.sqrt(pp)
    check(p.real, "p.real")
    check(p.imag, "p.imag")

    cchi = torch.exp(
        -2 * reff * p.imag
        - 2 * pp * (sigma2 - pp * fourth / 3.0)
        + 1j * (
            2 * q * reff
            + pha
            + 2 * p *(deltar - 2 * sigma2 / reff - 2 * pp * third / 3.0)
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

def check(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise RuntimeError(f"{name} has NaNs/Infs")

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
            
            ΔE0_max = k2s[sl] - 0.01
            ΔE0_min = -16 + k2s[sr]
            sim_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            exp_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            for i, struct_i_flt in enumerate(label):
                struct_i = struct_i_flt.int()
                chi = torch.zeros((len(config.abs_inds[struct_i]), len(k1s[sl:sr])), device = prediction.device)
                n_edge = batch.n_edge[i]
                edge_vecs = batch.edge_features["vectors"][tot_edge:(tot_edge + n_edge)]
                edge_preds = prediction[tot_edge:(tot_edge + n_edge)]
                check(edge_preds, "edge_preds")
                edge_Reffs = torch.linalg.norm(edge_vecs, dim = 1)
                receivers = batch.receivers[tot_edge:(tot_edge + n_edge)]
                senders = batch.senders[tot_edge:(tot_edge + n_edge)]
                scatterer_Zs = batch.node_features["atomic_numbers"][receivers]
                for j, abs_i in enumerate(config.abs_inds[struct_i]):
                    abs_mask = (senders - batch.n_node[:i].sum()) == abs_i
                    abs_Reffs = edge_Reffs[abs_mask]
                    abs_Zs = scatterer_Zs[abs_mask]
                    abs_preds = edge_preds[abs_mask]

                    E0 = torch.mean(abs_preds[:, 0])
                    if abs_preds.shape[0] == 0:
                        raise RuntimeError(f"Empty absorber at structure {i}, absorber {j}")
                    check(E0, "E0")
                    ΔE0 = torch.clamp(E0 - config.ss_paths_info[struct_i][j]["edge"], min = ΔE0_min, max = ΔE0_max)
                    check(ΔE0, "ΔE0")
                    
                    arg = k2s[sl:sr] - ΔE0
                    check(arg, "sqrt arg BEFORE clamp")

                    q = torch.sqrt(torch.clamp(arg, min=1e-12))
                    check(q, "q")

                    k_feff = config.ss_paths_info[struct_i][j]["k_feff"]

                    for k in range(len(abs_Reffs)):
                        path_ind = torch.argmin(torch.abs(torch.where(config.ss_paths_info[struct_i][j]["scatterer_Zs"] == abs_Zs[k], config.ss_paths_info[struct_i][j]["Reffs"], -10.0) - abs_Reffs[k]))
                        if torch.abs(config.ss_paths_info[struct_i][j]["Reffs"][path_ind] - abs_Reffs[k]) < 0.01:
                            amp = interp_soft_adaptive(q, k_feff, config.ss_paths_info[struct_i][j]["amp"][path_ind])
                            pha = interp_soft_adaptive(q, k_feff, config.ss_paths_info[struct_i][j]["pha"][path_ind])
                            rep = interp_soft_adaptive(q, k_feff, config.ss_paths_info[struct_i][j]["rep"][path_ind])
                            lam = interp_soft_adaptive(q, k_feff, config.ss_paths_info[struct_i][j]["lam"][path_ind])
                            
                            check(amp, "amp")
                            check(pha, "pha")
                            check(rep, "rep")
                            check(lam, "lam")

                            deltar = abs_preds[k, 1]
                            sigma2 = abs_preds[k, 2]
                            third = abs_preds[k, 3]

                            check(deltar, "deltar")
                            check(sigma2, "sigma2")
                            check(third, "third")

                            fourth = 0#abs_preds[k, 4]
                            chi[j] += calc_chi(q, deltar, sigma2, third, fourth, amp, pha, rep, lam, config.ss_paths_info[struct_i][j]["Reffs"][path_ind], 0.0)
                tot_edge += n_edge
            
                mean_sim_chi = torch.mean(chi, dim = 0) 
                sim_chis[i] = mean_sim_chi / torch.linalg.norm(mean_sim_chi)
                exp_chis[i] = config.exp_spectra[struct_i][sl:sr] / torch.linalg.norm(config.exp_spectra[struct_i][sl:sr])
                raise KeyError(f'Sim {sim_chis[i]} \n Exp {exp_chis[i]}')
            return F.mse_loss(sim_chis, exp_chis, reduction=config.reduction)

        case _:
            assert_never(config)
