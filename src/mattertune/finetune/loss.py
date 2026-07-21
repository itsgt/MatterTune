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
    avg_paths: bool = False

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

# https://github.com/xraypy/xraylarch/blob/ba8a45062a59670d64ab492b29a379265dcec34c/larch/xafs/sigma2_models.py#L93
def debint_z(t, N = 100):
    x = torch.linspace(0, 1, N, device = t.device, dtype = t.dtype)  # (N,)
    x = x.unsqueeze(0)  # (1, N)
    t = t.unsqueeze(1)  # (B, 1)
    y = (2.0 / t) * torch.ones_like(x)  # (B, N)
    xt = x[:, 1:] * t  # (B, N-1)
    exp_xt = torch.exp(torch.clamp(xt, max=80))
    coth_xt_2 = 1.0 + 2.0 / (exp_xt - 1.0 + 1e-12)
    y[:, 1:] = x[:, 1:] * coth_xt_2
    assert ~torch.any(torch.isnan(y)), f't: {t}\n y: {y}'
    return torch.trapz(y, x, dim=1)  # (B,)

def debint(r, t, N = 100):
    x = torch.linspace(0, 1, N, device = t.device, dtype = t.dtype)  # (N,)
    x = x.unsqueeze(0)  # (1, N)
    r = r.unsqueeze(1)  # (B, 1)
    t = t.unsqueeze(1)  # (B, 1)
    y = (2.0 / t) * torch.ones_like(x)  # (B, N)
    xt = x[:, 1:] * t  # (B, N-1)
    rx = r * x[:, 1:]  # (B, N-1)
    exp_xt = torch.exp(torch.clamp(xt, max=80))
    coth_xt_2 = 1.0 + 2.0 / (exp_xt - 1.0 + 1e-12)
    y[:, 1:] = x[:, 1:] * torch.sinc(rx / torch.pi) * coth_xt_2
    assert ~torch.any(torch.isnan(y)), f'r: {r}\n t: {t}\n y: {y}'
    return torch.trapz(y, x, dim=1)  # (B,)

def sigma2_debye_SS(tx, R, m_a, m_s, rnorman, t):
    conh = 72.7630804732553 / (t * tx)
    conr = 4.5693349700844 / rnorman
    deb_z = debint_z(tx)
    C_A  = deb_z / m_a
    C_S  = deb_z / m_s
    C_AS = debint(conr * R, tx) / torch.sqrt(m_a * m_s)
    return conh * (C_A + C_S - 2 * C_AS)

def sigma2_debye_MS(tx, natoms, pos, atwt, rnorman, t):
    conh = 72.7630804732553 / (2 * t * tx)
    conr = 4.5693349700844 / rnorman

    offsets = torch.cumsum(torch.cat([torch.zeros(1, device = natoms.device, 
        dtype = natoms.dtype), natoms[:-1]]), dim = 0)
    Ntot = pos.shape[0]
    assert Ntot == torch.sum(natoms)
    path_ids = torch.repeat_interleave(torch.arange(len(natoms), 
        device = pos.device), natoms)
    local_i = torch.arange(Ntot, device = pos.device) - offsets[path_ids]
    
    i0 = torch.arange(Ntot, device = pos.device)
    i1 = offsets[path_ids] + (local_i + 1) % natoms[path_ids]
    d = pos[i0] - pos[i1]
    ridotj = torch.sum(d ** 2, dim = 1)
    ri0j1 = conr * torch.sqrt(ridotj)
    ridotj /= torch.abs(ridotj)
    diz = debint_z(tx * torch.ones_like(ri0j1))
    ci0i1 = debint(ri0j1, tx * torch.ones_like(ri0j1)) / torch.sqrt(atwt[i0] * atwt[i1])
    sig2_eq_all = ridotj * (diz / atwt[i0] + diz / atwt[i1] - 2 * ci0i1) / 2
    sig2_eq = torch.zeros_like(natoms, dtype = pos.dtype, device = pos.device)
    sig2_eq = sig2_eq.scatter_add(0, path_ids, sig2_eq_all)

    pair_i0 = []
    pair_j0 = []
    pair_path_ids = []
    for b in range(len(natoms)):
        n = natoms[b]
        i0_b, j0_b = torch.triu_indices(n, n, offset = 1, device = pos.device)
        pair_i0.append(i0_b + offsets[b])
        pair_j0.append(j0_b + offsets[b])
        pair_path_ids.append(torch.full((i0_b.shape[0],), b, device = pos.device))
    pair_i0 = torch.cat(pair_i0)
    pair_j0 = torch.cat(pair_j0)
    pair_path_ids = torch.cat(pair_path_ids)
    pair_i1 = offsets[pair_path_ids] + (pair_i0 - offsets[pair_path_ids] + 1) % natoms[pair_path_ids]
    pair_j1 = offsets[pair_path_ids] + (pair_j0 - offsets[pair_path_ids] + 1) % natoms[pair_path_ids]
    ri0j0  = torch.linalg.norm(pos[pair_i0] - pos[pair_j0], dim = 1)
    ri1j1  = torch.linalg.norm(pos[pair_i1] - pos[pair_j1], dim = 1)
    ri0j1  = torch.linalg.norm(pos[pair_i0] - pos[pair_j1], dim = 1)
    ri1j0  = torch.linalg.norm(pos[pair_i1] - pos[pair_j0], dim = 1)
    ri0i1  = torch.linalg.norm(pos[pair_i0] - pos[pair_i1], dim = 1)
    rj0j1  = torch.linalg.norm(pos[pair_j0] - pos[pair_j1], dim = 1)
    ridotj = torch.sum((pos[pair_i0] - pos[pair_i1]) * (pos[pair_j0] - pos[pair_j1]), dim = 1)
    ci0j0 = debint(conr * ri0j0, tx * torch.ones_like(ri0j0)) / torch.sqrt(atwt[pair_i0] * atwt[pair_j0])
    ci1j1 = debint(conr * ri1j1, tx * torch.ones_like(ri1j1)) / torch.sqrt(atwt[pair_i1] * atwt[pair_j1])
    ci0j1 = debint(conr * ri0j1, tx * torch.ones_like(ri0j1)) / torch.sqrt(atwt[pair_i0] * atwt[pair_j1])
    ci1j0 = debint(conr * ri1j0, tx * torch.ones_like(ri1j0)) / torch.sqrt(atwt[pair_i1] * atwt[pair_j0])
    sig2_neq_all = ridotj * (ci0j0 + ci1j1 - ci0j1 - ci1j0) / (ri0i1 * rj0j1)
    sig2_neq = torch.zeros_like(natoms, dtype = pos.dtype, device = pos.device)
    sig2_neq = sig2_neq.scatter_add(0, pair_path_ids, sig2_neq_all)

    return conh * (sig2_eq + sig2_neq)

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
            total_loss = 0

            tot_edge = 0
            tot_path = 0
            tot_path_degen = 0
            tot_abs = 0
            for i, struct_i_flt in enumerate(label):
                struct_i = struct_i_flt.int()
                n_edge = batch.n_edge[i]
                n_path = batch.system_features["N_paths"][i]
                n_path_degen = batch.system_features["N_paths_degen"][i]
                
                edge_preds = prediction[tot_edge:(tot_edge + n_edge)]
                edge_path_inds = batch.system_features["edge_path_inds"][tot_path_degen:(tot_path_degen + n_path_degen)]
                edge_abs_inds = batch.system_features["abs_edge_inds"][tot_edge:(tot_edge + n_edge)]
                unique_abs_edge, inverse_abs_edge = torch.unique(edge_abs_inds, sorted = True, return_inverse = True)
                path_abs_inds = batch.system_features["abs_path_inds"][tot_path:(tot_path + n_path)]
                unique_abs_path, inverse_abs_path = torch.unique(path_abs_inds, sorted = True, return_inverse = True)
                edge_path_abs_inds = batch.system_features["abs_edge_path_inds"][tot_path_degen:(tot_path_degen + n_path_degen)]
                unique_abs_edge_path, inverse_abs_edge_path = torch.unique(edge_path_abs_inds, sorted = True, return_inverse = True)
                
                array_info_all = batch.system_features["array_info"][tot_path:(tot_path + n_path)]
                Reff_all = batch.system_features["Reffs"][tot_path:(tot_path + n_path)]
                m_a_all = batch.system_features["m_a"][tot_path:(tot_path + n_path)]
                m_s_all = batch.system_features["m_s"][tot_path:(tot_path + n_path)]
                rnorman_all = batch.system_features["rnorman"][tot_path:(tot_path + n_path)]
                degen_all = batch.system_features["degen"][tot_path:(tot_path + n_path)]

                structure_loss = 0
                for j in range(len(unique_abs_edge)):
                    abs_edge_mask = inverse_abs_edge == j
                    abs_path_mask = inverse_abs_path == j
                    abs_edge_path_mask = inverse_abs_edge_path == j
                    edge_path_mapping = edge_path_inds[abs_edge_path_mask]
                    preds_abs = edge_preds[abs_edge_mask]

                    degen_abs = degen_all[abs_path_mask]
                    Reff_abs = Reff_all[abs_path_mask] if config.avg_paths else Reff_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    m_a_abs = m_a_all[abs_path_mask] if config.avg_paths else m_a_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    m_s_abs = m_s_all[abs_path_mask] if config.avg_paths else m_s_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    rnorman_abs = rnorman_all[abs_path_mask] if config.avg_paths else rnorman_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    array_info_abs = array_info_all[abs_path_mask] if config.avg_paths else array_info_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    
                    if config.avg_paths:
                        segment_ids = torch.repeat_interleave(torch.arange(len(degen_abs), device = prediction.device), degen_abs.int())
                        pred_array_info = torch.zeros(
                            (len(degen_abs), edge_preds.shape[1]),
                            dtype=edge_preds.dtype,
                            device=edge_preds.device,
                        )

                        pred_array_info.scatter_add_(
                            0,
                            segment_ids[:, None].expand(-1, edge_preds.shape[1]),
                            edge_preds[edge_path_mapping]
                        )

                        pred_array_info /= degen_abs[:, None]
                    else:
                        pred_array_info = edge_preds[edge_path_mapping]

                    structure_loss = structure_loss + F.mse_loss(pred_array_info, array_info_abs, reduction = "mean")
                total_loss = total_loss + structure_loss / len(unique_abs_edge)
                tot_edge += n_edge
                tot_path += n_path
                tot_path_degen += n_path_degen
                tot_abs += len(unique_abs_edge)
            
            return total_loss / len(label)

        case _:
            assert_never(config)
