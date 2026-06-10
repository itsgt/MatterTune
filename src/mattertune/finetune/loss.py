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
    avg_paths: bool = False
    predict_deltar: bool = True
    predict_third: bool = True
    kw: int = 0
     

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
    y[:, 1:] = x[:, 1:] * (torch.exp(xt) + 1) / (torch.exp(xt) - 1)
    return torch.trapz(y, x, dim=1)  # (B,)

def debint(r, t, N = 100):
    x = torch.linspace(0, 1, N, device = t.device, dtype = t.dtype)  # (N,)
    x = x.unsqueeze(0)  # (1, N)
    r = r.unsqueeze(1)  # (B, 1)
    t = t.unsqueeze(1)  # (B, 1)
    y = (2.0 / t) * torch.ones_like(x)  # (B, N)
    xt = x[:, 1:] * t  # (B, N-1)
    rx = r * x[:, 1:]  # (B, N-1)
    y[:, 1:] = (x[:, 1:] * torch.sinc(rx / torch.pi) * 
        (torch.exp(xt) + 1) / (torch.exp(xt) - 1))
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
    sig2_eq_all = sign * (diz / atwt[i0] + diz / atwt[i1] - 2 * ci0i1) / 2
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
    ci0j0 = debint(conr * ri0j0, tx * np.ones_like(ri0j0)) / torch.sqrt(atwt[pair_i0] * atwt[pair_j0])
    ci1j1 = debint(conr * ri1j1, tx * np.ones_like(ri1j1)) / torch.sqrt(atwt[pair_i1] * atwt[pair_j1])
    ci0j1 = debint(conr * ri0j1, tx * np.ones_like(ri0j1)) / torch.sqrt(atwt[pair_i0] * atwt[pair_j1])
    ci1j0 = debint(conr * ri1j0, tx * np.ones_like(ri1j0)) / torch.sqrt(atwt[pair_i1] * atwt[pair_j0])
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
            k1s = torch.arange(0.0, 16.00001, 0.05, device = prediction.device)
            k2s = k1s ** 2
            kws = k1s ** config.kw
            sl = 60
            sr = -80
            k_feff = batch.system_features["feff_k"][:(len(batch.system_features["feff_k"]) // len(label))]
            sim_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            exp_chis = torch.zeros((len(label), len(k1s[sl:sr])), device = prediction.device)
            tot_edge = 0
            tot_path = 0
            tot_path_ms = 0
            tot_path_legs_ms = 0
            tot_path_degen = 0
            tot_abs = 0
            for i, struct_i_flt in enumerate(label):
                struct_i = struct_i_flt.int()
                n_edge = batch.n_edge[i]
                n_path = batch.system_features["N_paths"][i]
                n_path_ms = batch.system_features["N_paths_MS"][i]
                n_path_leg_ms = batch.system_features["N_paths_leg_MS"][i]
                n_path_degen = batch.system_features["N_paths_degen"][i]
                
                edge_preds = prediction[tot_edge:(tot_edge + n_edge)]
                edge_path_inds = batch.system_features["edge_path_inds"][tot_path_degen:(tot_path_degen + n_path_degen)]
                edge_abs_inds = batch.system_features["abs_edge_inds"][tot_edge:(tot_edge + n_edge)]
                unique_abs_edge, inverse_abs_edge = torch.unique(edge_abs_inds, sorted = True, return_inverse = True)
                path_abs_inds = batch.system_features["abs_path_inds"][tot_path:(tot_path + n_path)]
                unique_abs_path, inverse_abs_path = torch.unique(path_abs_inds, sorted = True, return_inverse = True)
                path_abs_inds_ms = batch.system_features["abs_path_inds_ms"][tot_path_ms:(tot_path_ms + n_path_ms)]
                unique_abs_path_ms, inverse_abs_path_ms = torch.unique(path_abs_inds_ms, sorted = True, return_inverse = True)
                path_leg_abs_inds_ms = batch.system_features["abs_path_leg_inds_ms"][tot_path_legs_ms:(tot_path_legs_ms + n_path_leg_ms)]
                unique_abs_path_leg_ms, inverse_abs_path_leg_ms = torch.unique(path_leg_abs_inds_ms, sorted = True, return_inverse = True)
                edge_path_abs_inds = batch.system_features["abs_edge_path_inds"][tot_path_degen:(tot_path_degen + n_path_degen)]
                unique_abs_edge_path, inverse_abs_edge_path = torch.unique(edge_path_abs_inds, sorted = True, return_inverse = True)
                
                amp_all = batch.system_features["amp"][tot_path:(tot_path + n_path)]
                pha_all = batch.system_features["pha"][tot_path:(tot_path + n_path)]
                rep_all = batch.system_features["rep"][tot_path:(tot_path + n_path)]
                lam_all = batch.system_features["lam"][tot_path:(tot_path + n_path)]
                Reff_all = batch.system_features["Reffs"][tot_path:(tot_path + n_path)]
                T_Ds_all = batch.system_features["T_Ds"][tot_path:(tot_path + n_path)]
                m_a_all = batch.system_features["m_a"][tot_path:(tot_path + n_path)]
                m_s_all = batch.system_features["m_s"][tot_path:(tot_path + n_path)]
                rnorman_all = batch.system_features["rnorman"][tot_path:(tot_path + n_path)]
                degen_all = batch.system_features["degen"][tot_path:(tot_path + n_path)]

                amp_ms_all = batch.system_features["amp_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                pha_ms_all = batch.system_features["pha_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                rep_ms_all = batch.system_features["rep_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                lam_ms_all = batch.system_features["lam_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                Reff_ms_all = batch.system_features["Reffs_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                nlegs_ms_all = batch.system_features["nlegs_MS"][tot_path_ms:(tot_path_ms + n_path_ms)]
                pos_ms_all = batch.system_features["pos_MS"][tot_path_legs_ms:(tot_path_legs_ms + n_path_leg_ms), :]
                atwt_ms_all = batch.system_features["atwt_MS"][tot_path_legs_ms:(tot_path_legs_ms + n_path_leg_ms)]

                chi = torch.zeros((len(unique_abs_edge), len(k1s[sl:sr])), device = prediction.device)
                for j in range(len(unique_abs_edge)):
                    abs_edge_mask = inverse_abs_edge == j
                    abs_path_mask = inverse_abs_path == j
                    abs_path_mask_ms = inverse_abs_path_ms == j
                    abs_edge_path_mask = inverse_abs_edge_path == j
                    abs_edge_path_leg_mask = inverse_abs_path_leg_ms == j

                    preds_abs = edge_preds[abs_edge_mask]
                    ΔE0 = batch.system_features["energy_edges"][tot_abs + j]
                    assert k2s[sl] - ΔE0 > 0, f'{torch.tanh(preds_abs[:, 0].mean())} {ΔE0}' 
                    q = torch.sqrt(k2s[sl:sr] - ΔE0)

                    degen_abs = degen_all[abs_path_mask]
                    Reff_abs = Reff_all[abs_path_mask] if config.avg_paths else Reff_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    T_Ds_abs = T_Ds_all[abs_path_mask] if config.avg_paths else T_Ds_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    m_a_abs = m_a_all[abs_path_mask] if config.avg_paths else m_a_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    m_s_abs = m_s_all[abs_path_mask] if config.avg_paths else m_s_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    rnorman_abs = rnorman_all[abs_path_mask] if config.avg_paths else rnorman_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    amp_abs = amp_all[abs_path_mask] if config.avg_paths else amp_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    pha_abs = pha_all[abs_path_mask] if config.avg_paths else pha_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    rep_abs = rep_all[abs_path_mask] if config.avg_paths else rep_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    lam_abs = lam_all[abs_path_mask] if config.avg_paths else lam_all[abs_path_mask].repeat_interleave(degen_abs.int(), dim = 0)
                    
                    edge_path_mapping = edge_path_inds[abs_edge_path_mask]
                    
                    if config.avg_paths:
                        segment_ids = torch.repeat_interleave(torch.arange(len(degen_abs), device = prediction.device), degen_abs.int())
                        c2_pred = torch.zeros(len(degen_abs), dtype = edge_preds[:, 1].dtype, device = prediction.device)
                        c2_pred = c2_pred.scatter_add(0, segment_ids, edge_preds[:, 1][edge_path_mapping])
                        c2_pred = c2_pred / degen_abs
                        if config.predict_deltar:
                            c1_pred = torch.zeros(len(degen_abs), dtype = edge_preds[:, 0].dtype, device = prediction.device)
                            c1_pred = c1_pred.scatter_add(0, segment_ids, edge_preds[:, 0][edge_path_mapping])
                            c1_pred = c1_pred / degen_abs
                        else:
                            c1_pred = torch.zeros_like(c2_pred, device = prediction.device)
                        
                        if config.predict_third:
                            c3_pred = torch.zeros(len(degen_abs), dtype = edge_preds[:, 2].dtype, device = prediction.device)
                            c3_pred = c3_pred.scatter_add(0, segment_ids, edge_preds[:, 2][edge_path_mapping])
                            c3_pred = c3_pred / degen_abs
                        else:
                            c3_pred = torch.zeros_like(c2_pred, device = prediction.device)
                    else:
                        c2_pred = edge_preds[:, 1][edge_path_mapping]
                        c1_pred = edge_preds[:, 0][edge_path_mapping] if config.predict_deltar else torch.zeros_like(c2_pred, device = prediction.device)
                        c3_pred = edge_preds[:, 2][edge_path_mapping] if config.predict_third else torch.zeros_like(c2_pred, device = prediction.device)

                    TD_T = (1 / 3) + 3 * torch.sigmoid(c2_pred) # Debye Temp between 100-1000K at room temp
                    sigma2 = sigma2_debye_SS(T_Ds_abs / 300, Reff_abs, m_a_abs, m_s_abs, rnorman_abs, 300. * torch.ones_like(Reff_abs))

                    chi_paths_SS = calc_chi_batch(q, 
                        0.15 * torch.tanh(c1_pred), 
                        sigma2, 
                        0.005 * torch.tanh(c3_pred), 
                        torch.zeros_like(c1_pred),
                        interp_soft_adaptive(q, k_feff, amp_abs), 
                        interp_soft_adaptive(q, k_feff, pha_abs), 
                        interp_soft_adaptive(q, k_feff, rep_abs), 
                        interp_soft_adaptive(q, k_feff, lam_abs),
                        Reff_abs, 0.0
                    )
                    if config.avg_paths:
                        chi_paths_SS = degen_abs.unsqueeze(-1) * chi_paths_SS

                    nleg_ms_abs = nlegs_ms_all[abs_path_mask_ms].int()
                    Reff_ms_abs = Reff_ms_all[abs_path_mask_ms]
                    amp_ms_abs = amp_ms_all[abs_path_mask_ms] 
                    pha_ms_abs = pha_ms_all[abs_path_mask_ms]
                    rep_ms_abs = rep_ms_all[abs_path_mask_ms]
                    lam_ms_abs = lam_ms_all[abs_path_mask_ms]
                    pos_ms_abs = pos_ms_all[abs_edge_path_leg_mask, :]
                    atwt_ms_abs = atwt_ms_all[abs_edge_path_leg_mask, :]

                    sigma2_MS = sigma2_debye_MS(
                        batch.system_features["T_Ds"][0] * torch.ones_like(Reff_ms_abs), 
                        nleg_ms_abs, 
                        pos_ms_abs, 
                        atwt_ms_abs, 
                        batch.system_features["rnorman"][0] * torch.ones_like(Reff_ms_abs), 
                        300.)

                    chi_paths_MS = calc_chi_batch(q, 
                        torch.zeros_like(sigma2_MS), 
                        sigma2_MS, 
                        torch.zeros_like(sigma2_MS), 
                        torch.zeros_like(sigma2_MS),
                        interp_soft_adaptive(q, k_feff, amp_ms_abs), 
                        interp_soft_adaptive(q, k_feff, pha_ms_abs), 
                        interp_soft_adaptive(q, k_feff, rep_ms_abs), 
                        interp_soft_adaptive(q, k_feff, lam_ms_abs),
                        Reff_ms_abs, 0.0
                    )
                    chi[j] = torch.sum(batch.system_features["degen_MS"] * chi_paths_MS + chi_paths_SS, dim = 0)
                tot_edge += n_edge
                tot_path += n_path
                tot_path_ms += n_path_ms
                tot_path_degen += n_path_degen
                tot_path_legs_ms += n_path_leg_ms
                tot_abs += len(unique_abs_edge)
            
                sim_chis[i] = kws[sl:sr] * torch.mean(chi, dim = 0)
                exp_chis[i] = (kws[sl:sr] * config.exp_spectra[struct_i][sl:sr])
            return (1 - F.cosine_similarity(sim_chis, exp_chis, dim = 1)).mean()

        case _:
            assert_never(config)
