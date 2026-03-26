#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_from_ckpt_multistep_basic_compare_plots_v3_peratom.py
-----------------------------------------------------------
与 v2_user 相同，但 dq 图改为“每个原子单独一张图”：
python infer_from_ckpt_multistep_basic_compare_plots_v3_peratom_fixed_for_fixed4_v9_nature_ci_heatmap_separate_cbars.py \
  --train_script train_nextstep_joint_schemeA_multistep_split_multiAnchor_valBoundaryTail_ESgate_fixed_fixed4.py \
  --ckpt out_x_multistep_splitA/best_model.pt \
  --xyz Ag4.xyz \
  --dq_dirs mulliken_x mulliken_y mulliken_z --dq_mode x --dir x \
  --dmx dm-gauss_x.dat --dmy dm-gauss_x.dat --dmz dm-gauss_x.dat \
  --mmx mm-COM-gauss_x.dat --mmy mm-COM-gauss_x.dat --mmz mm-COM-gauss_x.dat \
  --dm_col 3 --mm_col 2 \
  --train_end_fs 24 --mode single --start_fs 24 --end_fs 32 \
  --device cpu --torch_threads 64 --outdir infer_fixed4 --plot
- mu/mm：仍然一张图（mu_true/mu_pred/mm_true/mm_pred）
- dq：每个原子 i 输出一张图 q{i}_truth_vs_pred.png

其余逻辑不变：open-loop rollout + 保存 CSV。

用法同 v2_user。
"""

import argparse
import importlib.util
import math
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


AU_TO_FS = 0.02418884326505


def _import_train_module(train_script_path: str):
    p = Path(train_script_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"train_script not found: {p}")
    mod_name = p.stem + "_imported_for_infer"
    spec = importlib.util.spec_from_file_location(mod_name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore
    return m


def _nearest_idx(t: np.ndarray, x: float) -> int:
    t = np.asarray(t, dtype=np.float32)
    return int(np.argmin(np.abs(t - float(x))))


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _read_xyz_positions(xyz_path: str) -> np.ndarray:
    """鲁棒读取第一帧 xyz 坐标（Å），只取第一帧 N 行坐标。"""
    with open(xyz_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_lines = [l.strip() for l in f.readlines()]

    n_atoms = None
    for l in raw_lines:
        if not l:
            continue
        try:
            n_atoms = int(l.split()[0])
        except Exception:
            n_atoms = None
        break

    coords = []
    for l in raw_lines:
        if not l:
            continue
        parts = l.split()
        if len(parts) < 4:
            continue
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except Exception:
            continue
        coords.append([x, y, z])
        if n_atoms is not None and len(coords) >= n_atoms:
            break

    if not coords:
        raise ValueError(f"failed to parse any coordinates from xyz: {xyz_path}")

    if n_atoms is not None and len(coords) != n_atoms:
        raise ValueError(
            f"xyz parse mismatch: header n_atoms={n_atoms} but parsed {len(coords)} coord lines. "
            f"Check xyz formatting / multi-frame / blank lines: {xyz_path}"
        )

    return np.asarray(coords, dtype=np.float32)


def _make_phi_rinv_from_xyz(xyz_path: str, device: torch.device, rcut: float = 10.0):
    """构造 HIP 需要的 (phi, rinv)。"""
    pos = _read_xyz_positions(xyz_path)  # (N,3)
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
    dist_ij = torch.cdist(pos_t, pos_t)  # (N,N)
    phi = torch.where(
        dist_ij < float(rcut),
        torch.cos((math.pi * dist_ij) / (2.0 * float(rcut))) ** 2,
        torch.zeros_like(dist_ij),
    )
    rinv = torch.where(dist_ij > 0, 1.0 / dist_ij, torch.zeros_like(dist_ij))
    return phi, rinv


def _load_dq_from_dir(dirname: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 dq_t.npy & time_fs.npy，返回 dq(N,T), t_fs(T,)"""
    d = Path(dirname)
    dq_path = d / "dq_t.npy"
    t_path = d / "time_fs.npy"
    if not dq_path.exists() or not t_path.exists():
        raise FileNotFoundError(f"missing dq_t.npy/time_fs.npy in {dirname}")
    dq = np.load(dq_path).astype(np.float32)
    t = np.load(t_path).astype(np.float32)
    if dq.ndim != 2:
        raise ValueError(f"dq_t.npy must be 2D, got {dq.shape}")
    # 兼容 (T,N) / (N,T)
    if dq.shape[0] == t.shape[0]:
        dq = dq.T
    return dq, t


def _load_col_dat(path: str, col0: int) -> Tuple[np.ndarray, np.ndarray]:
    """加载 dm/mm dat：第一列 time(au)->fs，col0 为 0-based 值列"""
    arr = np.loadtxt(path)
    t_fs = (arr[:, 0] * AU_TO_FS).astype(np.float32)
    y = arr[:, col0].astype(np.float32)
    return t_fs, y


def _interp_1d(t_src: np.ndarray, y_src: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """插值到 dq 的时间栅格"""
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_new = np.asarray(t_new, dtype=np.float64)
    if np.any(np.diff(t_src) < 0):
        order = np.argsort(t_src)
        t_src = t_src[order]
        y_src = y_src[order]
    return np.interp(t_new, t_src, y_src).astype(np.float32)


def _broadcastable_stats(mean, std):
    """把 dq_mean/dq_std 变成可广播到 (N,T) 的形状。

    允许以下形状：
    - scalar
    - (N,)
    - (N,1)
    - (1,N)  （部分训练脚本会为了 torch 广播存成这个形状）
    """
    m = _to_np(mean)
    s = _to_np(std)
    m = np.asarray(m, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)

    # ---- mean ----
    if m.ndim == 0:
        pass
    elif m.ndim == 1:
        # (N,) -> (N,1)
        m = m[:, None]
    elif m.ndim == 2:
        if m.shape[1] == 1:
            # (N,1)
            pass
        elif m.shape[0] == 1:
            # (1,N) -> (N,1)
            m = m.reshape(-1, 1)
        else:
            raise ValueError(f"unsupported dq_mean shape: {m.shape}")
    else:
        raise ValueError(f"unsupported dq_mean shape: {m.shape}")

    # ---- std ----
    if s.ndim == 0:
        pass
    elif s.ndim == 1:
        s = s[:, None]
    elif s.ndim == 2:
        if s.shape[1] == 1:
            pass
        elif s.shape[0] == 1:
            s = s.reshape(-1, 1)
        else:
            raise ValueError(f"unsupported dq_std shape: {s.shape}")
    else:
        raise ValueError(f"unsupported dq_std shape: {s.shape}")

    return m, s


def _ckpt_keymap(ckpt: Dict[str, Any]) -> Dict[str, str]:
    keys = set(ckpt.keys())
    out: Dict[str, str] = {}
    out["hip"] = "hip" if "hip" in keys else ("model_hip" if "model_hip" in keys else "")
    if not out["hip"]:
        raise KeyError(f"Cannot find HIP weights in ckpt keys: {sorted(keys)}")

    if "mu_gru" in keys:
        out["mu"] = "mu_gru"
    elif "gru_mu" in keys:
        out["mu"] = "gru_mu"
    else:
        raise KeyError(f"Cannot find mu GRU weights in ckpt keys: {sorted(keys)}")

    if "mm_gru" in keys:
        out["mm"] = "mm_gru"
    elif "gru_mm" in keys:
        out["mm"] = "gru_mm"
    else:
        raise KeyError(f"Cannot find mm GRU weights in ckpt keys: {sorted(keys)}")

    return out


@torch.no_grad()
def _rollout_openloop(
    hip,
    mu_gru,
    mm_gru,
    phi: torch.Tensor,
    rinv: torch.Tensor,
    dq_phys: np.ndarray,     # (N,T)
    t_fs: np.ndarray,        # (T,)
    mu_phys: np.ndarray,     # (T,)
    mm_phys: np.ndarray,     # (T,)
    start_fs: float,
    end_fs: float,
    nhist: int,
    dq_mean, dq_std,
    mu_mean: float, mu_std: float,
    mm_mean: float, mm_std: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:

    i0 = _nearest_idx(t_fs, start_fs)
    i1 = _nearest_idx(t_fs, end_fs)
    if i1 <= i0 + nhist:
        raise ValueError(f"rollout interval too short: start={start_fs}, end={end_fs}, nhist={nhist}")

    dq_mean_b, dq_std_b = _broadcastable_stats(dq_mean, dq_std)

    dqN = (dq_phys - dq_mean_b) / (dq_std_b + 1e-12)
    muN = (mu_phys - float(mu_mean)) / (float(mu_std) + 1e-12)
    mmN = (mm_phys - float(mm_mean)) / (float(mm_std) + 1e-12)

    times = t_fs[i0:i1 + 1].copy()
    T = len(times)
    N = dq_phys.shape[0]

    q_true = dq_phys[:, i0:i1 + 1].copy()
    mu_true = mu_phys[i0:i1 + 1].copy()
    mm_true = mm_phys[i0:i1 + 1].copy()

    q_pred = np.zeros((N, T), dtype=np.float32)
    mu_pred = np.zeros((T,), dtype=np.float32)
    mm_pred = np.zeros((T,), dtype=np.float32)

    # 初始化窗口：用真值填 nhist
    q_pred[:, :nhist] = dq_phys[:, i0:i0 + nhist]
    mu_pred[:nhist] = mu_phys[i0:i0 + nhist]
    mm_pred[:nhist] = mm_phys[i0:i0 + nhist]

    q_histN = torch.as_tensor(dqN[:, i0:i0 + nhist][None, :, :], dtype=torch.float32, device=device)  # (1,N,nhist)
    mu_histN = torch.as_tensor(muN[i0:i0 + nhist][None, :, None], dtype=torch.float32, device=device) # (1,nhist,1)
    mm_histN = torch.as_tensor(mmN[i0:i0 + nhist][None, :, None], dtype=torch.float32, device=device)

    # denorm
    if np.asarray(dq_mean_b).ndim == 2:
        dq_mean_1d = np.asarray(dq_mean_b, dtype=np.float32).squeeze(-1)
    else:
        dq_mean_1d = float(np.asarray(dq_mean_b))
    if np.asarray(dq_std_b).ndim == 2:
        dq_std_1d = np.asarray(dq_std_b, dtype=np.float32).squeeze(-1)
    else:
        dq_std_1d = float(np.asarray(dq_std_b))

    for k in range(nhist, T):
        q_nextN = hip(q_histN, phi, rinv)      # (1,N)
        mu_nextN = mu_gru(mu_histN)            # (1,)
        mm_nextN = mm_gru(mm_histN)            # (1,)

        q_nextN_np = _to_np(q_nextN)[0].astype(np.float32)
        q_next = q_nextN_np * dq_std_1d + dq_mean_1d

        mu_next = float(_to_np(mu_nextN)[0] * (float(mu_std) + 1e-12) + float(mu_mean))
        mm_next = float(_to_np(mm_nextN)[0] * (float(mm_std) + 1e-12) + float(mm_mean))

        q_pred[:, k] = q_next
        mu_pred[k] = mu_next
        mm_pred[k] = mm_next

        # open-loop：把自己预测塞回去
        q_histN = torch.cat([q_histN[:, :, 1:], q_nextN.unsqueeze(-1)], dim=-1)
        mu_histN = torch.cat([mu_histN[:, 1:, :], mu_nextN[:, None, None]], dim=1)
        mm_histN = torch.cat([mm_histN[:, 1:, :], mm_nextN[:, None, None]], dim=1)

    return {
        "time_fs": times,
        "q_true": q_true,
        "q_pred": q_pred,
        "mu_true": mu_true,
        "mu_pred": mu_pred,
        "mm_true": mm_true,
        "mm_pred": mm_pred,
    }


def _save_csv(path: Path, out: Dict[str, np.ndarray]):
    t = out["time_fs"]
    q_true = out["q_true"]
    q_pred = out["q_pred"]
    mu_true = out["mu_true"]
    mu_pred = out["mu_pred"]
    mm_true = out["mm_true"]
    mm_pred = out["mm_pred"]

    cols = [
        ("time_fs", t),
        ("mu_true", mu_true),
        ("mu_pred", mu_pred),
        ("mm_true", mm_true),
        ("mm_pred", mm_pred),
    ]
    for i in range(q_true.shape[0]):
        cols.append((f"q{i}_true", q_true[i]))
        cols.append((f"q{i}_pred", q_pred[i]))

    header = ",".join([c[0] for c in cols])
    data = np.vstack([c[1] for c in cols]).T
    np.savetxt(path, data, delimiter=",", header=header, comments="")



def _apply_nature_style():
    """Nature-like styling: closed box axes, inward ticks, clean typography."""
    if plt is None:
        return
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 180,
        "savefig.dpi": 600,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.2,
        "lines.markersize": 3,

        # ticks inward + show on all sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        # closed axes (box)
        "axes.spines.top": True,
        "axes.spines.right": True,

        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _plot_mu_mm(path: Path, tag: str, out: Dict[str, np.ndarray], train_end_fs: float):
    """Nature-like 2-panel line plot for dipole (mu) and mm, with empirical 95% CI bands."""
    if plt is None:
        return
    _apply_nature_style()

    t = out["time_fs"].astype(np.float32)

    def _empirical_ci(y_pred: np.ndarray, y_true: np.ndarray, k0: int):
        # Empirical sigma from rollout residuals on the predicted region (post-hoc CI).
        if len(y_pred) <= k0:
            sig = float(np.std(y_pred - y_true) + 1e-12)
        else:
            err = (y_pred[k0:] - y_true[k0:]).astype(np.float64)
            sig = float(np.sqrt(np.mean(err * err)) + 1e-12)  # RMSE as sigma proxy
        z = 1.96
        return y_pred - z * sig, y_pred + z * sig

    # Find index of train_end within plotted window
    k_train = int(np.argmin(np.abs(t - float(train_end_fs))))

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 3.25), sharex=True)

    # --- mu ---
    ax = axes[0]
    mu_true = out["mu_true"].astype(np.float32)
    mu_pred = out["mu_pred"].astype(np.float32)
    lo, hi = _empirical_ci(mu_pred, mu_true, k_train)

    ax.plot(t, mu_true, color="#d62728", label="Truth")  # red
    ax.plot(t, mu_pred, color="#1f77b4", label="Pred")   # blue
    ax.fill_between(t, lo, hi, color="#1f77b4", alpha=0.15, linewidth=0)

    ax.axvline(train_end_fs, linestyle="--", linewidth=1.0, color="0.3")
    ax.set_ylabel(r"$\mu$ (a.u.)")
    ax.tick_params(which="both", top=True, right=True, direction="in")

    # legend outside (no遮挡)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, handlelength=2.0)

    # --- mm ---
    ax = axes[1]
    mm_true = out["mm_true"].astype(np.float32)
    mm_pred = out["mm_pred"].astype(np.float32)
    lo, hi = _empirical_ci(mm_pred, mm_true, k_train)

    ax.plot(t, mm_true, color="#d62728", label="Truth")
    ax.plot(t, mm_pred, color="#1f77b4", label="Pred")
    ax.fill_between(t, lo, hi, color="#1f77b4", alpha=0.15, linewidth=0)

    ax.axvline(train_end_fs, linestyle="--", linewidth=1.0, color="0.3")
    ax.set_ylabel(r"$m$ (a.u.)")
    ax.set_xlabel("Time (fs)")
    ax.tick_params(which="both", top=True, right=True, direction="in")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, handlelength=2.0)

    fig.tight_layout(pad=0.6)
    fig.savefig(path, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def _plot_dq_per_atom(outdir: Path, tag: str, out: Dict[str, np.ndarray], train_end_fs: float):
    """Per-atom dq truth vs pred with Nature-like styling + empirical 95% CI band."""
    if plt is None:
        return
    _apply_nature_style()

    t = out["time_fs"].astype(np.float32)
    q_true = out["q_true"].astype(np.float32)
    q_pred = out["q_pred"].astype(np.float32)
    N = q_true.shape[0]

    k_train = int(np.argmin(np.abs(t - float(train_end_fs))))

    def _empirical_ci_1d(y_pred: np.ndarray, y_true: np.ndarray):
        if len(y_pred) <= k_train:
            sig = float(np.std(y_pred - y_true) + 1e-12)
        else:
            err = (y_pred[k_train:] - y_true[k_train:]).astype(np.float64)
            sig = float(np.sqrt(np.mean(err * err)) + 1e-12)
        z = 1.96
        return y_pred - z * sig, y_pred + z * sig

    for i in range(N):
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 2.25))
        lo, hi = _empirical_ci_1d(q_pred[i], q_true[i])

        ax.plot(t, q_true[i], color="#d62728", label="Truth")  # red
        ax.plot(t, q_pred[i], color="#1f77b4", label="Pred")   # blue
        ax.fill_between(t, lo, hi, color="#1f77b4", alpha=0.15, linewidth=0)

        ax.axvline(train_end_fs, linestyle="--", linewidth=1.0, color="0.3")
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel(r"$\Delta q$ (e)")
        ax.tick_params(which="both", top=True, right=True, direction="in")

        # legend outside
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, handlelength=2.0)

        fig.tight_layout(pad=0.5)

        png_path = outdir / f"{tag}_dq_atom{i}.png"
        fig.savefig(png_path, bbox_inches="tight")
        try:
            fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)


def _plot_dq_heatmap(outdir: Path, tag: str, out: Dict[str, np.ndarray], train_end_fs: float):
    """Combined dq heatmaps: Truth (red), Pred (blue), Error (orange) with light palettes."""
    if plt is None:
        return
    _apply_nature_style()

    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    t = out["time_fs"].astype(np.float32)
    q_true = out["q_true"].astype(np.float32)  # (N,T)
    q_pred = out["q_pred"].astype(np.float32)
    err = (q_pred - q_true).astype(np.float32)
    N, _ = q_true.shape

    # light diverging colormaps (negative->very light tint, zero->white, positive->strong tint)
    cmap_red = LinearSegmentedColormap.from_list("red_light_div", ["#fde0dd", "#ffffff", "#de2d26"])
    cmap_blue = LinearSegmentedColormap.from_list("blue_light_div", ["#deebf7", "#ffffff", "#3182bd"])
    cmap_org = LinearSegmentedColormap.from_list("org_light_div", ["#fee6ce", "#ffffff", "#e6550d"])

    vmax = float(np.nanmax(np.abs(q_true)))
    vmax = max(vmax, float(np.nanmax(np.abs(q_pred))), 1e-6)
    evmax = float(np.nanmax(np.abs(err)))
    evmax = max(evmax, 1e-6)

    norm_q = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    norm_e = TwoSlopeNorm(vcenter=0.0, vmin=-evmax, vmax=evmax)

    fig = plt.figure(figsize=(3.6, 4.35))

    # Keep all three heatmaps perfectly aligned by reserving a dedicated colorbar column.
    # 3 rows (Truth/Pred/Error) × 2 columns (image / colorbar).
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.12,
        hspace=0.15,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    axes = [ax0, ax1, ax2]

    cax_truth = fig.add_subplot(gs[0, 1])
    cax_pred  = fig.add_subplot(gs[1, 1])
    cax_e     = fig.add_subplot(gs[2, 1])
    def _imshow(ax, data, cmap, norm, title):
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[float(t[0]), float(t[-1]), 0, N],
            cmap=cmap,
            norm=norm,
        )
        ax.axvline(train_end_fs, linestyle="--", linewidth=1.0, color="0.3")
        ax.set_ylabel("Atom")
        ax.text(0.01, 0.93, title, transform=ax.transAxes, fontsize=8, va="top")
        # closed frame + inward ticks (Nature-style), keep top/right ticks too
        ax.tick_params(which="both", top=True, right=True, direction="in")
        for sp in ax.spines.values():
            sp.set_visible(True)
        return im

    im0 = _imshow(axes[0], q_true, cmap_red, norm_q, "Truth")
    im1 = _imshow(axes[1], q_pred, cmap_blue, norm_q, "Pred")
    im2 = _imshow(axes[2], err, cmap_org, norm_e, "Error")
    axes[2].set_xlabel("Time (fs)")

    # Colorbars (separate for Truth/Pred to match their colormaps; fixed column => aligned widths)
    cbar_t = fig.colorbar(im0, cax=cax_truth)
    cbar_t.set_label(r"$\Delta q$ (e)")
    cbar_t.ax.tick_params(direction="in")

    cbar_p = fig.colorbar(im1, cax=cax_pred)
    cbar_p.set_label(r"$\Delta q$ (e)")
    cbar_p.ax.tick_params(direction="in")

    cbar_e = fig.colorbar(im2, cax=cax_e)
    cbar_e.set_label(r"Error (e)")
    cbar_e.ax.tick_params(direction="in")

    fig.tight_layout(pad=0.6)

    png_path = outdir / f"{tag}_dq_heatmap.png"
    fig.savefig(png_path, bbox_inches="tight")
    try:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--xyz", type=str, required=True)

    ap.add_argument("--dq_dirs", nargs=3, required=True)
    ap.add_argument("--dq_mode", type=str, default="x", choices=["x", "y", "z"])
    ap.add_argument("--dir", type=str, default="x", choices=["x", "y", "z"])

    ap.add_argument("--dmx", type=str, required=True)
    ap.add_argument("--dmy", type=str, required=True)
    ap.add_argument("--dmz", type=str, required=True)
    ap.add_argument("--mmx", type=str, required=True)
    ap.add_argument("--mmy", type=str, required=True)
    ap.add_argument("--mmz", type=str, required=True)
    ap.add_argument("--dm_col", type=int, default=3)  # 1-based
    ap.add_argument("--mm_col", type=int, default=2)  # 1-based

    ap.add_argument("--train_end_fs", type=float, default=24.0)
    ap.add_argument("--pred_end_fs", type=float, default=32.0)

    ap.add_argument("--mode", type=str, default="single", choices=["single", "boundary_test"])
    ap.add_argument("--start_fs", type=float, default=24.0)
    ap.add_argument("--end_fs", type=float, default=32.0)

    ap.add_argument("--val_start_fs", type=float, default=24.0)
    ap.add_argument("--val_end_fs", type=float, default=28.0)
    ap.add_argument("--test_start_fs", type=float, default=28.0)
    ap.add_argument("--test_end_fs", type=float, default=32.0)

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--torch_threads", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="infer_basic_out")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(int(args.torch_threads))
        except Exception:
            pass
    print(f"[Torch] num_threads={torch.get_num_threads()}")

    device = torch.device(args.device)

    train_mod = _import_train_module(args.train_script)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    keymap = _ckpt_keymap(ckpt)
    meta = ckpt.get("meta", {}) or {}

    # === 从 ckpt 里取训练时的超参数（兼容 old/new 字段）===
    saved_args = meta.get("args", None)
    if saved_args is None:
        saved_args = meta.get("args_dict", {}) or {}
    else:
        saved_args = saved_args or {}

    # nhist：优先 ckpt 顶层字段，其次 args 里的 nhist
    nhist = int(meta.get("nhist", meta.get("n_hist", saved_args.get("nhist", 12))))

    # === dq ===
    dq_dir_use = {"x": args.dq_dirs[0], "y": args.dq_dirs[1], "z": args.dq_dirs[2]}[args.dq_mode]
    dq_phys, t_fs = _load_dq_from_dir(dq_dir_use)

    # 统一用“相对时间轴”（与训练脚本一致）：把时间原点移动到第一帧
    t0 = float(t_fs[0])
    t_fs = (t_fs - t0).astype(np.float32)

    # === mu/mm：插值到 dq 的时间栅格 ===
    dm_idx0 = args.dm_col - 1
    mm_idx0 = args.mm_col - 1

    t_mx, mu_x = _load_col_dat(args.dmx, dm_idx0)
    t_my, mu_y = _load_col_dat(args.dmy, dm_idx0)
    t_mz, mu_z = _load_col_dat(args.dmz, dm_idx0)
    t_kx, mm_x = _load_col_dat(args.mmx, mm_idx0)
    t_ky, mm_y = _load_col_dat(args.mmy, mm_idx0)
    t_kz, mm_z = _load_col_dat(args.mmz, mm_idx0)

    mu_map = {"x": (t_mx, mu_x), "y": (t_my, mu_y), "z": (t_mz, mu_z)}
    mm_map = {"x": (t_kx, mm_x), "y": (t_ky, mm_y), "z": (t_kz, mm_z)}
    t_mu, mu_raw = mu_map[args.dir]
    t_mm, mm_raw = mm_map[args.dir]

    # 同样改为相对时间轴，避免 dq 与 dm/mm 原点不同导致插值错位
    t_mu = (t_mu - float(t_mu[0])).astype(np.float32)
    t_mm = (t_mm - float(t_mm[0])).astype(np.float32)

    mu_phys = _interp_1d(t_mu, mu_raw, t_fs)
    mm_phys = _interp_1d(t_mm, mm_raw, t_fs)

    # === norm stats ===
    i_end = _nearest_idx(t_fs, args.train_end_fs)
    if "dq_mean" in meta and "dq_std" in meta:
        dq_mean, dq_std = meta["dq_mean"], meta["dq_std"]
    else:
        dq_mean = float(dq_phys[:, :i_end + 1].mean())
        dq_std = float(dq_phys[:, :i_end + 1].std() + 1e-12)

    if "mu_mean" in meta and "mu_std" in meta and "mm_mean" in meta and "mm_std" in meta:
        mu_mean, mu_std = float(meta["mu_mean"]), float(meta["mu_std"])
        mm_mean, mm_std = float(meta["mm_mean"]), float(meta["mm_std"])
    else:
        mu_mean, mu_std = float(mu_phys[:i_end + 1].mean()), float(mu_phys[:i_end + 1].std() + 1e-12)
        mm_mean, mm_std = float(mm_phys[:i_end + 1].mean()), float(mm_phys[:i_end + 1].std() + 1e-12)

    # === model hparams：必须与训练一致，否则 load_state_dict 可能成功但输入特征不一致导致结果异常 ===
    hip_features = int(saved_args.get("hip_features", 64))
    hip_layers = int(saved_args.get("hip_layers", 4))
    hip_dropout = float(saved_args.get("hip_dropout", 0.0))
    gru_hidden = int(saved_args.get("gru_hidden", 128))
    gru_layers = int(saved_args.get("gru_layers", 2))
    gru_dropout = float(saved_args.get("gru_dropout", 0.0))
    rcut = float(saved_args.get("rcut", 10.0))
    print(f"[Infer] nhist={nhist} hip_features={hip_features} hip_layers={hip_layers} gru_hidden={gru_hidden} gru_layers={gru_layers} rcut={rcut}")

    HIPClass = getattr(train_mod, "HIPNNChargesBatch", None)
    GRUClass = getattr(train_mod, "GRUHead1D", None)
    if HIPClass is None or GRUClass is None:
        raise RuntimeError("训练脚本中未找到 HIPNNChargesBatch / GRUHead1D。请确认训练脚本版本。")

    hip = HIPClass(n_hist=nhist, n_features=hip_features, n_interaction=hip_layers, dropout=hip_dropout).to(device)
    mu_gru = GRUClass(in_dim=1, hidden=gru_hidden, num_layers=gru_layers, dropout=gru_dropout).to(device)
    mm_gru = GRUClass(in_dim=1, hidden=gru_hidden, num_layers=gru_layers, dropout=gru_dropout).to(device)

    hip.load_state_dict(ckpt[keymap["hip"]], strict=True)
    mu_gru.load_state_dict(ckpt[keymap["mu"]], strict=True)
    mm_gru.load_state_dict(ckpt[keymap["mm"]], strict=True)
    hip.eval(); mu_gru.eval(); mm_gru.eval()

    phi, rinv = _make_phi_rinv_from_xyz(args.xyz, device=device, rcut=rcut)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _run_and_save(tag: str, s: float, e: float):
        out = _rollout_openloop(
            hip, mu_gru, mm_gru, phi, rinv,
            dq_phys, t_fs, mu_phys, mm_phys,
            start_fs=s, end_fs=e, nhist=nhist,
            dq_mean=dq_mean, dq_std=dq_std,
            mu_mean=mu_mean, mu_std=mu_std,
            mm_mean=mm_mean, mm_std=mm_std,
            device=device,
        )
        _save_csv(outdir / f"rollout_{tag}.csv", out)
        if args.plot:
            _plot_mu_mm(outdir / f"{tag}_mu_mm.png", tag, out, train_end_fs=args.train_end_fs)
            _plot_dq_per_atom(outdir, tag, out, train_end_fs=args.train_end_fs)
            _plot_dq_heatmap(outdir, tag, out, train_end_fs=args.train_end_fs)

    if args.mode == "single":
        tag = f"{args.start_fs:g}to{args.end_fs:g}fs"
        _run_and_save(tag, args.start_fs, args.end_fs)
        print(f"[Saved] {outdir / f'rollout_{tag}.csv'}")
    else:
        tag1 = f"boundary_{args.val_start_fs:g}to{args.val_end_fs:g}fs"
        tag2 = f"test_{args.test_start_fs:g}to{args.test_end_fs:g}fs"
        _run_and_save(tag1, args.val_start_fs, args.val_end_fs)
        _run_and_save(tag2, args.test_start_fs, args.test_end_fs)
        print(f"[Saved] {outdir / f'rollout_{tag1}.csv'}")
        print(f"[Saved] {outdir / f'rollout_{tag2}.csv'}")

    print(f"[Out] {outdir.resolve()}")
    print("[Done]")


if __name__ == "__main__":
    main()
