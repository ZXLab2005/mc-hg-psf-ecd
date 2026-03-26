#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_nextstep_joint_schemeA_multistep_resume_fastbatch.py

在 train_nextstep_joint_schemeA_multistep_resume.py 的基础上做“算法级加速”的版本：
1) **K-step unroll 的 loss 只 backward 一次**（loss_total.backward() 不在 K-loop 内）
2) **训练/验证窗口一次性预张量化**（不在训练循环里反复 torch.tensor(...)）
3) HIPNN 支持 **batch 维度**，把“样本循环”挪到张量维度上（CPU 上速度提升通常很明显）
4) 预计算与 dist_ij 无关的 phi / rinv（减少 InteractionLayer 内重复计算）
python train_nextstep_joint_schemeA_multistep_split_multiAnchor_valBoundaryTail_ESgate_fixed_fixed4.py \
  --xyz Ag4.xyz \
  --dq_dirs mulliken_x mulliken_y mulliken_z --dq_mode x --dir x \
  --dmx dm-gauss_x.dat --dmy dm-gauss_x.dat --dmz dm-gauss_x.dat \
  --mmx mm-COM-gauss_x.dat --mmy mm-COM-gauss_x.dat --mmz mm-COM-gauss_x.dat \
  --dm_col 3 --mm_col 2 \
  --tmin 0.0 --nhist 12 --device cpu \
  --epochs 10000 --batch 32 \
  --w_q 1.0 --w_mu 0.5 --w_mm 0.5 \
  --alpha_qsum 1e-4 --lambda_mu_consistency 1e-2\
  --hip_features 64 --hip_layers 4 --hip_dropout 0.0 \
  --gru_hidden 128 --gru_layers 2 --gru_dropout 0.0 \
  --weight_decay 0 \
  --val_frac 0.2 --patience 400 --min_delta 1e-5 \
  --lr 1e-4 --lr_step 200 --lr_gamma 0.7 \
  --train_end_fs 24 --pred_end_fs 32 \
  --torch_threads 64 \
  --outdir out_x_multistep_splitA \
  --init_ckpt out_x_tuned/best_model.pt \
  --time_stratify --time_bins 10 --time_focus_start_fs 18 --time_focus_boost 2.0 \
  --unroll_k 16 \
  --ss_start 1.0 --ss_end 0.0 --ss_decay_epochs 1800 \
  --val_rollout --bptt --k_gamma 1.05 \
  --dq_norm_mode global \
  --dq_diff_w 0.3 --dq_corr_w 0.01 --dq_var_w 0.2 --dq_rms_step_w 0.2 --dq_mean_step_w 0.05 \
  --val_start_fs 24 --val_end_fs 28 \
  --test_start_fs 28 --test_end_fs 32 \
  --save_val_compare \
  --es_warmup_epochs 300 \
  --es_tf_threshold 0.6
输入/输出格式保持不变：
- 输入：--xyz, --dq_dirs, --dmx/dmy/dmz, --mmx/mmy/mmz 等完全一致
- 输出：log_{dir}.csv、best_model.pt、traj_mu_m_{dir}_0to{train_end}fs.csv、pred_mu_m_{dir}_{train_end}to{pred_end}fs.csv
"""

import argparse, os, math, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
#   Sampling helper: time-stratified mini-batch (train-only)
# ============================================================
def stratified_batch_indices(pool_idx: np.ndarray,
                             sample_times: np.ndarray,
                             batch_size: int,
                             n_bins: int,
                             t_min: float,
                             t_max: float,
                             focus_start: float,
                             focus_boost: float,
                             rng: np.random.Generator) -> np.ndarray:
    """Pick a mini-batch with (roughly) uniform coverage over time, optionally
    oversampling later-time bins (>= focus_start). Indices are drawn **only**
    from pool_idx (which must already be train-only, <= train_end_fs)."""
    pool_idx = np.asarray(pool_idx, dtype=np.int64)
    if batch_size >= len(pool_idx):
        return pool_idx.copy()

    n_bins = max(1, int(n_bins))
    edges = np.linspace(t_min, t_max + 1e-9, n_bins + 1, dtype=np.float64)

    t_pool = sample_times[pool_idx].astype(np.float64)
    bin_id = np.clip(np.digitize(t_pool, edges) - 1, 0, n_bins - 1)

    bins = [pool_idx[bin_id == b] for b in range(n_bins)]
    non_empty = [b for b in range(n_bins) if len(bins[b]) > 0]
    if not non_empty:
        return rng.choice(pool_idx, size=batch_size, replace=False)

    base = batch_size // len(non_empty)
    counts = {b: base for b in non_empty}
    remaining = batch_size - base * len(non_empty)

    if remaining > 0:
        centers = 0.5 * (edges[:-1] + edges[1:])
        w = np.array([focus_boost if centers[b] >= focus_start else 1.0 for b in non_empty], dtype=np.float64)
        w = w / w.sum()
        extra_bins = rng.choice(np.array(non_empty, dtype=np.int64), size=remaining, replace=True, p=w)
        for b in extra_bins:
            counts[int(b)] += 1

    picked = []
    for b, c in counts.items():
        arr = bins[b]
        if len(arr) >= c:
            picked.append(rng.choice(arr, size=c, replace=False))
        else:
            picked.append(rng.choice(arr, size=c, replace=True))
    out = np.concatenate(picked, axis=0)
    rng.shuffle(out)
    return out[:batch_size]


# ============================================================
#   1) IO helpers
# ============================================================

def load_xyz(xyz_file: str):
    atoms = []
    coords = []
    with open(xyz_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(atoms), np.array(coords, dtype=np.float32)


def load_dm_file(dmfile: str, col_idx0: int):
    """Load (time_fs, values) from GPAW dm/mm dat file. col_idx0 is 0-based."""
    arr = np.loadtxt(dmfile)
    t_au = arr[:, 0]
    au_to_fs = 0.02418884326505
    t_fs = t_au * au_to_fs
    values = arr[:, col_idx0]
    return t_fs.astype(np.float32), values.astype(np.float32)


def charge_cons_penalty(q_pred_real: torch.Tensor) -> torch.Tensor:
    """Penalty enforcing total charge conservation.

    Args:
        q_pred_real: (B, N) de-normalized charges in physical units.
    Returns:
        scalar tensor: mean_b (sum_i q_i)^2
    """
    # per-sample total charge (B,)
    qsum = q_pred_real.sum(dim=1)
    return torch.mean(qsum * qsum)


# ============================================================
#   2) HIPNN (batch version)
# ============================================================

class InteractionLayerBatch(nn.Module):
    def __init__(self, n_features: int, n_sensitivity: int = 8, dropout: float = 0.0):
        super().__init__()
        self.nf = n_features
        self.ns = n_sensitivity

        # (mu, sigma) in 1/r space
        self.sens = nn.Parameter(torch.randn(self.ns, 2) * 0.1)

        self.edge = nn.Linear(self.ns, self.nf, bias=False)
        self.self_lin = nn.Linear(self.nf, self.nf)

        self.act = nn.Softplus()
        self.ln = nn.LayerNorm(self.nf)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat: torch.Tensor, phi: torch.Tensor, rinv: torch.Tensor):
        """
        feat: (B, N, nf)
        phi : (N, N)  cutoff envelope (precomputed)
        rinv: (N, N)  1/r (precomputed)
        """
        # radial sensitivity
        mu = self.sens[:, 0]                               # (ns,)
        sig = torch.abs(self.sens[:, 1]) + 1e-6            # (ns,)

        # s: (N, N, ns)
        #   exp(-((rinv - mu)/2sig)^2) * phi
        x = rinv.unsqueeze(-1)                             # (N, N, 1)
        s = torch.exp(-((x - mu.view(1, 1, -1)) / (2.0 * sig.view(1, 1, -1))) ** 2)
        s = s * phi.unsqueeze(-1)                          # (N, N, ns)

        # e: (N, N, nf)
        e = self.edge(s.reshape(-1, self.ns)).view(rinv.size(0), rinv.size(1), self.nf)

        # msg: (B, N, nf)  msg_i = sum_j e_ij * feat_j
        msg = torch.einsum("ijh,bjh->bih", e, feat)

        h = self.self_lin(feat) + msg
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return h


class HIPNNChargesBatch(nn.Module):
    def __init__(self, n_hist: int = 9, n_features: int = 32, n_interaction: int = 3,
                 n_sensitivity: int = 8, dropout: float = 0.0):
        super().__init__()
        self.n_hist = n_hist
        self.inp = nn.Linear(n_hist, n_features)
        self.inp_ln = nn.LayerNorm(n_features)

        self.blocks = nn.ModuleList()
        self.outs = nn.ModuleList()
        for _ in range(n_interaction):
            self.blocks.append(InteractionLayerBatch(n_features, n_sensitivity=n_sensitivity, dropout=dropout))
            self.outs.append(
                nn.Sequential(
                    nn.Linear(n_features, n_features),
                    nn.Softplus(),
                    nn.Dropout(dropout),
                    nn.Linear(n_features, 1),
                )
            )

    def forward(self, q_hist: torch.Tensor, phi: torch.Tensor, rinv: torch.Tensor):
        """
        q_hist: (B, N, n_hist)
        returns: (B, N)
        """
        h = self.inp(q_hist)               # (B, N, nf)
        h = self.inp_ln(h)

        pred = 0.0
        for blk, head in zip(self.blocks, self.outs):
            h = blk(h, phi, rinv)
            pred = pred + head(h).squeeze(-1)  # (B, N)
        return pred


# ============================================================
#   3) GRU head (same as before, but batch-friendly)
# ============================================================

class GRUHead1D(nn.Module):
    def __init__(self, in_dim=1, hidden=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(-1)  # (B,)


# ============================================================
#   4) Window builders (pre-tensorized)
# ============================================================

def build_windows_q(dqN: np.ndarray, time_fs: np.ndarray, nhist: int, tmin_fs: float,
                    train_end_fs: float, unroll_k: int):
    """
    dqN: (N_atoms, T) normalized
    time_fs: (T,) time stamps in fs (recommended: relative to the first frame, i.e. time_fs[0]=0)

    returns:
      q_hist_all: (n_total, N_atoms, nhist)
      q_tgts_all: (n_total, K, N_atoms)
      starts    : (n_total,)  window start indices
    """
    N, T = dqN.shape
    if time_fs.shape[0] != T:
        raise ValueError(f"time_fs length {time_fs.shape[0]} != dqN T {T}")
    max_i = T - (nhist + unroll_k)
    starts = []
    for i in range(0, max_i + 1):
        t_hist_end = float(time_fs[i + nhist - 1])
        t_target_last = float(time_fs[i + nhist + unroll_k - 1])
        if t_hist_end < tmin_fs:
            continue
        if t_target_last > train_end_fs:
            continue
        starts.append(i)
    starts = np.array(starts, dtype=np.int64)

    n_total = int(starts.shape[0])
    if n_total == 0:
        return None, None, None

    q_hist_all = np.empty((n_total, N, nhist), dtype=np.float32)
    q_tgts_all = np.empty((n_total, unroll_k, N), dtype=np.float32)
    for j, i in enumerate(starts):
        q_hist_all[j] = dqN[:, i:i+nhist]
        for k in range(unroll_k):
            q_tgts_all[j, k] = dqN[:, i+nhist+k]
    return q_hist_all, q_tgts_all, starts


def build_windows_1d(seriesN: np.ndarray, time_fs: np.ndarray, tmin_fs: float,
                     train_end_fs: float, nhist: int, unroll_k: int):
    """
    seriesN: (T,) normalized
    time_fs: (T,) time stamps in fs (recommended: relative to the first frame)

    returns:
      hist_all: (n_total, nhist)
      tgts_all: (n_total, K)
      starts  : (n_total,)
    """
    T = int(seriesN.shape[0])
    if time_fs.shape[0] != T:
        raise ValueError(f"time_fs length {time_fs.shape[0]} != series length {T}")
    max_i = T - (nhist + unroll_k)
    starts = []
    for i in range(0, max_i + 1):
        t_hist_end = float(time_fs[i + nhist - 1])
        t_target_last = float(time_fs[i + nhist + unroll_k - 1])
        if t_hist_end < tmin_fs:
            continue
        if t_target_last > train_end_fs:
            continue
        starts.append(i)
    starts = np.array(starts, dtype=np.int64)

    n_total = int(starts.shape[0])
    if n_total == 0:
        return None, None, None

    hist_all = np.empty((n_total, nhist), dtype=np.float32)
    tgts_all = np.empty((n_total, unroll_k), dtype=np.float32)
    for j, i in enumerate(starts):
        hist_all[j] = seriesN[i:i+nhist]
        tgts_all[j] = seriesN[i+nhist:i+nhist+unroll_k]
    return hist_all, tgts_all, starts

def teacher_forcing_prob(epoch: int, ss_start: float, ss_end: float, ss_decay_epochs: int) -> float:
    """Scheduled sampling / teacher forcing probability.

    Convention: epoch=0 starts at ss_start, then linearly decays to ss_end by ss_decay_epochs.
    If ss_decay_epochs<=0, keep ss_start (i.e., no decay schedule).
    """
    if ss_decay_epochs <= 0:
        return float(ss_start)
    if epoch >= ss_decay_epochs:
        return float(ss_end)
    # linear decay
    frac = float(epoch) / float(ss_decay_epochs)
    return float(ss_start + (ss_end - ss_start) * frac)


# ============================================================
#   5) Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    # original args (keep)
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--dq_dirs", nargs="+", required=True)
    parser.add_argument("--dq_mode", choices=["x", "y", "z"], required=True)
    parser.add_argument("--dir", choices=["x", "y", "z"], required=True)

    parser.add_argument("--dmx", required=True); parser.add_argument("--dmy", required=True); parser.add_argument("--dmz", required=True)
    parser.add_argument("--mmx", required=True); parser.add_argument("--mmy", required=True); parser.add_argument("--mmz", required=True)
    parser.add_argument("--dm_col", type=int, default=3)   # 1-based
    parser.add_argument("--mm_col", type=int, default=2)   # 1-based

    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--nhist", type=int, default=9)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--w_q", type=float, default=1.0)
    parser.add_argument("--w_mu", type=float, default=1.0)
    parser.add_argument("--w_mm", type=float, default=1.0)
    parser.add_argument("--alpha_qsum", type=float, default=0.0)
    parser.add_argument("--lambda_mu_consistency", type=float, default=0.0)

    parser.add_argument("--outdir", required=True)

    # model hyperparams (same names)
    parser.add_argument("--hip_features", type=int, default=32)
    parser.add_argument("--hip_layers", type=int, default=3)
    parser.add_argument("--hip_dropout", type=float, default=0.0)

    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--gru_layers", type=int, default=1)
    parser.add_argument("--gru_dropout", type=float, default=0.0)

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--val_frac", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=0.0)

    parser.add_argument("--lr_step", type=int, default=0)
    parser.add_argument("--lr_gamma", type=float, default=0.5)

    # scheme A split + rollout options (keep)
    parser.add_argument("--train_end_fs", type=float, default=24.0)
    parser.add_argument("--pred_end_fs", type=float, default=32.0)
    # === NEW: fixed-horizon validation & test rollouts (no backprop beyond train_end) ===
    # Train uses labels only up to --train_end_fs.
    # Validation metric is computed on an *open-loop rollout* from --val_start_fs to --val_end_fs
    # (uses ground truth ONLY for scoring, not for feeding).
    # Test rollout is produced from --test_start_fs to --test_end_fs with no ground truth used after start.
    parser.add_argument("--val_start_fs", type=float, default=None,
                        help="Validation rollout start time (fs). Default: train_end_fs.")
    parser.add_argument("--val_end_fs", type=float, default=None,
                        help="Validation rollout end time (fs). Default: train_end_fs + 4.")
    parser.add_argument("--test_start_fs", type=float, default=None,
                        help="Test rollout start time (fs). Default: val_end_fs.")
    parser.add_argument("--test_end_fs", type=float, default=None,
                        help="Test rollout end time (fs). Default: pred_end_fs.")
    parser.add_argument("--save_val_compare", action="store_true",
                        help="Save a CSV comparing val-rollout predictions vs truth (for debugging).")
    parser.add_argument("--save_test_truth", action="store_true",
                        help="Also save truth columns for the test rollout CSV (for debugging; set false to keep test clean).")

    # --- Advanced validation / early-stopping controls (multi-anchor + boundary tail-weighted) ---
    parser.add_argument("--es_warmup_epochs", type=int, default=0,
                        help="Delay updating best/early-stopping until after this many epochs (still logs val).")
    parser.add_argument("--es_tf_threshold", type=float, default=1.0,
                        help="Only update best/early-stopping when teacher-forcing prob tf_p <= this threshold.")
    parser.add_argument("--val_w_internal", type=float, default=0.5,
                        help="Weight for internal multi-anchor validation metric.")
    parser.add_argument("--val_w_boundary", type=float, default=0.5,
                        help="Weight for boundary validation metric (near train_end).")
    parser.add_argument("--val_internal_anchors", type=str, default="",
                        help="Comma-separated anchor times (fs) within [tmin, train_end_fs) for internal open-loop validation. "
                             "Empty -> auto-generate.")
    parser.add_argument("--val_internal_horizon_steps", type=int, default=16,
                        help="Open-loop rollout horizon (steps) for each internal anchor.")
    parser.add_argument("--val_tail_only_frac", type=float, default=0.5,
                        help="For boundary val, only score the last fraction of steps (0<frac<=1). Example: 0.5 -> last half.")
    parser.add_argument("--val_tail_pow", type=float, default=2.0,
                        help="Exponent for tail weighting inside the scored region (>=1). Larger -> more emphasis on the end.")

    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--torch_threads", type=int, default=0)

    # multistep + scheduled sampling
    parser.add_argument("--unroll_k", type=int, default=1)
    parser.add_argument("--ss_start", type=float, default=1.0)
    parser.add_argument("--ss_end", type=float, default=0.0)
    parser.add_argument("--ss_decay_epochs", type=int, default=0)

    # phase/amplitude stabilization (optional, default off)
    parser.add_argument("--bptt", action="store_true",
                        help="Enable truncated BPTT through the K-step unroll (do NOT detach model predictions when feeding next step). "
                             "Slower/more memory, but usually fixes phase drift / amplitude collapse.")
    parser.add_argument("--dq_norm_mode", choices=["global", "per_atom"], default="global",
                        help="dq normalization: global (single mean/std) or per_atom (per-atom mean/std across time).")
    parser.add_argument("--dq_diff_w", type=float, default=0.0,
                        help="Extra weight for dq finite-difference (derivative) loss over the K-step window; helps phase.")
    parser.add_argument("--dq_corr_w", type=float, default=0.0,
                        help="Extra weight for dq correlation loss over the K-step window; helps phase alignment.")
    parser.add_argument("--dq_var_w", type=float, default=0.0,
                        help="Extra weight for dq std/variance matching loss over the K-step window; helps amplitude.")

    # per-step amplitude/DC constraints (optional; default off)
    parser.add_argument("--dq_rms_step_w", type=float, default=0.0,
                        help="Extra weight for matching RMS(|dq|) across atoms at each rollout step (helps amplitude stability).")
    parser.add_argument("--dq_mean_step_w", type=float, default=0.0,
                        help="Extra weight for matching mean(dq) across atoms at each rollout step (reduces DC drift).")

    # sampling strategy (optional; default keeps original random window sampling)
    parser.add_argument("--time_stratify", action="store_true",
                        help="Use time-stratified mini-batch sampling so early/late windows are seen more uniformly (train-only).")
    parser.add_argument("--time_bins", type=int, default=8,
                        help="Number of time strata bins for --time_stratify.")
    parser.add_argument("--time_focus_start_fs", type=float, default=18.0,
                        help="When --time_stratify is on, mildly oversample bins with center >= this time (fs).")
    parser.add_argument("--time_focus_boost", type=float, default=2.0,
                        help="Oversampling factor for late-time bins (>= time_focus_start_fs).")
    parser.add_argument("--k_gamma", type=float, default=1.0,
                        help="Per-step weight multiplier for unroll loss (>1 emphasizes later steps).")

    # validation behavior (default keeps same as current teacher-forcing validation)
    parser.add_argument(
        "--val_rollout",
        action="store_true",
        help="Use rollout-style validation (feed model predictions instead of ground truth) within the train window only. "
             "This does NOT use any labels beyond --train_end_fs.",
    )


    # resume
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="initialize weights from a pretrained (e.g., one-step) checkpoint before multi-step training")
    parser.add_argument("--resume_ckpt", default="")

    args = parser.parse_args()
    device = torch.device(args.device)
    # resolve default split ranges
    if args.val_start_fs is None:
        args.val_start_fs = float(args.train_end_fs)
    if args.val_end_fs is None:
        args.val_end_fs = float(args.train_end_fs) + 4.0
    if args.test_start_fs is None:
        args.test_start_fs = float(args.val_end_fs)
    if args.test_end_fs is None:
        args.test_end_fs = float(args.pred_end_fs)

    if not (args.val_start_fs <= args.val_end_fs):
        raise RuntimeError("--val_start_fs must be <= --val_end_fs")
    if not (args.test_start_fs <= args.test_end_fs):
        raise RuntimeError("--test_start_fs must be <= --test_end_fs")
    if args.val_start_fs < args.train_end_fs - 1e-6:
        print("[Warn] val_start_fs < train_end_fs: validation rollout starts inside training window.")
    if args.test_start_fs < args.val_end_fs - 1e-6:
        print("[Warn] test_start_fs < val_end_fs: test starts before validation ends.")


    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # threads
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_threads // 4)))
        except Exception:
            pass
        print(f"[Torch] num_threads={torch.get_num_threads()}")

    # -------------------------
    # Load geometry & dist
    # -------------------------
    atoms, coords = load_xyz(args.xyz)
    
    pos_t = torch.tensor(coords, dtype=torch.float32, device=device)  # (N,3)
    # Center positions to remove origin-dependence in μ-consistency (translation invariance)
    pos_center = pos_t - pos_t.mean(dim=0, keepdim=True)              # (N,3)
    dist_ij = torch.cdist(pos_t, pos_t)                               # (N,N)

    # precompute phi/rinv once (cutoff = 10 by your default usage)
    rcut = 10.0
    phi = torch.where(
        dist_ij < rcut,
        torch.cos((math.pi * dist_ij) / (2.0 * rcut)) ** 2,
        torch.zeros_like(dist_ij)
    )
    rinv = torch.where(dist_ij > 0, 1.0 / dist_ij, torch.zeros_like(dist_ij))

    # -------------------------
    # Load μ / m
    # -------------------------
    dm_idx0 = args.dm_col - 1
    mm_idx0 = args.mm_col - 1

    t_x, mu_x = load_dm_file(args.dmx, dm_idx0)
    t_y, mu_y = load_dm_file(args.dmy, dm_idx0)
    t_z, mu_z = load_dm_file(args.dmz, dm_idx0)

    t_x2, mm_x = load_dm_file(args.mmx, mm_idx0)
    t_y2, mm_y = load_dm_file(args.mmy, mm_idx0)
    t_z2, mm_z = load_dm_file(args.mmz, mm_idx0)

    mu_map = {'x': mu_x, 'y': mu_y, 'z': mu_z}
    mm_map = {'x': mm_x, 'y': mm_y, 'z': mm_z}
    t_map  = {'x': t_x,  'y': t_y,  'z': t_z}

    mu_vals = mu_map[args.dir]
    mm_vals = mm_map[args.dir]
    times   = t_map[args.dir]


    # NOTE:
    # mu/mm time stamps come from dm/mm dat files and may not perfectly match dq_times.
    # We will later resample mu/mm onto the dq time grid (dq_times) to guarantee alignment
    # before building windows or doing consistency losses.
    if len(times) < 3:
        raise RuntimeError("dm/mm time series too short.")

    # -------------------------
    # Load dq (charges)
    # -------------------------
    def load_dq_from_dir(dirname):
        dq_path = Path(dirname) / "dq_t.npy"
        t_path  = Path(dirname) / "time_fs.npy"
        if not dq_path.exists():
            raise RuntimeError(f"Cannot find dq_t.npy in {dirname}")
        if not t_path.exists():
            raise RuntimeError(f"Cannot find time_fs.npy in {dirname}")


        dq_raw = np.load(dq_path).astype(np.float32)
        t  = np.load(t_path).astype(np.float32)      # (N_time,)
        if dq_raw.ndim != 2:
            raise RuntimeError(f"dq_t.npy must be 2D, got shape={dq_raw.shape} in {dirname}")
        # We want dq as (N_atoms, T). Support both (N_atoms,T) and (T,N_atoms).
        if dq_raw.shape[1] == t.shape[0]:
            dq = dq_raw
        elif dq_raw.shape[0] == t.shape[0]:
            dq = dq_raw.T
        else:
            raise RuntimeError(
                f"dq_t.npy shape {dq_raw.shape} is incompatible with time_fs length {t.shape[0]} in {dirname}"
            )
        return dq, t

    dq_list = []
    time_list = []
    for d in args.dq_dirs:
        dq, t = load_dq_from_dir(d)
        dq_list.append(dq)
        time_list.append(t)

    for i in range(1, len(time_list)):
        if not np.allclose(time_list[i], time_list[0]):
            raise RuntimeError("time_fs.npy mismatch among dq_dirs directories")

    

    dq_times = time_list[0].astype(np.float32)

    # Use a relative time axis (fs) so that args like --tmin/--train_end_fs are interpreted from 0 fs.
    time_fs = (dq_times - float(dq_times[0])).astype(np.float32)

    # infer dt_fs from dq time (robust to small numerical jitter)
    if len(time_fs) < 3:
        raise RuntimeError("dq time series too short.")
    dt_fs = float(np.median(np.diff(time_fs)))

    # -------------------------
    # Resample mu/mm onto dq time grid (fixes time-axis mismatch)
    # -------------------------
    dm_time_fs = (times - float(times[0])).astype(np.float32)
    if dm_time_fs.shape[0] != mu_vals.shape[0] or dm_time_fs.shape[0] != mm_vals.shape[0]:
        raise RuntimeError("dm/mm time/value length mismatch.")

    # Linear interpolation; if dq grid slightly extends beyond dm grid, clamp to edge values.
    mu_vals_rs = np.interp(time_fs, dm_time_fs, mu_vals, left=float(mu_vals[0]), right=float(mu_vals[-1])).astype(np.float32)
    mm_vals_rs = np.interp(time_fs, dm_time_fs, mm_vals, left=float(mm_vals[0]), right=float(mm_vals[-1])).astype(np.float32)

    # normalize μ/m using TRAIN-ONLY stats (time_fs <= train_end_fs) to avoid time-leakage
    train_mask = time_fs <= (float(args.train_end_fs) + 1e-9)
    if int(train_mask.sum()) < 2:
        raise RuntimeError("train_end_fs is too small; cannot compute mu/mm normalization on train segment.")
    mu_mean = float(mu_vals_rs[train_mask].mean())
    mu_std  = float(mu_vals_rs[train_mask].std() + 1e-12)
    mm_mean = float(mm_vals_rs[train_mask].mean())
    mm_std  = float(mm_vals_rs[train_mask].std() + 1e-12)
    muN = (mu_vals_rs - mu_mean) / mu_std
    mmN = (mm_vals_rs - mm_mean) / mm_std

    mu_mean_t = torch.tensor(mu_mean, dtype=torch.float32, device=device)
    mu_std_t  = torch.tensor(mu_std,  dtype=torch.float32, device=device)
    mm_mean_t = torch.tensor(mm_mean, dtype=torch.float32, device=device)
    mm_std_t  = torch.tensor(mm_std,  dtype=torch.float32, device=device)


    dq_all = np.stack(dq_list, axis=0)   # (3, N_atoms, N_time)
    mode_map = {"x": 0, "y": 1, "z": 2}
    dq_sel = dq_all[mode_map[args.dq_mode]]  # (N_atoms, N_time)


    # ---- dq normalization (global vs per-atom), TRAIN-ONLY stats to avoid time-leakage ----
    i_dq_end = int(np.searchsorted(time_fs, args.train_end_fs, side="right") - 1)
    if i_dq_end < 1:
        raise RuntimeError("train_end_fs is too small; cannot compute dq normalization on train segment.")
    i_dq_end = min(i_dq_end, dq_sel.shape[1] - 1)
    dq_train = dq_sel[:, :i_dq_end + 1]
    if args.dq_norm_mode == "per_atom":
        # (N_atoms, 1) mean/std so broadcasts with (B,N_atoms)
        dq_mean = dq_train.mean(axis=1, keepdims=True)
        dq_std  = dq_train.std(axis=1, keepdims=True) + 1e-12
        dqN = (dq_sel - dq_mean) / dq_std
    else:
        dq_mean = float(dq_train.mean())
        dq_std  = float(dq_train.std() + 1e-12)
        dqN = (dq_sel - dq_mean) / dq_std

    # dq mean/std numpy (broadcastable for (T,N) arrays in rollout)
    dq_mean_np = dq_mean
    dq_std_np  = dq_std
    if isinstance(dq_mean_np, np.ndarray) and dq_mean_np.ndim == 2 and dq_mean_np.shape[1] == 1:
        dq_mean_np = dq_mean_np.T  # (1,N)
        dq_std_np  = dq_std_np.T

    # dq mean/std tensors (broadcastable)
    dq_mean_t = torch.as_tensor(dq_mean, dtype=torch.float32, device=device)
    dq_std_t  = torch.as_tensor(dq_std,  dtype=torch.float32, device=device)
    if dq_mean_t.ndim == 2 and dq_mean_t.shape[1] == 1:
        # (N,1) -> (1,N)
        dq_mean_t = dq_mean_t.T
        dq_std_t  = dq_std_t.T
    # ============================================================
    # Build TRAIN windows (0..train_end)
    # ============================================================
    K = int(args.unroll_k)
    if K < 1:
        raise RuntimeError("--unroll_k must be >=1")

    
    q_hist_np, q_tgts_np, starts_q = build_windows_q(
        dqN=dqN, time_fs=time_fs, nhist=args.nhist, tmin_fs=args.tmin,
        train_end_fs=args.train_end_fs, unroll_k=K
    )
    mu_hist_np, mu_tgts_np, starts_mu = build_windows_1d(
        seriesN=muN, time_fs=time_fs, tmin_fs=args.tmin, train_end_fs=args.train_end_fs,
        nhist=args.nhist, unroll_k=K
    )
    mm_hist_np, mm_tgts_np, starts_mm = build_windows_1d(
        seriesN=mmN, time_fs=time_fs, tmin_fs=args.tmin, train_end_fs=args.train_end_fs,
        nhist=args.nhist, unroll_k=K
    )

    if q_hist_np is None or mu_hist_np is None or mm_hist_np is None:
        raise RuntimeError("No training windows built; check --nhist/--tmin/--train_end_fs and time steps.")
    # Ensure window starts are aligned across q / mu / mm (prevent silent time-misalignment)
    if not (np.array_equal(starts_mu, starts_q) and np.array_equal(starts_mm, starts_q)):
        common = np.intersect1d(starts_q, np.intersect1d(starts_mu, starts_mm))
        if common.size == 0:
            raise RuntimeError("No common window starts across q/mu/mm; check time grids and filtering.")
        idx_q  = {int(s): i for i, s in enumerate(starts_q)}
        idx_mu = {int(s): i for i, s in enumerate(starts_mu)}
        idx_mm = {int(s): i for i, s in enumerate(starts_mm)}
        sel_q  = np.array([idx_q[int(s)] for s in common], dtype=np.int64)
        sel_mu = np.array([idx_mu[int(s)] for s in common], dtype=np.int64)
        sel_mm = np.array([idx_mm[int(s)] for s in common], dtype=np.int64)
        q_hist_np, q_tgts_np = q_hist_np[sel_q], q_tgts_np[sel_q]
        mu_hist_np, mu_tgts_np = mu_hist_np[sel_mu], mu_tgts_np[sel_mu]
        mm_hist_np, mm_tgts_np = mm_hist_np[sel_mm], mm_tgts_np[sel_mm]
        starts_q = common

    n_total = int(starts_q.shape[0])

    # representative time (fs) for each window (time of the last target step), used by optional --time_stratify
    win_times = time_fs[starts_q + args.nhist + K - 1].astype(np.float32)

    # pre-tensorize once (stay on device; for cpu it's fine)
    q_hist_all = torch.from_numpy(q_hist_np).to(device)          # (n_total, N, nhist)
    q_tgts_all = torch.from_numpy(q_tgts_np).to(device)          # (n_total, K, N)
    mu_hist_all = torch.from_numpy(mu_hist_np).to(device)        # (n_total, nhist)
    mu_tgts_all = torch.from_numpy(mu_tgts_np).to(device)        # (n_total, K)
    mm_hist_all = torch.from_numpy(mm_hist_np).to(device)        # (n_total, nhist)
    mm_tgts_all = torch.from_numpy(mm_tgts_np).to(device)        # (n_total, K)
    # Train/Val split (LEGACY)
    # NOTE: val metric is no longer computed by holding out windows within 0..train_end_fs.
    # We always train on all train-window samples, and validate via a fixed-horizon open-loop rollout
    # (see --val_start_fs/--val_end_fs). Keep val_frac for backward-compat but do not use it.
    rng = np.random.default_rng(42)
    idx_all = np.arange(n_total)
    train_idx = idx_all
    val_idx = np.array([], dtype=np.int64)

    # ============================================================
    # Instantiate models
    # ============================================================
    hip = HIPNNChargesBatch(
        n_hist=args.nhist,
        n_features=args.hip_features,
        n_interaction=args.hip_layers,
        n_sensitivity=8,
        dropout=args.hip_dropout,
    ).to(device)

    mu_gru = GRUHead1D(
        in_dim=1, hidden=args.gru_hidden, num_layers=args.gru_layers, dropout=args.gru_dropout
    ).to(device)

    mm_gru = GRUHead1D(
        in_dim=1, hidden=args.gru_hidden, num_layers=args.gru_layers, dropout=args.gru_dropout
    ).to(device)

    def _load_ckpt_weights(ckpt_obj, hip, mu_gru, mm_gru, tag):
        """Load weights from various checkpoint formats (dict with keys or plain state_dict)."""
        if isinstance(ckpt_obj, dict) and ("hip" in ckpt_obj or "mu_gru" in ckpt_obj or "mm_gru" in ckpt_obj):
            if "hip" in ckpt_obj:
                hip.load_state_dict(ckpt_obj["hip"], strict=False)
            if "mu_gru" in ckpt_obj:
                mu_gru.load_state_dict(ckpt_obj["mu_gru"], strict=False)
            if "mm_gru" in ckpt_obj:
                mm_gru.load_state_dict(ckpt_obj["mm_gru"], strict=False)
        elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            # some scripts save under state_dict
            sd = ckpt_obj["state_dict"]
            # try direct load into hip; other heads may be absent
            hip.load_state_dict(sd, strict=False)
        else:
            # assume ckpt_obj is a plain state_dict for hip
            hip.load_state_dict(ckpt_obj, strict=False)
        print(f"[{tag}] loaded weights")

    # Optional: initialize from a pretrained (e.g., one-step) checkpoint (weights only).
    # This does NOT resume optimizer/scheduler states; it just warm-starts parameters.
    if args.init_ckpt:
        ckpt_path = Path(args.init_ckpt)
        if not ckpt_path.exists():
            raise RuntimeError(f"--init_ckpt not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        _load_ckpt_weights(ckpt, hip, mu_gru, mm_gru, tag="Init")
        print(f"[Init] from {ckpt_path}")

    # Optional: resume (weights only) — will override init_ckpt if both are provided.
    if args.resume_ckpt:
        ckpt_path = Path(args.resume_ckpt)
        if not ckpt_path.exists():
            raise RuntimeError(f"--resume_ckpt not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
        _load_ckpt_weights(ckpt, hip, mu_gru, mm_gru, tag="Resume")
        print(f"[Resume] from {ckpt_path}")

    # Optimizer / scheduler
    optim_all = optim.AdamW(
        list(hip.parameters()) + list(mu_gru.parameters()) + list(mm_gru.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    mse = nn.MSELoss()

    scheduler = None
    if args.lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optim_all, step_size=args.lr_step, gamma=args.lr_gamma)
    # ------------------------------------------------------------
    # Fixed-horizon open-loop rollout evaluation (no backprop)
    # ------------------------------------------------------------
    def _idx_from_time(time_arr, tfs: float) -> int:
        """Return index of closest time in fs (robust to float rounding)."""
        ta = np.asarray(time_arr, dtype=float).reshape(-1)
        idx = int(np.argmin(np.abs(ta - float(tfs))))
        if len(ta) >= 3:
            dt = float(np.median(np.diff(ta)))
            if abs(float(ta[idx] - float(tfs))) > max(1e-3, 0.6 * dt):
                print(f"[Warn] time mismatch: requested {tfs:.6f} fs, closest is {ta[idx]:.6f} fs (dt~{dt:.6f} fs)")
        return idx

    def _rollout_openloop(start_fs: float, end_fs: float, *, return_truth: bool = True):
        """
        Open-loop rollout on [start_fs, end_fs] with NO teacher forcing.
        Uses truth only to build the initial nhist context (i0-nhist .. i0-1).
        Returned arrays:
          time_fs: [steps]
          q_pred:  [steps, N] (de-norm)
          mu_pred/mm_pred: [steps] (de-norm)
          q_predN/mu_predN/mm_predN: normalized
          (optional) *_true, *_trueN
        """
        hip.eval(); mu_gru.eval(); mm_gru.eval()
    
        i0 = _idx_from_time(time_fs, start_fs)
        i1 = _idx_from_time(time_fs, end_fs)
        if i1 < i0:
            raise RuntimeError(f"rollout end_fs must be >= start_fs (i0={i0}, i1={i1}).")
        if i0 - args.nhist < 0:
            raise RuntimeError(f"Need i0-nhist>=0 for rollout seed (i0={i0}, nhist={args.nhist}).")
    
        # initial histories in normalized space (history ends at i0-1)
        q_hist = torch.from_numpy(dqN[:, i0-args.nhist:i0].T).float().to(device)   # [nhist, N]
        mu_hist = torch.from_numpy(muN[i0-args.nhist:i0]).float().to(device)       # [nhist]
        mm_hist = torch.from_numpy(mmN[i0-args.nhist:i0]).float().to(device)       # [nhist]
    
        q_hist = q_hist.unsqueeze(0).transpose(1, 2).contiguous()  # [1, N, nhist]
        mu_hist = mu_hist.unsqueeze(0)                              # [1, nhist]
        mm_hist = mm_hist.unsqueeze(0)                              # [1, nhist]
    
        steps = int(i1 - i0 + 1)
    
        q_predN_list, mu_predN_list, mm_predN_list = [], [], []
        q_trueN_list, mu_trueN_list, mm_trueN_list = [], [], []
        t_list = []
    
        with torch.no_grad():
            for s in range(steps):
                q_predN = hip(q_hist, phi, rinv)                  # [1, N]
                mu_predN = mu_gru(mu_hist.unsqueeze(-1))          # [1]
                mm_predN = mm_gru(mm_hist.unsqueeze(-1))          # [1]
    
                # record prediction at this time step
                q_predN_list.append(q_predN.squeeze(0).detach().cpu().numpy())
                mu_predN_list.append(float(mu_predN.item()))
                mm_predN_list.append(float(mm_predN.item()))
    
                idx_t = i0 + s
                t_list.append(float(time_fs[idx_t]))
    
                if return_truth:
                    q_trueN_list.append(dqN[:, idx_t].astype(np.float32))
                    mu_trueN_list.append(float(muN[idx_t]))
                    mm_trueN_list.append(float(mmN[idx_t]))
    
                # advance histories with predictions
                q_hist = torch.cat([q_hist[:, :, 1:], q_predN.unsqueeze(-1)], dim=-1)
                mu_hist = torch.cat([mu_hist[:, 1:], mu_predN.unsqueeze(1)], dim=1)
                mm_hist = torch.cat([mm_hist[:, 1:], mm_predN.unsqueeze(1)], dim=1)
    
        q_predN = np.asarray(q_predN_list, dtype=np.float32)
        mu_predN = np.asarray(mu_predN_list, dtype=np.float32)
        mm_predN = np.asarray(mm_predN_list, dtype=np.float32)
    
        out = {
            "time_fs": np.asarray(t_list, dtype=np.float32),
            "q_predN": q_predN,
            "mu_predN": mu_predN,
            "mm_predN": mm_predN,
            "q_pred": q_predN * dq_std_np + dq_mean_np,
            "mu_pred": mu_predN * mu_std + mu_mean,
            "mm_pred": mm_predN * mm_std + mm_mean,
        }
        if return_truth:
            q_trueN = np.asarray(q_trueN_list, dtype=np.float32)
            mu_trueN = np.asarray(mu_trueN_list, dtype=np.float32)
            mm_trueN = np.asarray(mm_trueN_list, dtype=np.float32)
            out.update({
                "q_trueN": q_trueN,
                "mu_trueN": mu_trueN,
                "mm_trueN": mm_trueN,
                "q_true": q_trueN * dq_std_np + dq_mean_np,
                "mu_true": mu_trueN * mu_std + mu_mean,
                "mm_true": mm_trueN * mm_std + mm_mean,
            })
        return out

    def _val_metric_rollout() -> float:
        """Validation metric: open-loop rollout from val_start_fs to val_end_fs."""
        out = _rollout_openloop(args.val_start_fs, args.val_end_fs, return_truth=True)

        q_predN, q_trueN = out["q_predN"], out["q_trueN"]
        mu_predN, mu_trueN = out["mu_predN"], out["mu_trueN"]
        mm_predN, mm_trueN = out["mm_predN"], out["mm_trueN"]

        # step weights (k_gamma)
        if args.k_gamma is None:
            step_w = np.ones(len(q_predN), dtype=float)
        else:
            step_w = (float(args.k_gamma) ** np.arange(len(q_predN))).astype(float)
            step_w = step_w / (step_w.mean() + 1e-12)

        def _mse_per_step(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            d = (a - b) ** 2
            # q: (T,N) -> (T,), mu/mm: (T,) -> (T,)
            return d if d.ndim == 1 else d.mean(axis=1)

        q_mse = _mse_per_step(q_predN, q_trueN)
        mu_mse = _mse_per_step(mu_predN, mu_trueN)
        mm_mse = _mse_per_step(mm_predN, mm_trueN)

        loss_steps = args.w_q * q_mse + args.w_mu * mu_mse + args.w_mm * mm_mse
        return float(np.sum(loss_steps * step_w) / (np.sum(step_w) + 1e-12))


    
    # ------------------------------------------------------------
    # Composite validation:
    #   (1) internal multi-anchor open-loop metric within [tmin, train_end_fs)
    #   (2) boundary open-loop metric on [val_start_fs, val_end_fs] with tail-weighting
    # ------------------------------------------------------------
    def _as_step_mean(arr) -> np.ndarray:
        """Convert (T,), (T,1), (T,3) ... -> (T,) by averaging feature dims."""
        a = np.asarray(arr)
        if a.ndim == 1:
            return a.astype(float)
        if a.ndim == 2:
            return a.astype(float).mean(axis=1)
        a = a.reshape(a.shape[0], -1)
        return a.astype(float).mean(axis=1)

    def _parse_internal_anchors() -> list[float]:
        s = (args.val_internal_anchors or "").strip()
        if s:
            out = []
            for tok in s.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                out.append(float(tok))
            return out

        # auto-generate anchors with a coarse spacing (in fs)
        dt = float(np.median(np.diff(np.asarray(time_fs, dtype=float))))
        # need enough history: i0 - nhist >= 0
        min_fs = float(time_fs[args.nhist]) if len(dq_times) > args.nhist else float(time_fs[-1])
        horizon_fs = float(args.val_internal_horizon_steps) * dt
        max_fs = float(args.train_end_fs) - horizon_fs
        if max_fs <= min_fs + 1e-9:
            return []

        step_fs = 4.0  # fs; coarse internal probes
        a = max(min_fs, float(args.tmin))
        anchors = []
        while a <= max_fs + 1e-9:
            anchors.append(a)
            a += step_fs
        return anchors

    def _tail_weights(steps: int) -> np.ndarray:
        """Weights over steps emphasizing the tail (last fraction). Sum to 1."""
        steps = int(steps)
        if steps <= 0:
            return np.zeros((0,), dtype=float)
        frac = float(args.val_tail_only_frac)
        frac = min(max(frac, 1e-6), 1.0)
        pow_ = max(float(args.val_tail_pow), 1.0)

        start = int(math.floor((1.0 - frac) * steps))
        start = min(max(start, 0), steps - 1)

        w = np.zeros((steps,), dtype=float)
        denom = max((steps - 1 - start), 1)
        for s in range(start, steps):
            x = (s - start) / float(denom)  # 0..1
            w[s] = x ** pow_
        if w.sum() <= 0:
            w[start:] = 1.0
        w = w / (w.sum() + 1e-12)
        return w

    def _val_metric_boundary_tail() -> float:
        out = _rollout_openloop(args.val_start_fs, args.val_end_fs, return_truth=True)
        q_mse = ((np.asarray(out["q_predN"]) - np.asarray(out["q_trueN"])) ** 2).mean(axis=1)  # (T,)
        mu_mse = ((np.asarray(out["mu_predN"]) - np.asarray(out["mu_trueN"])) ** 2)
        mm_mse = ((np.asarray(out["mm_predN"]) - np.asarray(out["mm_trueN"])) ** 2)

        mu_mse = _as_step_mean(mu_mse)
        mm_mse = _as_step_mean(mm_mse)

        loss_step = args.w_q * q_mse + args.w_mu * mu_mse + args.w_mm * mm_mse  # (T,)

        # tail-weighted (optionally also k_gamma)
        w_tail = _tail_weights(len(loss_step))
        w_k = np.array([(args.k_gamma ** s) for s in range(len(loss_step))], dtype=float)
        w = w_tail * w_k
        w = w / (w.sum() + 1e-12)
        return float((w * loss_step).sum())

    def _val_metric_internal_multi_anchor() -> tuple[float, int]:
        anchors = _parse_internal_anchors()
        if not anchors:
            return (float("nan"), 0)

        # determine train_end index and dt
        dt = float(np.median(np.diff(np.asarray(time_fs, dtype=float))))
        i_train_end = _idx_from_time(time_fs, float(args.train_end_fs))
        horizon = int(args.val_internal_horizon_steps)

        losses = []
        used = 0
        for a_fs in anchors:
            i0 = _idx_from_time(time_fs, a_fs)
            i1 = i0 + horizon
            if i0 - args.nhist < 0:
                continue
            if i1 > i_train_end:
                continue
            start_fs = float(time_fs[i0])
            end_fs = float(time_fs[i1])
            out = _rollout_openloop(start_fs, end_fs, return_truth=True)

            q_mse = ((np.asarray(out["q_predN"]) - np.asarray(out["q_trueN"])) ** 2).mean(axis=1)
            mu_mse = _as_step_mean((np.asarray(out["mu_predN"]) - np.asarray(out["mu_trueN"])) ** 2)
            mm_mse = _as_step_mean((np.asarray(out["mm_predN"]) - np.asarray(out["mm_trueN"])) ** 2)
            loss_step = args.w_q * q_mse + args.w_mu * mu_mse + args.w_mm * mm_mse

            w = np.array([(args.k_gamma ** s) for s in range(len(loss_step))], dtype=float)
            w = w / (w.sum() + 1e-12)
            losses.append(float((w * loss_step).sum()))
            used += 1

        if used == 0:
            return (float("nan"), 0)
        return (float(np.mean(losses)), used)

    def _val_metric_composite() -> tuple[float, float, float, int]:
        v_internal, n_used = _val_metric_internal_multi_anchor()
        v_boundary = _val_metric_boundary_tail()

        wi = float(args.val_w_internal)
        wb = float(args.val_w_boundary)
        wsum = wi + wb
        if not np.isfinite(v_internal):
            wi = 0.0
        if wsum <= 0:
            v_total = v_boundary
        else:
            v_total = (wi * (v_internal if np.isfinite(v_internal) else 0.0) + wb * v_boundary) / (wi + wb + 1e-12)
        return (float(v_total), float(v_internal) if np.isfinite(v_internal) else float("nan"), float(v_boundary), int(n_used))

# ============================================================
    # Training loop (vectorized batch + single backward per epoch-step)
    # ============================================================
    log_rows = []
    best_metric = float("inf")
    best_state = None
    no_improve = 0

    # pos component vector for mu-consistency
    comp = {'x': 0, 'y': 1, 'z': 2}[args.dir]
    pos_comp = pos_t[:, comp]  # (N,) use raw coordinates to match dm dipole origin

    for ep in range(args.epochs):
        hip.train(); mu_gru.train(); mm_gru.train()

        if len(train_idx) == 0:
            cur_idx = idx_all
        else:
            bs = min(args.batch, len(train_idx))
            if args.time_stratify:
                cur_idx = stratified_batch_indices(
                    train_idx, win_times, bs, args.time_bins,
                    t_min=args.tmin, t_max=args.train_end_fs,
                    focus_start=args.time_focus_start_fs,
                    focus_boost=args.time_focus_boost, rng=rng
                )
            else:
                cur_idx = rng.choice(train_idx, size=bs, replace=False)

        cur_idx_t = torch.as_tensor(cur_idx, dtype=torch.long, device=device)

        # gather batch tensors (NO python per-sample loop)
        q_hist = q_hist_all[cur_idx_t].clone()    # (B,N,nhist)  clone because we'll roll
        mu_hist = mu_hist_all[cur_idx_t].clone()  # (B,nhist)
        mm_hist = mm_hist_all[cur_idx_t].clone()

        q_tgts = q_tgts_all[cur_idx_t]           # (B,K,N)
        mu_tgts = mu_tgts_all[cur_idx_t]         # (B,K)
        mm_tgts = mm_tgts_all[cur_idx_t]         # (B,K)

        B = q_hist.size(0)

        tf_p = teacher_forcing_prob(ep, args.ss_start, args.ss_end, args.ss_decay_epochs)

        loss_q_sum = torch.zeros((), device=device)
        loss_mu_sum = torch.zeros((), device=device)
        loss_mm_sum = torch.zeros((), device=device)

        optim_all.zero_grad(set_to_none=True)


        # store dq sequence for window-level losses
        q_pred_seq = []
        w_sum = 0.0

        for s in range(K):
            w_s = float(args.k_gamma) ** s
            q_trueN = q_tgts[:, s, :]
            mu_trueN = mu_tgts[:, s]
            mm_trueN = mm_tgts[:, s]

            q_predN = hip(q_hist, phi, rinv)
            mu_predN = mu_gru(mu_hist.unsqueeze(-1))     # [1]
            mm_predN = mm_gru(mm_hist.unsqueeze(-1))     # [1]

            q_pred_seq.append(q_predN)

            loss_q_sum = loss_q_sum + w_s * mse(q_predN, q_trueN)
            loss_mu_sum = loss_mu_sum + w_s * mse(mu_predN, mu_trueN)
            loss_mm_sum = loss_mm_sum + w_s * mse(mm_predN, mm_trueN)
            # Optional regularizers that require de-normalized charges
            if (args.lambda_mu_consistency > 0.0) or (args.alpha_qsum > 0.0):
                q_pred_real = q_predN * dq_std_t + dq_mean_t

            # Total-charge conservation penalty (was previously a no-op)
            if args.alpha_qsum > 0.0:
                loss_q_sum = loss_q_sum + w_s * args.alpha_qsum * charge_cons_penalty(q_pred_real)

            # μ-consistency: enforce μ ≈ Σ_i q_i * r_i (with centered coordinates)
            if args.lambda_mu_consistency > 0.0:
                mu_base_real = (q_pred_real * pos_comp).sum(dim=1)
                mu_baseN = (mu_base_real - mu_mean_t) / mu_std_t
                loss_mu_sum = loss_mu_sum + w_s * args.lambda_mu_consistency * mse(mu_predN, mu_baseN)

            w_sum = w_sum + w_s

            if s < K - 1:
                use_tf = (torch.rand(B, device=device) < tf_p).float()
                if args.bptt:
                    q_model, mu_model, mm_model = q_predN, mu_predN, mm_predN
                else:
                    q_model, mu_model, mm_model = q_predN.detach(), mu_predN.detach(), mm_predN.detach()

                q_feed = use_tf.view(-1, 1) * q_trueN + (1.0 - use_tf.view(-1, 1)) * q_model
                mu_feed = use_tf * mu_trueN + (1.0 - use_tf) * mu_model
                mm_feed = use_tf * mm_trueN + (1.0 - use_tf) * mm_model

                q_hist = torch.roll(q_hist, shifts=-1, dims=2)
                q_hist[:, :, -1] = q_feed
                mu_hist = torch.roll(mu_hist, shifts=-1, dims=1)
                mu_hist[:, -1] = mu_feed
                mm_hist = torch.roll(mm_hist, shifts=-1, dims=1)
                mm_hist[:, -1] = mm_feed

        loss_q = loss_q_sum / max(w_sum, 1e-12)
        loss_mu = loss_mu_sum / max(w_sum, 1e-12)
        loss_mm = loss_mm_sum / max(w_sum, 1e-12)

        # window-level dq losses (phase/amplitude)
        if (args.dq_diff_w > 0.0) or (args.dq_corr_w > 0.0) or (args.dq_var_w > 0.0):
            q_pred_seq_t = torch.stack(q_pred_seq, dim=1)
            q_true_seq_t = q_tgts

            if args.dq_diff_w > 0.0 and K >= 2:
                dqdp = q_pred_seq_t[:, 1:, :] - q_pred_seq_t[:, :-1, :]
                dqdt = q_true_seq_t[:, 1:, :] - q_true_seq_t[:, :-1, :]
                loss_q = loss_q + args.dq_diff_w * mse(dqdp, dqdt)

            if args.dq_corr_w > 0.0:
                xp = q_pred_seq_t - q_pred_seq_t.mean(dim=1, keepdim=True)
                xt = q_true_seq_t - q_true_seq_t.mean(dim=1, keepdim=True)
                num = (xp * xt).sum(dim=1)
                den = (xp.norm(dim=1) * xt.norm(dim=1) + 1e-8)
                corr = num / den
                loss_q = loss_q + args.dq_corr_w * (1.0 - corr).mean()

            # per-step amplitude/DC constraints across atoms (real units)
            if (args.dq_rms_step_w != 0.0) or (args.dq_mean_step_w != 0.0):
                q_pred_real_seq = q_pred_seq_t * dq_std_t + dq_mean_t
                q_true_real_seq = q_true_seq_t * dq_std_t + dq_mean_t
                rms_pred_step = torch.sqrt(torch.mean(q_pred_real_seq ** 2, dim=2) + 1e-12)  # (B,K)
                rms_true_step = torch.sqrt(torch.mean(q_true_real_seq ** 2, dim=2) + 1e-12)  # (B,K)
                mean_pred_step = torch.mean(q_pred_real_seq, dim=2)  # (B,K)
                mean_true_step = torch.mean(q_true_real_seq, dim=2)  # (B,K)
                loss_q = loss_q + args.dq_rms_step_w * mse(rms_pred_step, rms_true_step) + args.dq_mean_step_w * mse(mean_pred_step, mean_true_step)

            if args.dq_var_w > 0.0:
                # --- amplitude / envelope constraints ---
                # 1) per-atom std over the K-step window (original behavior)
                std_p = q_pred_seq_t.std(dim=1)     # (B, N)
                std_t = q_true_seq_t.std(dim=1)     # (B, N)
                loss_q = loss_q + args.dq_var_w * mse(std_p, std_t)

                # 2) per-atom RMS over the K-step window (captures mean-shift + amplitude)
                #    This is more informative than std when rollout develops a DC bias.
                rms_p = torch.sqrt((q_pred_seq_t ** 2).mean(dim=1) + 1e-12)  # (B, N)
                rms_t = torch.sqrt((q_true_seq_t ** 2).mean(dim=1) + 1e-12)  # (B, N)
                loss_q = loss_q + (0.5 * args.dq_var_w) * mse(rms_p, rms_t)

                # 3) small mean matching (prevents drifting to a wrong constant offset in long rollouts)
                mean_p = q_pred_seq_t.mean(dim=1)   # (B, N)
                mean_t = q_true_seq_t.mean(dim=1)   # (B, N)
                loss_q = loss_q + (0.1 * args.dq_var_w) * mse(mean_p, mean_t)
        loss_total = args.w_q * loss_q + args.w_mu * loss_mu + args.w_mm * loss_mm
        loss_total.backward()
        optim_all.step()
        if scheduler is not None:
            scheduler.step()

        row = dict(
            epoch=int(ep),
            loss=float(loss_total.item()),
            l_q=float(loss_q.item()),
            l_mu=float(loss_mu.item()),
            l_mm=float(loss_mm.item()),
            l_q_base=float((loss_q_sum / max(w_sum, 1e-12)).item()),
            l_mu_base=float((loss_mu_sum / max(w_sum, 1e-12)).item()),
            l_mm_base=float((loss_mm_sum / max(w_sum, 1e-12)).item()),
            tf_prob=float(tf_p),
        )
        # -------------------------
        # Validation (open-loop rollout, no backprop)
        # -------------------------
        # Protocol:
        # - train: supervised/backprop only within [0, train_end_fs]
        # - val metric: open-loop rollout (val_start_fs -> val_end_fs], compare against truth for scoring only
        hip.eval(); mu_gru.eval(); mm_gru.eval()
        with torch.no_grad():
            val_total, val_internal, val_boundary, n_anchor = _val_metric_composite()

        row["val_loss"] = val_total
        row["val_internal"] = val_internal
        row["val_boundary"] = val_boundary
        row["val_n_anchor"] = int(n_anchor)
        row["val_start_fs"] = float(args.val_start_fs)
        row["val_end_fs"] = float(args.val_end_fs)

        # save optional val compare CSV
        if args.save_val_compare:
            try:
                _outv = _rollout_openloop(args.val_start_fs, args.val_end_fs, return_truth=True)
                import pandas as _pd
                _df = _pd.DataFrame({"time_fs": _outv["time_fs"]})

                # charges: always (T, N)
                _q_pred = np.asarray(_outv["q_pred"])
                _q_true = np.asarray(_outv["q_true"])
                for _ai in range(_q_pred.shape[1]):
                    _df[f"q{_ai+1}_pred"] = _q_pred[:, _ai]
                    _df[f"q{_ai+1}_true"] = _q_true[:, _ai]

                def _add_vec(_name: str, _arr, _suffix: str):
                    """Add vector-like series to df.
                    - If arr is (T,), save as <name><dir>_<suffix> for x/y/z dir mode.
                    - If arr is (T,1), same as above.
                    - If arr is (T,3), save as <name>x_<suffix>, <name>y_<suffix>, <name>z_<suffix>.
                    """
                    _arr = np.asarray(_arr)
                    if _arr.ndim == 1:
                        _df[f"{_name}{args.dir}_{_suffix}"] = _arr
                    elif _arr.ndim == 2 and _arr.shape[1] == 1:
                        _df[f"{_name}{args.dir}_{_suffix}"] = _arr[:, 0]
                    elif _arr.ndim == 2 and _arr.shape[1] == 3:
                        _df[f"{_name}x_{_suffix}"] = _arr[:, 0]
                        _df[f"{_name}y_{_suffix}"] = _arr[:, 1]
                        _df[f"{_name}z_{_suffix}"] = _arr[:, 2]
                    else:
                        # Fallback: store first column
                        _flat = _arr.reshape(_arr.shape[0], -1)
                        _df[f"{_name}{args.dir}_{_suffix}"] = _flat[:, 0]

                _add_vec("mu", _outv["mu_pred"], "pred")
                _add_vec("mm", _outv["mm_pred"], "pred")
                _add_vec("mu", _outv["mu_true"], "true")
                _add_vec("mm", _outv["mm_true"], "true")

                _fname = f"val_rollout_compare_{args.dir}_{args.val_start_fs:g}to{args.val_end_fs:g}fs.csv"
                _df.to_csv(os.path.join(args.outdir, _fname), index=False)
            except Exception as _e:
                print("[Warn] failed to save val compare:", _e)

        metric = row["val_loss"]

        log_rows.append(row)

        if args.print_every > 0 and ((ep + 1) % args.print_every == 0 or ep == 0):
            msg = f"[{args.dir}] Ep {ep+1:04d} | loss {row['loss']:.4e}"
            if "val_loss" in row:
                msg += f" | val {row['val_loss']:.4e}"
            msg += f" | tf_p {row['tf_prob']:.2f}"
            print(msg)

        # Early stopping & best ckpt (gated)
        gate_ok = ((ep + 1) >= args.es_warmup_epochs) and (tf_p <= args.es_tf_threshold + 1e-12)
        row["es_gate"] = int(gate_ok)

        if gate_ok:
            if metric + args.min_delta < best_metric:
                best_metric = metric
                no_improve = 0
                best_state = {
                    "hip": hip.state_dict(),
                    "mu_gru": mu_gru.state_dict(),
                    "mm_gru": mm_gru.state_dict(),
                    "meta": {
                        "args": vars(args),
                        "best_metric": float(best_metric),
                        "mu_mean": mu_mean, "mu_std": mu_std,
                        "mm_mean": mm_mean, "mm_std": mm_std,
                        "dq_mean": dq_mean, "dq_std": dq_std,
                        "dt_fs": dt_fs,
                    }
                }
                torch.save(best_state, outdir / "best_model.pt")
            else:
                no_improve += 1

            if args.patience > 0 and no_improve >= args.patience:
                print(f"[EarlyStopping] stop at epoch {ep+1}, best={best_metric:.4e} (gate tf_p<={args.es_tf_threshold}, warmup={args.es_warmup_epochs})")
                break
        else:
            # Before gate opens, do not count patience / do not update best.
            no_improve = 0


    # Save training log
    pd.DataFrame(log_rows).to_csv(outdir / f"log_{args.dir}.csv", index=False)

    # Load best (or fallback to last epoch if ES gate never opened)
    if best_state is None:
        best_path = outdir / "best_model.pt"
        if best_path.exists():
            best_state = torch.load(str(best_path), map_location="cpu")
        else:
            print("[Warn] early-stopping gate never opened or no improvement; using last-epoch weights as best_model.pt")
            best_state = {
                "hip": hip.state_dict(),
                "mu_gru": mu_gru.state_dict(),
                "mm_gru": mm_gru.state_dict(),
                "meta": {
                    "args": vars(args),
                    "best_metric": float("nan") if not np.isfinite(best_metric) else float(best_metric),
                    "mu_mean": mu_mean, "mu_std": mu_std,
                    "mm_mean": mm_mean, "mm_std": mm_std,
                    "dq_mean": dq_mean, "dq_std": dq_std,
                    "dt_fs": dt_fs,
                }
            }
            torch.save(best_state, outdir / "best_model.pt")

    hip.load_state_dict(best_state["hip"])
    mu_gru.load_state_dict(best_state["mu_gru"])
    mm_gru.load_state_dict(best_state["mm_gru"])

    hip.eval(); mu_gru.eval(); mm_gru.eval()

    # ============================================================
    # 6) Save traj on 0..train_end (one-step teacher forcing windows)
    #    这里的 traj 含义：对 0..train_end 内每个时间点，用真实历史窗口预测“下一步”，
    #    得到一条完整的“逐点预测轨迹”（不是外推 rollout）。
    # ============================================================
    # Build 1-step windows for mu/mm aligned to train_end (same tmin, nhist)
    K1 = 1
    mu_hist1_np, mu_tgts1_np, starts1 = build_windows_1d(seriesN=muN, time_fs=time_fs, tmin_fs=args.tmin, train_end_fs=args.train_end_fs, nhist=args.nhist, unroll_k=K1)
    mm_hist1_np, mm_tgts1_np, _      = build_windows_1d(seriesN=mmN, time_fs=time_fs, tmin_fs=args.tmin, train_end_fs=args.train_end_fs, nhist=args.nhist, unroll_k=K1)
    if mu_hist1_np is None:
        raise RuntimeError("No traj windows for mu/mm. Check tmin/nhist/train_end.")

    n_traj = min(len(mu_hist1_np), len(mm_hist1_np))
    mu_hist1 = torch.from_numpy(mu_hist1_np[:n_traj]).to(device)
    mm_hist1 = torch.from_numpy(mm_hist1_np[:n_traj]).to(device)
    mu_true1 = mu_tgts1_np[:n_traj, 0]
    mm_true1 = mm_tgts1_np[:n_traj, 0]
    # time for target point is (start + nhist)*dt_fs
    time_traj_fs = (starts1[:n_traj] + args.nhist) * dt_fs
    mu_predN_all = np.zeros(n_traj, dtype=np.float32)
    mm_predN_all = np.zeros(n_traj, dtype=np.float32)

    with torch.no_grad():
        for st in range(0, n_traj, args.batch):
            ed = min(n_traj, st + args.batch)
            mu_b = mu_hist1[st:ed].unsqueeze(-1)
            mm_b = mm_hist1[st:ed].unsqueeze(-1)
            mu_predN_all[st:ed] = mu_gru(mu_b).detach().cpu().numpy()
            mm_predN_all[st:ed] = mm_gru(mm_b).detach().cpu().numpy()

    # denorm
    mu_true = mu_true1 * mu_std + mu_mean
    mm_true = mm_true1 * mm_std + mm_mean
    mu_pred = mu_predN_all * mu_std + mu_mean
    mm_pred = mm_predN_all * mm_std + mm_mean

    # masks
    train_mask = np.zeros(n_traj, dtype=np.int32)
    val_mask = np.zeros(n_traj, dtype=np.int32)
    # map train/val idx (based on training windows). For simplicity mark all as unknown,
    # because traj windows length may differ; keep columns but set 0/0.
    # If you need exact mask mapping, you can add mapping by start index.
    df_traj = pd.DataFrame({
        "time_fs": time_traj_fs.astype(np.float32),
        f"mu_{args.dir}_true": mu_true.astype(np.float32),
        f"mu_{args.dir}_pred": mu_pred.astype(np.float32),
        f"m_{args.dir}_true": mm_true.astype(np.float32),
        f"m_{args.dir}_pred": mm_pred.astype(np.float32),
        "is_train": train_mask,
        "is_val": val_mask,
    })
    df_traj.to_csv(outdir / f"traj_mu_m_{args.dir}_0to{args.train_end_fs:g}fs.csv", index=False)
    # 兼容旧脚本命名：再存一份不带范围的文件名
    df_traj.to_csv(outdir / f"traj_mu_m_{args.dir}.csv", index=False)

    # ============================================================
    # 7) Rollout prediction: use 0..train_end as seed, autoregress to pred_end
    #    输出 24-32fs 的预测 CSV（不需要真值，也不打分）
    # ============================================================
    
    # determine indices on the (relative) dq time grid for train_end and pred_end
    i_train_end = _idx_from_time(time_fs, float(args.train_end_fs))
    i_pred_end  = _idx_from_time(time_fs, float(args.pred_end_fs))

    # seed histories at train_end (use last nhist points ending at index i_train_end-1)
    if i_train_end < args.nhist:
        raise RuntimeError("train_end_fs too early for nhist.")
    if i_pred_end < i_train_end:
        raise RuntimeError("pred_end_fs must be >= train_end_fs.")

    # build seed from normalized series
    mu_hist_seed = torch.from_numpy(muN[i_train_end - args.nhist:i_train_end].astype(np.float32)).to(device).unsqueeze(0)  # (1,nhist)
    mm_hist_seed = torch.from_numpy(mmN[i_train_end - args.nhist:i_train_end].astype(np.float32)).to(device).unsqueeze(0)

    # charge history seed: dqN is (N_atoms,T)
    q_hist_seed = torch.from_numpy(dqN[:, i_train_end - args.nhist:i_train_end].T.astype(np.float32)).to(device).transpose(0,1)  # (N,nhist)
    q_hist_seed = q_hist_seed.unsqueeze(0)  # (1,N,nhist)

    # predict on [train_end_fs, pred_end_fs] inclusive (first predicted frame is exactly train_end_fs index)
    steps = int(i_pred_end - i_train_end + 1)
    mu_pred_seqN = []
    mm_pred_seqN = []
    t_pred_seq = []

    with torch.no_grad():
        qh = q_hist_seed.clone()
        muh = mu_hist_seed.clone()
        mmh = mm_hist_seed.clone()

        for s in range(steps):
            qp = hip(qh, phi, rinv)                    # (1,N)
            mup = mu_gru(muh.unsqueeze(-1))            # (1,)
            mmp = mm_gru(mmh.unsqueeze(-1))            # (1,)

            mu_pred_seqN.append(float(mup.item()))
            mm_pred_seqN.append(float(mmp.item()))
            t_pred_seq.append(float(time_fs[i_train_end + s]))

            # feed predictions (pure rollout)
            qh = torch.roll(qh, shifts=-1, dims=2); qh[:, :, -1] = qp
            muh = torch.roll(muh, shifts=-1, dims=1); muh[:, -1] = mup
            mmh = torch.roll(mmh, shifts=-1, dims=1); mmh[:, -1] = mmp

    mu_pred_seq = np.array(mu_pred_seqN, dtype=np.float32) * mu_std + mu_mean
    mm_pred_seq = np.array(mm_pred_seqN, dtype=np.float32) * mm_std + mm_mean
    t_pred_seq = np.array(t_pred_seq, dtype=np.float32)

    df_pred = pd.DataFrame({
        "time_fs": t_pred_seq,
        f"mu_{args.dir}_pred": mu_pred_seq,
        f"m_{args.dir}_pred": mm_pred_seq,
    })
    df_pred.to_csv(outdir / f"pred_mu_m_{args.dir}_{args.train_end_fs:g}to{args.pred_end_fs:g}fs.csv", index=False)
    # 兼容旧脚本命名：再存一份不带范围的文件名
    df_pred.to_csv(outdir / f"pred_mu_m_{args.dir}.csv", index=False)
    # ------------------------------------------------------------
    # NEW: test rollout (no truth used after test_start_fs)

    # If early-stopping gate never opened (e.g., tf_p never dropped below threshold),
    # ensure we still have a usable checkpoint.
    if best_state is None or not np.isfinite(best_metric):
        best_metric = float(row.get("val_loss", float("nan")))
        best_state = {
            "hip": hip.state_dict(),
            "mu_gru": mu_gru.state_dict(),
            "mm_gru": mm_gru.state_dict(),
            "meta": {
                "args": vars(args),
                "best_metric": float(best_metric),
                "mu_mean": mu_mean, "mu_std": mu_std,
                "mm_mean": mm_mean, "mm_std": mm_std,
                "dq_mean": dq_mean, "dq_std": dq_std,
                "dt_fs": dt_fs,
            }
        }
        torch.save(best_state, outdir / "best_model.pt")
        print("[Warn] early-stopping gate never opened; saved last epoch as best_model.pt")

    # ------------------------------------------------------------
    try:
        out_test = _rollout_openloop(args.test_start_fs, args.test_end_fs, return_truth=bool(args.save_test_truth))
        df_test = pd.DataFrame({"time_fs": out_test["time_fs"]})

        # predicted charges
        for ai in range(out_test["q_pred"].shape[1]):
            df_test[f"q{ai+1}_pred"] = out_test["q_pred"][:, ai]

        # predicted mu/m (supports scalar dir mode or full 3-vector)
        def _add_vec_test(_base: str, _arr, _suffix: str):
            _arr = np.asarray(_arr)
            _pfx = "mu" if _base == "mu" else "m"
            if _arr.ndim == 1:
                df_test[f"{_pfx}_{args.dir}_{_suffix}"] = _arr
            elif _arr.ndim == 2 and _arr.shape[1] == 1:
                df_test[f"{_pfx}_{args.dir}_{_suffix}"] = _arr[:, 0]
            elif _arr.ndim == 2 and _arr.shape[1] == 3:
                df_test[f"{_pfx}_x_{_suffix}"] = _arr[:, 0]
                df_test[f"{_pfx}_y_{_suffix}"] = _arr[:, 1]
                df_test[f"{_pfx}_z_{_suffix}"] = _arr[:, 2]
            else:
                _flat = _arr.reshape(_arr.shape[0], -1)
                df_test[f"{_pfx}_{args.dir}_{_suffix}"] = _flat[:, 0]

        _add_vec_test("mu", out_test["mu_pred"], "pred")
        _add_vec_test("mm", out_test["mm_pred"], "pred")

        if bool(args.save_test_truth) and ("q_true" in out_test):
            for ai in range(out_test["q_true"].shape[1]):
                df_test[f"q{ai+1}_true"] = out_test["q_true"][:, ai]
            _add_vec_test("mu", out_test["mu_true"], "true")
            _add_vec_test("mm", out_test["mm_true"], "true")

        test_name = f"test_rollout_{args.dir}_{args.test_start_fs:g}to{args.test_end_fs:g}fs.csv"
        df_test.to_csv(outdir / test_name, index=False)
        print(f"[Saved] {outdir / test_name}")
    except Exception as e:
        print("[Warn] failed to run/save test rollout:", e)

    print(f"[Done] best_metric={best_metric:.4e}")
    print(f"[Saved] {outdir / 'best_model.pt'}")
    print(f"[Saved] {outdir / f'traj_mu_m_{args.dir}_0to{args.train_end_fs:g}fs.csv'}")
    print(f"[Saved] {outdir / f'pred_mu_m_{args.dir}_{args.train_end_fs:g}to{args.pred_end_fs:g}fs.csv'}")


if __name__ == "__main__":
    main()
