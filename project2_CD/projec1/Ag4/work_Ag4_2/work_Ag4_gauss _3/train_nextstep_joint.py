#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced version of train_nextstep_joint.py
- HIPNN blocks strengthened (LayerNorm + Dropout + deeper MLP)
- GRU strengthened (multi-layer, GELU, LayerNorm)
- Add weight decay, dropout, LR scheduler, early stopping
- All new hyperparameters are optional (default = original behavior)
python train_nextstep_joint.py   --xyz Ag4.xyz   --dq_dirs mulliken_x mulliken_y mulliken_z --dq_mode x --dir x   --dmx dm-gauss_x.dat --dmy dm-gauss_y.dat --dmz dm-gauss_z.dat   --mmx mm-COM-gauss_x.dat --mmy mm-COM-gauss_y.dat --mmz mm-COM-gauss_z.dat   --dm_col 3   --tmin 0.0 --nhist 12 --epochs 400 --device cpu   --w_q 1.0 --w_mu 0.5 --w_mm 0.5   --hip_features 64 --hip_layers 4 --hip_dropout 0.1   --gru_hidden 128 --gru_layers 2 --gru_dropout 0.1   --weight_decay 1e-4   --val_frac 0.2 --patience 40 --min_delta 1e-4   --lr 3e-4 --lr_step 80 --lr_gamma 0.5   --outdir out_x_tuned

"""

import argparse, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ------------------ 全局画图风格（不影响模型训练） ------------------
plt.rcParams.update({
    "figure.figsize": (7.0, 3.0),     # 接近 Nature 单栏宽度
    "font.family": "Arial",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "lines.linewidth": 1.4,
})

# ============================================================
#   1) Utility functions
# ============================================================

def load_xyz(xyz_file):
    atoms = []
    coords = []
    with open(xyz_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(atoms), np.array(coords)

def charge_cons_penalty(q):
    return torch.mean((q.sum() - q.sum().detach()) ** 2)


def load_dm_file(dmfile, col_idx):
    """Load (time, value) from dm/mm .dat file.

    Parameters
    ----------
    dmfile : str
        Path to the .dat file written by GPAW (time in a.u.).
    col_idx : int
        Zero-based column index of the desired component (e.g., dmx / mmx).

    Returns
    -------
    t_fs : ndarray
        Time in femtoseconds.
    values : ndarray
        Selected component values.
    """
    arr = np.loadtxt(dmfile)
    # First column is time in atomic units -> convert to fs
    t_au = arr[:, 0]
    au_to_fs = 0.02418884326505
    t_fs = t_au * au_to_fs
    values = arr[:, col_idx]
    return t_fs, values

def load_joint_histories(n_hist, tmin, dt, times, values):
    items = []
    nt = len(times)
    for i in range(nt):
        if times[i] < tmin:
            continue
        idx = math.floor((times[i] - tmin) / dt)
        if idx < n_hist or (i+1) >= nt:
            continue
        hist = values[idx - n_hist + 1: idx + 1]
        target = values[idx + 1]
        items.append((i, hist, target))
    return items



# ============================================================
#   2) Enhanced HIPNN
# ============================================================

class InteractionLayer(nn.Module):
    def __init__(self, n_features, n_sensitivity=8, rcut=10.0, dropout=0.0):
        super().__init__()
        self.nf = n_features
        self.ns = n_sensitivity
        self.rc = rcut
        self.sens = nn.Parameter(torch.randn(self.ns, 2) * 0.1)   # (mu, sigma)

        self.edge = nn.Linear(self.ns, self.nf, bias=False)
        self.self_lin = nn.Linear(self.nf, self.nf)

        self.act = nn.Softplus()
        self.ln = nn.LayerNorm(self.nf)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feat, dist_ij):
        phi = torch.where(
            dist_ij < self.rc,
            torch.cos((math.pi * dist_ij) / (2.0 * self.rc)) ** 2,
            torch.zeros_like(dist_ij)
        )
        rinv = torch.where(dist_ij > 0, 1.0 / dist_ij, torch.zeros_like(dist_ij))

        mu, sig = self.sens[:, 0], torch.abs(self.sens[:, 1]) + 1e-6
        x = rinv.unsqueeze(-1)
        s = torch.exp(-((x - mu.view(1,1,-1)) / (2 * sig.view(1,1,-1))) ** 2) * phi.unsqueeze(-1)

        e = self.edge(s.reshape(-1, self.ns)).view(dist_ij.size(0), -1, self.nf)
        msg = (e * feat.unsqueeze(0)).sum(dim=1)

        h = self.self_lin(feat) + msg
        h = self.ln(h)
        h = self.act(h)
        h = self.dropout(h)
        return h


class HIPNNCharges(nn.Module):
    def __init__(
        self,
        n_hist=9,
        n_features=32,
        n_interaction=3,
        n_sensitivity=8,
        cutoff=10.0,
        dropout=0.0,
    ):
        super().__init__()

        self.inp = nn.Linear(n_hist, n_features)
        self.inp_ln = nn.LayerNorm(n_features)

        self.blocks = nn.ModuleList()
        self.outs = nn.ModuleList()

        for _ in range(n_interaction):
            self.blocks.append(InteractionLayer(n_features, n_sensitivity, cutoff, dropout=dropout))
            self.outs.append(
                nn.Sequential(
                    nn.Linear(n_features, n_features),
                    nn.Softplus(),
                    nn.Dropout(dropout),
                    nn.Linear(n_features, 1),
                )
            )

    def forward(self, q_hist, dist_ij):
        h = self.inp(q_hist)
        h = self.inp_ln(h)

        pred = 0.0
        for blk, head in zip(self.blocks, self.outs):
            h = blk(h, dist_ij)
            pred = pred + head(h).squeeze(-1)
        return pred


# ============================================================
#   3) Enhanced GRU head
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
        out, _ = self.gru(x)
        h = out[:, -1, :]
        return self.fc(h)



# ============================================================
#   4) Main training script
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    # --------------------------
    # 原有参数
    # --------------------------
    parser.add_argument("--xyz", required=True)
    parser.add_argument("--dq_dirs", nargs="+")
    parser.add_argument("--dq_mode", choices=["x","y","z"])
    parser.add_argument("--dir", choices=["x","y","z"], required=True)
    parser.add_argument("--dmx"); parser.add_argument("--dmy"); parser.add_argument("--dmz")
    parser.add_argument("--mmx"); parser.add_argument("--mmy"); parser.add_argument("--mmz")
    parser.add_argument("--dm_col", type=int, default=3,
                        help="Column index (1-based) in dm dat file for the dipole component")
    parser.add_argument("--mm_col", type=int, default=2,
                        help="Column index (1-based) in mm dat file for the magnetic dipole component")
    parser.add_argument("--tmin", type=float, default=5.0)
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

    # -----------------------------------------------------
    #  新增超参数
    # -----------------------------------------------------
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

    args = parser.parse_args()
    device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load geometry
    # -------------------------
    atoms, coords = load_xyz(args.xyz)
    pos_t = torch.tensor(coords, dtype=torch.float32, device=device)
    dist_ij = torch.cdist(pos_t, pos_t)

    # Column indices (zero-based) for dm/mm components
    dm_idx = args.dm_col - 1
    mm_idx = args.mm_col - 1

    # -------------------------
    # Load dipole/mag-dipole
    # -------------------------
    t_x, mu_x = load_dm_file(args.dmx, dm_idx)
    t_y, mu_y = load_dm_file(args.dmy, dm_idx)
    t_z, mu_z = load_dm_file(args.dmz, dm_idx)

    t_x2, mm_x = load_dm_file(args.mmx, mm_idx)
    t_y2, mm_y = load_dm_file(args.mmy, mm_idx)
    t_z2, mm_z = load_dm_file(args.mmz, mm_idx)

    # -------------------------
    # Determine working direction
    # -------------------------
    mu_map = {'x': mu_x, 'y': mu_y, 'z': mu_z}
    mm_map = {'x': mm_x, 'y': mm_y, 'z': mm_z}
    tt_map = {'x': t_x,  'y': t_y,  'z': t_z}

    mu_vals = mu_map[args.dir]
    mm_vals = mm_map[args.dir]
    times = tt_map[args.dir]

    # ----- μ, m 标准化 -----
    mu_mean = mu_vals.mean()
    mu_std = mu_vals.std()
    mm_mean = mm_vals.mean()
    mm_std = mm_vals.std()

    mu_vals_norm = (mu_vals - mu_mean) / mu_std
    mm_vals_norm = (mm_vals - mm_mean) / mm_std

    # 转成 torch 标量，给一致性损失用
    mu_mean_t = torch.tensor(mu_mean, dtype=torch.float32, device=device)
    mu_std_t = torch.tensor(mu_std, dtype=torch.float32, device=device)
    mm_mean_t = torch.tensor(mm_mean, dtype=torch.float32, device=device)
    mm_std_t = torch.tensor(mm_std, dtype=torch.float32, device=device)

    # -------------------------
    # Histories for μ and m（用标准化后的值）
    # -------------------------
    dt = times[1] - times[0]
    items_mu = load_joint_histories(args.nhist, args.tmin, dt, times, mu_vals_norm)
    items_mm = load_joint_histories(args.nhist, args.tmin, dt, times, mm_vals_norm)

    # -------------------------
    # Load charges dq from directories containing dq_t.npy and time_fs.npy
    # -------------------------

    def load_dq_from_dir(dirname):
        dq_path = Path(dirname) / "dq_t.npy"
        t_path = Path(dirname) / "time_fs.npy"

        if not dq_path.exists():
            raise RuntimeError(f"Cannot find dq_t.npy in {dirname}")
        if not t_path.exists():
            raise RuntimeError(f"Cannot find time_fs.npy in {dirname}")

        dq = np.load(dq_path).T      # (N_atoms, N_time)
        t = np.load(t_path)          # (N_time,)
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

    dq_times = time_list[0]
    dq_all = np.stack(dq_list, axis=0)   # (3, N_atoms, N_time)

    mode_map = {"x": 0, "y": 1, "z": 2}
    dq_sel = dq_all[mode_map[args.dq_mode]]    # (N_atoms, N_time)

    dq_mean = dq_sel.mean()
    dq_std  = dq_sel.std()
    dqN = (dq_sel - dq_mean) / dq_std

    items_q = []
    nt = dqN.shape[1]
    for i in range(nt - args.nhist - 1):
        hist = dqN[:, i:i+args.nhist]
        targ = dqN[:, i+args.nhist]
        items_q.append((i, hist, targ))

    # ============================================================
    #  Instantiate models
    # ============================================================

    hip = HIPNNCharges(
        n_hist=args.nhist,
        n_features=args.hip_features,
        n_interaction=args.hip_layers,
        n_sensitivity=8,
        cutoff=10.0,
        dropout=args.hip_dropout,
    ).to(device)

    mu_gru = GRUHead1D(
        in_dim=1,
        hidden=args.gru_hidden,
        num_layers=args.gru_layers,
        dropout=args.gru_dropout,
    ).to(device)

    mm_gru = GRUHead1D(
        in_dim=1,
        hidden=args.gru_hidden,
        num_layers=args.gru_layers,
        dropout=args.gru_dropout,
    ).to(device)

    optim_all = optim.AdamW(
        list(hip.parameters()) + list(mu_gru.parameters()) + list(mm_gru.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    mse = nn.MSELoss()

    # ---------------------
    # Train / Val split
    # ---------------------
    rng = np.random.default_rng(42)

    n_total = min(len(items_q), len(items_mu), len(items_mm))
    if n_total == 0:
        raise RuntimeError("No training windows built; check nhist and tmin settings.")

    idx_all = np.arange(n_total)
    rng.shuffle(idx_all)

    if args.val_frac > 0 and n_total > 10:
        n_val = max(1, int(args.val_frac * n_total))
        val_idx = idx_all[:n_val]
        train_idx = idx_all[n_val:]
    else:
        train_idx, val_idx = idx_all, []

    if args.lr_step > 0:
        scheduler = optim.lr_scheduler.StepLR(optim_all, step_size=args.lr_step, gamma=args.lr_gamma)
    else:
        scheduler = None

    log_rows = []
    best_metric = float("inf")
    best_state = None
    no_improve = 0

    # ============================================================
    #   Training loop
    # ============================================================

    for ep in range(args.epochs):

        hip.train(); mu_gru.train(); mm_gru.train()

        if len(train_idx) == 0:
            cur_idx = idx_all
        else:
            bs = min(args.batch, len(train_idx))
            cur_idx = rng.choice(train_idx, size=bs, replace=False)

        if len(cur_idx) == 0:
            raise RuntimeError("Current batch has zero samples; check data construction.")

        lq = lmu = lmm = 0.0
        loss_sum = 0.0

        optim_all.zero_grad()

        for k in cur_idx:
            # -------- charges --------
            _, q_hist, q_next = items_q[k]
            q_hist_t = torch.tensor(q_hist, dtype=torch.float32, device=device)
            q_next_t = torch.tensor(q_next, dtype=torch.float32, device=device)

            q_predN = hip(q_hist_t, dist_ij)
            loss_q = mse(q_predN, q_next_t) + args.alpha_qsum * charge_cons_penalty(q_predN)

            # -------- μ（标准化空间）--------
            _, mu_hist, mu_next = items_mu[k]
            mu_hist_t = torch.tensor(mu_hist.reshape(1,-1,1), dtype=torch.float32, device=device)
            mu_next_t = torch.tensor([[mu_next]], dtype=torch.float32, device=device)

            mu_predN = mu_gru(mu_hist_t)
            loss_mu_main = mse(mu_predN, mu_next_t)

            # consistency：用同一套标准化
            q_pred_real = q_predN * dq_std + dq_mean
            comp = {'x':0,'y':1,'z':2}[args.dir]
            mu_base_real = torch.dot(q_pred_real, pos_t[:,comp])
            mu_baseN = (mu_base_real - mu_mean_t) / mu_std_t

            loss_mu_cons = mse(mu_predN.squeeze(), mu_baseN)
            loss_mu = loss_mu_main + args.lambda_mu_consistency * loss_mu_cons

            # -------- m（标准化空间）--------
            _, mm_hist, mm_next = items_mm[k]
            mm_hist_t = torch.tensor(mm_hist.reshape(1,-1,1), dtype=torch.float32, device=device)
            mm_next_t = torch.tensor([[mm_next]], dtype=torch.float32, device=device)

            mm_predN = mm_gru(mm_hist_t)
            loss_mm = mse(mm_predN, mm_next_t)

            loss = args.w_q * loss_q + args.w_mu * loss_mu + args.w_mm * loss_mm
            loss.backward()

            loss_sum += float(loss.item())
            lq += float(loss_q.item())
            lmu += float(loss_mu.item())
            lmm += float(loss_mm.item())

        optim_all.step()
        if scheduler is not None:
            scheduler.step()

        row = dict(
            epoch=ep,
            loss=loss_sum / len(cur_idx),
            l_q=lq / len(cur_idx),
            l_mu=lmu / len(cur_idx),
            l_mm=lmm / len(cur_idx),
        )

        # -------- Validation --------
        if len(val_idx) > 0:
            hip.eval(); mu_gru.eval(); mm_gru.eval()
            vloss = 0.0
            with torch.no_grad():
                for k in val_idx:
                    _, q_hist, q_next = items_q[k]
                    q_hist_t = torch.tensor(q_hist, dtype=torch.float32, device=device)
                    q_next_t = torch.tensor(q_next, dtype=torch.float32, device=device)
                    q_predN = hip(q_hist_t, dist_ij)
                    loss_q = mse(q_predN, q_next_t)

                    _, mu_hist, mu_next = items_mu[k]
                    mu_hist_t = torch.tensor(mu_hist.reshape(1,-1,1), dtype=torch.float32, device=device)
                    mu_next_t = torch.tensor([[mu_next]], dtype=torch.float32, device=device)
                    mu_predN = mu_gru(mu_hist_t)
                    loss_mu_main = mse(mu_predN, mu_next_t)

                    _, mm_hist, mm_next = items_mm[k]
                    mm_hist_t = torch.tensor(mm_hist.reshape(1,-1,1), dtype=torch.float32, device=device)
                    mm_next_t = torch.tensor([[mm_next]], dtype=torch.float32, device=device)
                    mm_predN = mm_gru(mm_hist_t)
                    loss_mm = mse(mm_predN, mm_next_t)

                    vloss += float((args.w_q*loss_q + args.w_mu*loss_mu_main + args.w_mm*loss_mm).item())

            row["val_loss"] = vloss / len(val_idx)
            metric = row["val_loss"]
        else:
            metric = row["loss"]

        log_rows.append(row)

        if (ep+1) % 10 == 0:
            msg = f"[{args.dir}] Ep {ep+1:03d} Loss {row['loss']:.3e}"
            if "val_loss" in row:
                msg += f" | Val {row['val_loss']:.3e}"
            print(msg)

        # Early stopping
        if metric + args.min_delta < best_metric:
            best_metric = metric
            no_improve = 0
            best_state = {
                "hip": hip.state_dict(),
                "mu_gru": mu_gru.state_dict(),
                "mm_gru": mm_gru.state_dict(),
            }
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            print(f"[Early Stopping] epoch {ep+1}")
            break

    # Save log
    pd.DataFrame(log_rows).to_csv(outdir / f"log_{args.dir}.csv", index=False)

    # Load best model
    if best_state:
        hip.load_state_dict(best_state["hip"])
        mu_gru.load_state_dict(best_state["mu_gru"])
        mm_gru.load_state_dict(best_state["mm_gru"])

    # ============================================================
    #   Evaluation & plotting: μ and m (truth vs train/val pred)
    # ============================================================
    hip.eval()
    mu_gru.eval()
    mm_gru.eval()

    mu_times = np.array([times[i_idx] for (i_idx, _, _) in items_mu[:n_total]])
    mu_true = np.array([tgt for (_, _, tgt) in items_mu[:n_total]])
    mm_true = np.array([tgt for (_, _, tgt) in items_mm[:n_total]])

    mu_pred_all = np.zeros(n_total, dtype=float)
    mm_pred_all = np.zeros(n_total, dtype=float)

    with torch.no_grad():
        for k in range(n_total):
            _, mu_hist, _ = items_mu[k]
            mu_hist_t = torch.tensor(mu_hist.reshape(1, -1, 1),
                                     dtype=torch.float32, device=device)
            mu_pred_all[k] = mu_gru(mu_hist_t).item()

            _, mm_hist, _ = items_mm[k]
            mm_hist_t = torch.tensor(mm_hist.reshape(1, -1, 1),
                                     dtype=torch.float32, device=device)
            mm_pred_all[k] = mm_gru(mm_hist_t).item()

    # 反标准化回物理值
    mu_true = mu_true * mu_std + mu_mean
    mu_pred_all = mu_pred_all * mu_std + mu_mean
    mm_true = mm_true * mm_std + mm_mean
    mm_pred_all = mm_pred_all * mm_std + mm_mean

    train_mask = np.zeros(n_total, dtype=bool)
    val_mask = np.zeros(n_total, dtype=bool)
    train_mask[train_idx] = True
    if len(val_idx) > 0:
        val_mask[val_idx] = True

    mu_pred_train = np.full(n_total, np.nan)
    mu_pred_val = np.full(n_total, np.nan)
    mu_pred_train[train_mask] = mu_pred_all[train_mask]
    mu_pred_val[val_mask] = mu_pred_all[val_mask]

    mm_pred_train = np.full(n_total, np.nan)
    mm_pred_val = np.full(n_total, np.nan)
    mm_pred_train[train_mask] = mm_pred_all[train_mask]
    mm_pred_val[val_mask] = mm_pred_all[val_mask]

    def compute_sigma(true_arr, pred_arr, mask):
        if mask.sum() > 0:
            diff = true_arr[mask] - pred_arr[mask]
            return float(np.sqrt(np.mean(diff ** 2)))
        else:
            diff = true_arr - pred_arr
            return float(np.sqrt(np.mean(diff ** 2)))

    sigma_mu = compute_sigma(mu_true, mu_pred_all, val_mask)
    sigma_mm = compute_sigma(mm_true, mm_pred_all, val_mask)

    # 不确定性带围绕预测值
    mu_lower = mu_pred_all - sigma_mu
    mu_upper = mu_pred_all + sigma_mu
    mm_lower = mm_pred_all - sigma_mm
    mm_upper = mm_pred_all + sigma_mm

    # ========= 新增：保存偶极 & 磁偶极到 CSV（重新模拟结果） =========
    df_series = pd.DataFrame({
        "time_fs": mu_times,
        f"mu_{args.dir}_true": mu_true,
        f"mu_{args.dir}_pred": mu_pred_all,
        f"m_{args.dir}_true": mm_true,
        f"m_{args.dir}_pred": mm_pred_all,
        "is_train": train_mask.astype(int),
        "is_val": val_mask.astype(int),
    })
    df_series.to_csv(outdir / f"traj_mu_m_{args.dir}.csv", index=False)
    # =============================================================

    # 一致的配色：不确定性带用预测线颜色的浅色
    truth_color = "black"
    train_color = "#1f77b4"   # 蓝
    val_color = "#d62728"     # 红
    band_alpha = 0.18         # 半透明

    # ---------- 画偶极 μ ----------
    fig, ax = plt.subplots()

    # 不确定性带：使用 train 颜色的淡色
    ax.fill_between(mu_times, mu_lower, mu_upper,
                    color=train_color, alpha=band_alpha,
                    label=r"$\mu$ uncertainty band", zorder=1)

    ax.plot(mu_times, mu_true, color=truth_color, linewidth=1.8,
            label="Truth", zorder=3)
    ax.plot(mu_times, mu_pred_train, color=train_color, linewidth=1.4,
            label="Pred (train)", alpha=0.95, zorder=4)
    if val_mask.sum() > 0:
        ax.plot(mu_times, mu_pred_val, color=val_color, linewidth=1.4,
                label="Pred (val)", alpha=0.95, zorder=4)

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(f"μ_{args.dir} (a.u.)")
    ax.set_title(f"Dipole μ_{args.dir}: truth vs prediction")

    # 自动设置 y 轴范围并加一点 padding
    ymin = min(mu_true.min(), mu_lower.min())
    ymax = max(mu_true.max(), mu_upper.max())
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(mu_times.min(), mu_times.max())

    # 四周封闭框 + 刻度朝内并加粗
    ax.tick_params(direction="in", length=5, width=1.2, top=True, right=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    # 图例放到外侧，不挡曲线
    ax.legend(frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(outdir / f"mu_compare_{args.dir}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    # ---------- 画磁偶极 m ----------
    fig, ax = plt.subplots()

    ax.fill_between(mu_times, mm_lower, mm_upper,
                    color=train_color, alpha=band_alpha,
                    label=r"$m$ uncertainty band", zorder=1)

    ax.plot(mu_times, mm_true, color=truth_color, linewidth=1.8,
            label="Truth", zorder=3)
    ax.plot(mu_times, mm_pred_train, color=train_color, linewidth=1.4,
            label="Pred (train)", alpha=0.95, zorder=4)
    if val_mask.sum() > 0:
        ax.plot(mu_times, mm_pred_val, color=val_color, linewidth=1.4,
                label="Pred (val)", alpha=0.95, zorder=4)

    ax.set_xlabel("Time (fs)")
    ax.set_ylabel(f"m_{args.dir} (a.u.)")
    ax.set_title(f"Magnetic dipole m_{args.dir}: truth vs prediction")

    ymin = min(mm_true.min(), mm_lower.min())
    ymax = max(mm_true.max(), mm_upper.max())
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlim(mu_times.min(), mu_times.max())

    ax.tick_params(direction="in", length=5, width=1.2, top=True, right=True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.legend(frameon=False, loc="upper left",
              bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.tight_layout()
    fig.savefig(outdir / f"mm_compare_{args.dir}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved comparison plots to {outdir}")
    print(f"Training finished. Best metric={best_metric:.4e}")


if __name__ == "__main__":
    main()
