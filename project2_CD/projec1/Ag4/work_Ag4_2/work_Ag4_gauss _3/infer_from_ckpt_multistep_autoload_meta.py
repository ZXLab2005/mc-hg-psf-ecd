#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
infer_from_ckpt_multistep_autoload_meta.py
-----------------------------------------
✅ 目标：
- 直接读取 multistep 训练脚本生成的 ckpt(best_model.pt)
- 从 ckpt["meta"]["args"] 自动取 nhist/hip/gru 等超参数
- 从 ckpt["meta"] 自动取 dq/mu/mm 的归一化统计量（dq_mean/std, mu_mean/std, mm_mean/std）
- 使用与训练脚本 *完全一致* 的 HIPNNChargesBatch / GRUHead1D / load_dm_file（含 a.u.→fs）
- 在 [start_fs, pred_end_fs] 区间做 rollout 推理（自回归）
- 输出 24fs 之后(可自定义 start_fs)：
  * 每个原子 dq 的真值 vs 预测
  * μ 与 m 的真值 vs 预测
  * 可选画图

用法示例：
python infer_from_ckpt_multistep_autoload_meta.py \
  --train_script train_nextstep_joint_schemeA_multistep_normtrain_amp_strat_rms_fixed_v3.py \
  --ckpt out_x_multistep_from_onestep_v3/best_model.pt \
  --xyz Ag4.xyz \
  --dq_dirs mulliken_x mulliken_y mulliken_z --dq_mode x --dir x \
  --dmx dm-gauss_x.dat --dmy dm-gauss_x.dat --dmz dm-gauss_x.dat \
  --mmx mm-COM-gauss_x.dat --mmy mm-COM-gauss_x.dat --mmz mm-COM-gauss_x.dat \
  --dm_col 3 --mm_col 2 \
  --start_fs 24 --pred_end_fs 32 \
  --device cpu --torch_threads 64 \
  --outdir infer_x_compare_24to32_autoload \
  --plot

注意：
- 本脚本不再“手写模型结构”，而是从训练脚本 import 类，保证 state_dict key 100% 对齐。
"""

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------
# basic helpers
# ---------------------------
def _import_train_module(train_script: str):
    p = Path(train_script)
    if not p.exists():
        raise FileNotFoundError(f"--train_script not found: {train_script}")
    spec = importlib.util.spec_from_file_location("train_mod", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _load_ckpt(path: str, map_location="cpu"):
    # 兼容旧 torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _nearest_idx(t: np.ndarray, x: float) -> int:
    t = np.asarray(t, dtype=np.float32)
    return int(np.argmin(np.abs(t - float(x))))


def _read_xyz_positions(xyz_path: str) -> np.ndarray:
    """Read first frame positions (Angstrom) from simple XYZ."""
    lines = Path(xyz_path).read_text().strip().splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"Bad xyz: {xyz_path}")
    n = int(lines[0].strip())
    body = lines[2:2+n]
    pos = []
    for ln in body:
        parts = ln.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        pos.append([x, y, z])
    return np.array(pos, dtype=np.float32)


def _make_phi_rinv_from_xyz(xyz_path: str, device: torch.device, rcut: float = 10.0):
    pos = _read_xyz_positions(xyz_path)  # (N,3)
    pos_t = torch.as_tensor(pos, dtype=torch.float32, device=device)
    dist_ij = torch.cdist(pos_t, pos_t)  # (N,N)
    phi = torch.where(
        dist_ij < rcut,
        torch.cos((math.pi * dist_ij) / (2.0 * rcut)) ** 2,
        torch.zeros_like(dist_ij),
    )
    rinv = torch.where(dist_ij > 0, 1.0 / dist_ij, torch.zeros_like(dist_ij))
    return phi, rinv


def _load_dq_from_dir(dirname: str):
    d = Path(dirname)
    dq_path = d / "dq_t.npy"
    t_path = d / "time_fs.npy"
    if not dq_path.exists():
        raise RuntimeError(f"Cannot find dq_t.npy in {dirname}")
    if not t_path.exists():
        raise RuntimeError(f"Cannot find time_fs.npy in {dirname}")
    dq = np.load(dq_path).T.astype(np.float32)  # (N_atoms, N_time)
    t = np.load(t_path).astype(np.float32)      # (N_time,)
    return dq, t


def _broadcastable_dq_stats(dq_mean, dq_std):
    """
    dq_mean/std can be:
    - float (global)
    - (N,1) per_atom saved by training
    - (N,) some older variants
    Return as numpy arrays broadcastable to (N,T).
    """
    dq_mean = _to_np(dq_mean)
    dq_std = _to_np(dq_std)
    if np.isscalar(dq_mean):
        return float(dq_mean), float(dq_std)
    dq_mean = np.asarray(dq_mean, dtype=np.float32)
    dq_std = np.asarray(dq_std, dtype=np.float32)
    if dq_mean.ndim == 2 and dq_mean.shape[1] == 1:
        dq_mean = dq_mean  # (N,1)
        dq_std = dq_std
    elif dq_mean.ndim == 1:
        dq_mean = dq_mean[:, None]  # (N,1)
        dq_std = dq_std[:, None]
    else:
        raise RuntimeError(f"Unexpected dq_mean shape: {dq_mean.shape}")
    return dq_mean, dq_std


@torch.no_grad()
def rollout_compare(
    hip,
    mu_gru,
    mm_gru,
    phi: torch.Tensor,
    rinv: torch.Tensor,
    dq_sel: np.ndarray,         # (N,Tq) physical
    dq_times: np.ndarray,       # (Tq,) fs
    mu_vals: np.ndarray,        # (Tm,) physical
    mm_vals: np.ndarray,        # (Tm,) physical
    mu_times: np.ndarray,       # (Tm,) fs
    start_fs: float,
    pred_end_fs: float,
    nhist: int,
    dq_mean, dq_std,
    mu_mean: float, mu_std: float,
    mm_mean: float, mm_std: float,
    device: torch.device,
):
    # indices
    q_i0 = _nearest_idx(dq_times, start_fs)
    q_i1 = _nearest_idx(dq_times, pred_end_fs)
    m_i0 = _nearest_idx(mu_times, start_fs)
    m_i1 = _nearest_idx(mu_times, pred_end_fs)
    if q_i1 < q_i0 or m_i1 < m_i0:
        raise RuntimeError("pred_end_fs < start_fs on time grids")

    if q_i0 - nhist < 0:
        raise RuntimeError(f"Not enough dq history before start_fs={start_fs} (need nhist={nhist})")
    if m_i0 - nhist < 0:
        raise RuntimeError(f"Not enough mu history before start_fs={start_fs} (need nhist={nhist})")

    # stats broadcast
    dq_mean_b, dq_std_b = _broadcastable_dq_stats(dq_mean, dq_std)

    # normalize full series
    dqN = (dq_sel - dq_mean_b) / (dq_std_b + 1e-12)
    muN = (mu_vals - mu_mean) / (mu_std + 1e-12)
    mmN = (mm_vals - mm_mean) / (mm_std + 1e-12)

    # rollout length: use common steps to keep aligned
    steps_q = q_i1 - q_i0 + 1
    steps_m = m_i1 - m_i0 + 1
    steps = min(steps_q, steps_m)
    if steps <= 0:
        raise RuntimeError("No steps to rollout")

    # seed histories (normalized)
    # q_hist: (1,N,nhist)
    q_hist = torch.from_numpy(dqN[:, q_i0 - nhist:q_i0].T).to(device)  # (nhist,N)
    q_hist = q_hist.unsqueeze(0).transpose(1, 2).contiguous()  # (1,N,nhist)

    mu_hist = torch.from_numpy(muN[m_i0 - nhist:m_i0]).to(device).unsqueeze(0)  # (1,nhist)
    mm_hist = torch.from_numpy(mmN[m_i0 - nhist:m_i0]).to(device).unsqueeze(0)

    q_predN_seq = []
    mu_predN_seq = []
    mm_predN_seq = []

    for _ in range(steps):
        q_nextN = hip(q_hist, phi, rinv)                  # (1,N)
        mu_nextN = mu_gru(mu_hist.unsqueeze(-1))          # (1,)
        mm_nextN = mm_gru(mm_hist.unsqueeze(-1))          # (1,)

        q_predN_seq.append(q_nextN.squeeze(0).detach().cpu().numpy())  # (N,)
        mu_predN_seq.append(float(mu_nextN.item()))
        mm_predN_seq.append(float(mm_nextN.item()))

        # update histories (use normalized preds)
        q_hist = torch.cat([q_hist[:, :, 1:], q_nextN.unsqueeze(-1)], dim=-1)
        mu_hist = torch.cat([mu_hist[:, 1:], mu_nextN.unsqueeze(0)], dim=1)
        mm_hist = torch.cat([mm_hist[:, 1:], mm_nextN.unsqueeze(0)], dim=1)

    q_predN = np.stack(q_predN_seq, axis=1).astype(np.float32)   # (N,steps)
    mu_predN = np.array(mu_predN_seq, dtype=np.float32)          # (steps,)
    mm_predN = np.array(mm_predN_seq, dtype=np.float32)

    # de-norm
    q_pred = q_predN * (dq_std_b + 1e-12) + dq_mean_b
    mu_pred = mu_predN * (mu_std + 1e-12) + mu_mean
    mm_pred = mm_predN * (mm_std + 1e-12) + mm_mean

    # true slices in physical units
    q_true = dq_sel[:, q_i0:q_i0 + steps]
    mu_true = mu_vals[m_i0:m_i0 + steps]
    mm_true = mm_vals[m_i0:m_i0 + steps]

    t_q = dq_times[q_i0:q_i0 + steps]
    t_m = mu_times[m_i0:m_i0 + steps]

    return (t_q, q_true, q_pred), (t_m, mu_true, mu_pred, mm_true, mm_pred)


def _metrics_1d(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    # pearson
    if y_true.std() < 1e-12 or y_pred.std() < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"mae": mae, "rmse": rmse, "corr": corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", required=True, help="你的训练脚本 .py，用来 import 类/函数保证结构一致")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xyz", required=True)

    ap.add_argument("--dq_dirs", nargs="+", required=True)
    ap.add_argument("--dq_mode", choices=["x", "y", "z"], required=True)
    ap.add_argument("--dir", choices=["x", "y", "z"], required=True)

    ap.add_argument("--dmx", required=True); ap.add_argument("--dmy", required=True); ap.add_argument("--dmz", required=True)
    ap.add_argument("--mmx", required=True); ap.add_argument("--mmy", required=True); ap.add_argument("--mmz", required=True)
    ap.add_argument("--dm_col", type=int, default=3)  # 1-based
    ap.add_argument("--mm_col", type=int, default=2)  # 1-based

    ap.add_argument("--start_fs", type=float, default=24.0)
    ap.add_argument("--pred_end_fs", type=float, default=32.0)

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--torch_threads", type=int, default=0)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(int(args.torch_threads))
        try:
            torch.set_num_interop_threads(max(1, int(args.torch_threads // 4)))
        except Exception:
            pass

    device = torch.device(args.device)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) import training module (classes + load_dm_file)
    train_mod = _import_train_module(args.train_script)
    HIP = getattr(train_mod, "HIPNNChargesBatch")
    GRU = getattr(train_mod, "GRUHead1D")
    load_dm_file = getattr(train_mod, "load_dm_file")

    # 2) load ckpt and meta
    ckpt = _load_ckpt(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})
    if not isinstance(meta, dict):
        raise RuntimeError("ckpt.meta missing or invalid")
    meta_args = meta.get("args", {})
    if not isinstance(meta_args, dict):
        raise RuntimeError("ckpt.meta.args missing or invalid")

    # auto hyperparams (fallback to meta_args or raise)
    nhist = int(meta_args.get("nhist"))
    hip_features = int(meta_args.get("hip_features"))
    hip_layers = int(meta_args.get("hip_layers"))
    hip_dropout = float(meta_args.get("hip_dropout", 0.0))

    gru_hidden = int(meta_args.get("gru_hidden"))
    gru_layers = int(meta_args.get("gru_layers"))
    gru_dropout = float(meta_args.get("gru_dropout", 0.0))

    # 3) build exact model
    hip = HIP(n_hist=nhist, n_features=hip_features, n_interaction=hip_layers,
              n_sensitivity=8, dropout=hip_dropout).to(device)
    mu_gru = GRU(in_dim=1, hidden=gru_hidden, num_layers=gru_layers, dropout=gru_dropout).to(device)
    mm_gru = GRU(in_dim=1, hidden=gru_hidden, num_layers=gru_layers, dropout=gru_dropout).to(device)

    # 4) load weights strict=True (must match)
    if isinstance(ckpt, dict) and ("hip" in ckpt and "mu_gru" in ckpt and "mm_gru" in ckpt):
        hip.load_state_dict(ckpt["hip"], strict=True)
        mu_gru.load_state_dict(ckpt["mu_gru"], strict=True)
        mm_gru.load_state_dict(ckpt["mm_gru"], strict=True)
    else:
        raise RuntimeError("Unsupported ckpt format: expect keys hip/mu_gru/mm_gru/meta")

    hip.eval(); mu_gru.eval(); mm_gru.eval()

    # 5) norms from meta (exact training stats)
    if not all(k in meta for k in ["dq_mean", "dq_std", "mu_mean", "mu_std", "mm_mean", "mm_std"]):
        raise RuntimeError("ckpt.meta missing normalization stats (dq/mu/mm mean/std)")
    dq_mean = meta["dq_mean"]
    dq_std = meta["dq_std"]
    mu_mean = float(_to_np(meta["mu_mean"]))
    mu_std = float(_to_np(meta["mu_std"]))
    mm_mean = float(_to_np(meta["mm_mean"]))
    mm_std = float(_to_np(meta["mm_std"]))

    # 6) load dq (charges)
    dq_list, t_list = [], []
    for d in args.dq_dirs:
        dq, t = _load_dq_from_dir(d)
        dq_list.append(dq)
        t_list.append(t)
    for i in range(1, len(t_list)):
        if not np.allclose(t_list[i], t_list[0]):
            raise RuntimeError("time_fs.npy mismatch among dq_dirs")
    dq_times = t_list[0]
    dq_all = np.stack(dq_list, axis=0)  # (3,N,T)
    mode_map = {"x": 0, "y": 1, "z": 2}
    dq_sel = dq_all[mode_map[args.dq_mode]]  # (N,T)

    # 7) load mu/mm via training script load_dm_file (a.u.->fs fixed)
    dm_idx0 = args.dm_col - 1
    mm_idx0 = args.mm_col - 1

    t_x, mu_x = load_dm_file(args.dmx, dm_idx0)
    t_y, mu_y = load_dm_file(args.dmy, dm_idx0)
    t_z, mu_z = load_dm_file(args.dmz, dm_idx0)

    t_x2, mm_x = load_dm_file(args.mmx, mm_idx0)
    t_y2, mm_y = load_dm_file(args.mmy, mm_idx0)
    t_z2, mm_z = load_dm_file(args.mmz, mm_idx0)

    mu_map = {"x": mu_x, "y": mu_y, "z": mu_z}
    mm_map = {"x": mm_x, "y": mm_y, "z": mm_z}
    t_map = {"x": t_x, "y": t_y, "z": t_z}

    mu_vals = np.asarray(mu_map[args.dir], dtype=np.float32)
    mm_vals = np.asarray(mm_map[args.dir], dtype=np.float32)
    mu_times = np.asarray(t_map[args.dir], dtype=np.float32)

    # 8) geometry => phi/rinv
    phi, rinv = _make_phi_rinv_from_xyz(args.xyz, device=device, rcut=10.0)

    # 9) rollout + compare
    (t_q, q_true, q_pred), (t_m, mu_true, mu_pred, mm_true, mm_pred) = rollout_compare(
        hip=hip, mu_gru=mu_gru, mm_gru=mm_gru,
        phi=phi, rinv=rinv,
        dq_sel=dq_sel, dq_times=dq_times,
        mu_vals=mu_vals, mm_vals=mm_vals, mu_times=mu_times,
        start_fs=args.start_fs, pred_end_fs=args.pred_end_fs,
        nhist=nhist,
        dq_mean=dq_mean, dq_std=dq_std,
        mu_mean=mu_mean, mu_std=mu_std,
        mm_mean=mm_mean, mm_std=mm_std,
        device=device,
    )

    # 10) save dq compare (long)
    N, steps = q_true.shape
    rows = []
    for a in range(N):
        for j in range(steps):
            rows.append({
                "time_fs": float(t_q[j]),
                "atom": int(a),
                "q_true": float(q_true[a, j]),
                "q_pred": float(q_pred[a, j]),
                "q_err": float(q_pred[a, j] - q_true[a, j]),
            })
    df_q = pd.DataFrame(rows)
    q_csv = outdir / f"pred_q_{args.dir}_{int(args.start_fs)}to{int(args.pred_end_fs)}fs.csv"
    df_q.to_csv(q_csv, index=False)

    # save mu/m compare
    df_m = pd.DataFrame({
        "time_fs": t_m.astype(np.float32),
        "mu_true": mu_true.astype(np.float32),
        "mu_pred": mu_pred.astype(np.float32),
        "mu_err": (mu_pred - mu_true).astype(np.float32),
        "m_true": mm_true.astype(np.float32),
        "m_pred": mm_pred.astype(np.float32),
        "m_err": (mm_pred - mm_true).astype(np.float32),
    })
    m_csv = outdir / f"pred_mu_m_{args.dir}_{int(args.start_fs)}to{int(args.pred_end_fs)}fs.csv"
    df_m.to_csv(m_csv, index=False)

    # 11) metrics summary
    # dq metrics: per-atom MAE/RMSE, and global
    metrics = {"dq": {}, "mu": {}, "mm": {}}
    dq_mae_atoms = []
    dq_rmse_atoms = []
    for a in range(N):
        mt = _metrics_1d(q_true[a], q_pred[a])
        metrics["dq"][f"atom_{a}"] = mt
        dq_mae_atoms.append(mt["mae"])
        dq_rmse_atoms.append(mt["rmse"])
    metrics["dq"]["global"] = {
        "mae": float(np.mean(np.abs(q_pred - q_true))),
        "rmse": float(np.sqrt(np.mean((q_pred - q_true) ** 2))),
        "mae_mean_atoms": float(np.mean(dq_mae_atoms)),
        "rmse_mean_atoms": float(np.mean(dq_rmse_atoms)),
    }
    metrics["mu"] = _metrics_1d(mu_true, mu_pred)
    metrics["mm"] = _metrics_1d(mm_true, mm_pred)

    import json
    (outdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved: {q_csv}")
    print(f"[OK] Saved: {m_csv}")
    print(f"[OK] Saved: {outdir / 'metrics.json'}")
    print(f"[Metrics] mu:  {metrics['mu']}")
    print(f"[Metrics] mm:  {metrics['mm']}")
    print(f"[Metrics] dq(global): {metrics['dq']['global']}")

    # 12) plots
    if args.plot:
        import matplotlib.pyplot as plt

        # mu
        plt.figure()
        plt.plot(t_m, mu_true, label="mu true")
        plt.plot(t_m, mu_pred, label="mu pred")
        plt.xlabel("time (fs)")
        plt.ylabel(f"mu_{args.dir}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"mu_compare_{int(args.start_fs)}to{int(args.pred_end_fs)}fs.png", dpi=150)
        plt.close()

        # m
        plt.figure()
        plt.plot(t_m, mm_true, label="m true")
        plt.plot(t_m, mm_pred, label="m pred")
        plt.xlabel("time (fs)")
        plt.ylabel(f"m_{args.dir}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"m_compare_{int(args.start_fs)}to{int(args.pred_end_fs)}fs.png", dpi=150)
        plt.close()

        # dq per atom
        for a in range(N):
            plt.figure()
            plt.plot(t_q, q_true[a], label=f"q true a{a}")
            plt.plot(t_q, q_pred[a], label=f"q pred a{a}")
            plt.xlabel("time (fs)")
            plt.ylabel("dq")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"q_compare_a{a}_{int(args.start_fs)}to{int(args.pred_end_fs)}fs.png", dpi=150)
            plt.close()

        print("[OK] Plots saved.")


if __name__ == "__main__":
    main()
