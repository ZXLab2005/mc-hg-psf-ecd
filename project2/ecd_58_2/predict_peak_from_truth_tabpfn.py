#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_peak_from_truth_tabpfn.py (fusion v3 + CI)
--------------------------------------------------
peaks_truth.csv 峰监督 + TabPFN 预测 (peak-head)
并可选叠加 PCA-shape head (global shape) 融合。

示例：
python predict_peak_from_truth_tabpfn.py \
  --X_mace mace_emb.csv \
  --X_compact3d X_compact3D.cleaned.csv \
  --Y Y_all.csv \
  --peaks_truth peaks_truth.csv \
  --outdir out_fusion_peak_pca \
  --gauss_sigma 0.15 --ma_win 11 \
  --regressor tabpfn \
  --model_path "./out/tabpfn_cache/tabpfn-v2-regressor.ckpt" \
  --device cpu --cv 5 --repeats 2 --log-level INFO \
  --use_pca_fusion --pca_k 5 --alpha_pca 0.5 \
  --tabpfn_estimators 8 \
  --alpha_grid "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
  --save_ci_files --ci_quantiles "0.15,0.85" \
  --plot_n 9
"""

import argparse, json, time, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ===== TabPFN imports（保持你原逻辑）=====
_HAS_AUTO = False
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    _HAS_AUTO = True
except Exception:
    _HAS_AUTO = False

try:
    from tabpfn import TabPFNRegressor
except Exception as e:
    raise RuntimeError("请先安装 tabpfn：pip install tabpfn") from e


# -------------------------- utils --------------------------
def setup_logger(level="INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("ecd-peak-tabpfn")


def parse_energy_cols(cols):
    e_list = []
    for c in cols:
        if c == "id":
            continue
        s = c.replace("y_e", "")
        a, b = s.split("p")
        e_list.append(float(a) + float(b) / 1000.0)
    return np.array(e_list, dtype=float)


def moving_average(Y, win):
    if win <= 1:
        return Y
    k = np.ones(int(win), dtype=float) / int(win)
    Ys = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        Ys[i] = np.convolve(Y[i], k, mode="same")
    return Ys


def gaussian_kernel(sigma_bins, radius=4):
    if sigma_bins <= 1e-12:
        return np.array([1.0])
    half = int(max(1, round(radius * sigma_bins)))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    return k / k.sum()


def gaussian_broaden(Y, sigma_eV, energies):
    if sigma_eV <= 0:
        return Y
    dE = float(np.median(np.diff(energies)))
    sigma_bins = sigma_eV / dE
    k = gaussian_kernel(sigma_bins)
    Yb = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        Yb[i] = np.convolve(Y[i], k, mode="same")
    return Yb


def shape_normalize(Y):
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Yc = Yc / (np.abs(Yc).max(axis=1, keepdims=True) + 1e-12)
    return normalize(Yc, norm="l2", axis=1)


def align_global_sign(Y_shape, energies, emin=3.0, emax=6.0):
    m = (energies >= emin) & (energies <= emax)
    if m.sum() == 0:
        return Y_shape
    sgn = Y_shape[:, m].sum(axis=1)
    Y2 = Y_shape.copy()
    Y2[sgn < 0] *= -1.0
    return Y2


def spectrum_centroid(Y_shape, energies):
    w = np.abs(Y_shape) + 1e-12
    num = (w * energies[None, :]).sum(axis=1)
    den = w.sum(axis=1)
    return num / den


# ===== amp log 压缩/逆变换 =====
def amp_transform(aS, a0=0.02):
    return np.sign(aS) * np.log1p(np.abs(aS) / a0)


def amp_inverse(z, a0=0.02):
    return np.sign(z) * a0 * (np.expm1(np.abs(z)))


def render_spectrum_from_peaks(pos, amp_signed, energies, sigma_eV):
    N, K = pos.shape
    M = len(energies)
    Y = np.zeros((N, M), dtype=float)
    for i in range(N):
        y = np.zeros(M, dtype=float)
        for k in range(K):
            e0 = pos[i, k]
            aS = amp_signed[i, k]
            y += aS * np.exp(-0.5 * ((energies - e0) / sigma_eV) ** 2)
        Y[i] = y
    return shape_normalize(Y)


# ===== band-wise whitening for PCA head =====
def bandwise_whiten(Y_shape, energies, bands):
    Yw = np.zeros_like(Y_shape)
    mu = np.zeros((1, Y_shape.shape[1]))
    std = np.zeros((1, Y_shape.shape[1]))

    for (emin, emax) in bands:
        m = (energies >= emin) & (energies < emax)
        if m.sum() == 0:
            continue
        mu_b = Y_shape[:, m].mean(axis=0, keepdims=True)
        std_b = Y_shape[:, m].std(axis=0, keepdims=True) + 1e-12
        Yw[:, m] = (Y_shape[:, m] - mu_b) / std_b
        mu[:, m], std[:, m] = mu_b, std_b

    uncovered = (std == 0).flatten()
    if uncovered.any():
        mu_u = Y_shape[:, uncovered].mean(axis=0, keepdims=True)
        std_u = Y_shape[:, uncovered].std(axis=0, keepdims=True) + 1e-12
        Yw[:, uncovered] = (Y_shape[:, uncovered] - mu_u) / std_u
        mu[:, uncovered], std[:, uncovered] = mu_u, std_u

    return Yw, mu, std


def dewhiten(Yw, mu, std):
    return Yw * std + mu


def load_peak_truth(peaks_truth_csv):
    df = pd.read_csv(peaks_truth_csv)
    if "id" not in df.columns:
        raise ValueError("peaks_truth.csv 必须有 id 列")

    pos_cols = sorted([c for c in df.columns if c.endswith("_pos")])
    amp_cols = sorted([c for c in df.columns if c.endswith("_amp_signed")])

    if not amp_cols:
        amp_cols_abs = sorted([c for c in df.columns if c.endswith("_amp")])
        sgn_cols = sorted([c for c in df.columns if c.endswith("_sign")])
        if not amp_cols_abs or not sgn_cols:
            raise ValueError("peaks_truth.csv 至少需要 pos + (amp_signed 或 amp+sign)")
        amp_signed = df[amp_cols_abs].to_numpy(float) * df[sgn_cols].to_numpy(int)
        amp_cols = [c.replace("_amp", "_amp_signed") for c in amp_cols_abs]
        for j, c in enumerate(amp_cols):
            df[c] = amp_signed[:, j]

    if not pos_cols or not amp_cols:
        raise ValueError("peaks_truth.csv 没找到 p*_pos / p*_amp_signed 列")

    K = len(pos_cols)
    if len(amp_cols) != K:
        raise ValueError("pos 与 amp_signed 列数不一致")

    ids = df["id"].to_numpy()
    pos = df[pos_cols].to_numpy(float)
    amp_signed = df[amp_cols].to_numpy(float)

    return ids, pos, amp_signed, K, (pos_cols, amp_cols)


# -------- TabPFN helpers --------
def tabpfn_regressor_factory(args):
    if args.regressor == "auto_tabpfn":
        if not _HAS_AUTO:
            raise RuntimeError("AutoTabPFNRegressor 未安装：pip install tabpfn-extensions")
        return AutoTabPFNRegressor(
            device=args.device,
            model_path=args.model_path,
            n_estimators=args.tabpfn_estimators,
        )
    else:
        return TabPFNRegressor(
            device=args.device,
            model_path=args.model_path,
            n_estimators=args.tabpfn_estimators,
        )


def fit_predict_tabpfn_multi_output(Xtr, Ytr, Xva, args, quantiles=None):
    """
    如果 quantiles=None：只返回 preds (mean)
    如果 quantiles 不为 None：返回 (preds, q_low, q_high)
    其中 q_low / q_high 是基于 TabPFNRegressor.predict(output_type="quantiles") 的分位数。
    """
    n_targets = Ytr.shape[1]
    preds = np.zeros((Xva.shape[0], n_targets), dtype=float)

    if quantiles is not None:
        q_low = np.zeros_like(preds)
        q_high = np.zeros_like(preds)
    else:
        q_low = q_high = None

    for j in range(n_targets):
        reg = tabpfn_regressor_factory(args)
        reg.fit(Xtr, Ytr[:, j])

        # 点预测：mean（保持原行为）
        preds[:, j] = reg.predict(Xva)

        # 置信区间
        if quantiles is not None:
            q_list = reg.predict(Xva, output_type="quantiles", quantiles=quantiles)
            # q_list 是 list，每个元素 shape=(n_va,)
            q_low[:, j] = q_list[0]
            q_high[:, j] = q_list[-1]

    if quantiles is None:
        return preds
    else:
        return preds, q_low, q_high


# ===== v4: simple Nature-style plots =====
def plot_mean_curve(energies, Y_true, Y_pred, out_png):
    import matplotlib.pyplot as plt
    mt = Y_true.mean(axis=0)
    mp = Y_pred.mean(axis=0)
    fig = plt.figure(figsize=(6.5, 4.0))
    plt.plot(energies, mt, lw=2.5, label="Mean truth")
    plt.plot(energies, mp, lw=2.5, label="Mean pred")
    plt.xlabel("E (eV)")
    plt.ylabel("R (a.u.)")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_random_samples(energies, Y_true, Y_pred, ids, out_png, n=9, seed=0):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    n = min(n, Y_true.shape[0])
    pick = rng.choice(Y_true.shape[0], size=n, replace=False)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)

    for ax, i in zip(axes.ravel(), pick):
        ax.plot(energies, Y_true[i], lw=2.0, label="TDDFT truth")
        ax.plot(energies, Y_pred[i], lw=2.0, label="Pred")
        ax.set_title(f"id={ids[i]}", fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_xlabel("E (eV)")
        ax.set_ylabel("R (a.u.)")

    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# -------------------------- main --------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--X_mace", required=True)
    ap.add_argument("--Y", required=True)
    ap.add_argument("--peaks_truth", required=True)
    ap.add_argument("--outdir", default="./out_peak_truth_tabpfn")

    ap.add_argument("--mace_dim", type=int, default=16)
    ap.add_argument("--gauss_sigma", type=float, default=0.15)
    ap.add_argument("--ma_win", type=int, default=11)

    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_jobs", type=int, default=1)

    # ===== TabPFN 参数（位置/名字与原 full_spectrum_tabpfn 一样）=====
    ap.add_argument("--regressor", choices=["tabpfn", "auto_tabpfn"], default="tabpfn")
    ap.add_argument("--model_path", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu")

    # ===== v4 最小新增（不影响旧命令）=====
    ap.add_argument(
        "--tabpfn_estimators",
        type=int,
        default=8,
        help="TabPFN ensemble 数 (ensembling)，默认8",
    )
    ap.add_argument(
        "--alpha_grid",
        type=str,
        default=None,
        help="alpha 网格搜索列表，如 '0,0.1,...,1'；不设则默认0~1步长0.1",
    )
    ap.add_argument(
        "--plot_n",
        type=int,
        default=9,
        help="随机画 n 条 OOF 对比谱；0则不画",
    )

    ap.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    ap.add_argument("--progress", action="store_true")

    # ===== 新增：compact3d + PCA-fusion =====
    ap.add_argument("--X_compact3d", default=None, help="可选：X_compact3D.cleaned.csv (id+feat)")
    ap.add_argument("--use_pca_fusion", action="store_true", help="启用 PCA-shape head 融合")
    ap.add_argument("--pca_k", type=int, default=5)
    ap.add_argument("--bands", default="2.0-3.5,3.5-5.0,5.0-7.0")
    ap.add_argument(
        "--alpha_pca",
        type=float,
        default=0.5,
        help="PCA分支权重 α (0=仅Peak, 1=仅PCA)",
    )
    ap.add_argument(
        "--ridge_alpha",
        type=float,
        default=5.0,
        help="PCA分支 Ridge 回归正则",
    )

    # ===== 新增：TabPFN 置信区间 =====
    ap.add_argument(
        "--ci_quantiles",
        type=str,
        default=None,
        help="例如 '0.15,0.85'；指定则对峰参数输出 TabPFN 置信区间",
    )
    ap.add_argument(
        "--save_ci_files",
        action="store_true",
        help="保存带置信区间的峰参数预测（写入 pred_oof_peaks.csv）",
    )

    args = ap.parse_args()
    logger = setup_logger(args.log_level)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("[START] fusion v3: Peak(TabPFN) + optional PCA-head + optional compact3d")
    logger.info("Args: %s", vars(args))
    t0 = time.time()

    # ==== 解析置信区间分位数 ====
    if args.ci_quantiles is not None:
        ci_qs = [float(x) for x in args.ci_quantiles.split(",")]
        if len(ci_qs) < 2:
            raise ValueError("ci_quantiles 至少要给两个值，例如 '0.15,0.85' 或 '0.1,0.5,0.9'")
    else:
        ci_qs = None

    Xm = pd.read_csv(args.X_mace)
    Ydf = pd.read_csv(args.Y)
    ids_p, pos_truth, ampS_truth, K, col_groups = load_peak_truth(args.peaks_truth)

    if args.X_compact3d:
        Xc = pd.read_csv(args.X_compact3d)
        common = sorted(set(Xm["id"]) & set(Ydf["id"]) & set(ids_p) & set(Xc["id"]))
        Xc = Xc[Xc["id"].isin(common)].sort_values("id").reset_index(drop=True)
        X_compact_raw = Xc.drop(columns=["id"]).to_numpy(float)
    else:
        common = sorted(set(Xm["id"]) & set(Ydf["id"]) & set(ids_p))
        X_compact_raw = None

    Xm = Xm[Xm["id"].isin(common)].sort_values("id").reset_index(drop=True)
    Ydf = Ydf[Ydf["id"].isin(common)].sort_values("id").reset_index(drop=True)
    p_df = pd.read_csv(args.peaks_truth)
    p_df = p_df[p_df["id"].isin(common)].sort_values("id").reset_index(drop=True)
    pos_truth = p_df[col_groups[0]].to_numpy(float)
    ampS_truth = p_df[col_groups[1]].to_numpy(float)

    ids = Xm["id"].to_numpy()
    X_mace_raw = Xm.drop(columns=["id"]).to_numpy(float)

    spec_cols = [c for c in Ydf.columns if c != "id"]
    energies = parse_energy_cols(spec_cols)
    Y = Ydf[spec_cols].to_numpy(float)

    emin_all, emax_all = float(energies.min()), float(energies.max())
    logger.info("[PEAKS] K_truth=%d", K)
    logger.info("[DATA] X_mace_raw=%s Y=%s", X_mace_raw.shape, Y.shape)

    # ===== X-PCA on MACE =====
    pca_m = PCA(
        n_components=min(args.mace_dim, X_mace_raw.shape[0] - 1),
        random_state=args.seed,
    )
    X_mace = pca_m.fit_transform(X_mace_raw)

    # ===== optional concat compact3d =====
    if X_compact_raw is not None:
        sc_c = StandardScaler()
        X_compact = sc_c.fit_transform(X_compact_raw)
        X_main = np.concatenate([X_mace, X_compact], axis=1)
        logger.info("[X-MAIN] concat MACE+compact3d -> %s", X_main.shape)
    else:
        X_main = X_mace
        logger.info("[X-MAIN] only MACE -> %s", X_main.shape)

    # ===== eval space =====
    Y_s = moving_average(Y, args.ma_win)
    Y_b = gaussian_broaden(Y_s, args.gauss_sigma, energies)
    Y_shape = shape_normalize(Y_b)
    Y_shape = align_global_sign(Y_shape, energies, 3.0, 6.0)

    centroids = spectrum_centroid(Y_shape, energies)

    # ===== PCA head precompute =====
    if args.use_pca_fusion:
        bands = []
        for seg in args.bands.split(","):
            a, b = seg.split("-")
            bands.append((float(a), float(b)))

        Yw, mu_e, std_e = bandwise_whiten(Y_shape, energies, bands)
        pca_y = PCA(
            n_components=int(args.pca_k),
            svd_solver="full",
            random_state=args.seed,
        )
        Z = pca_y.fit_transform(Yw)
        logger.info("[PCA-HEAD] enabled: k=%d, bands=%s", Z.shape[1], bands)
    else:
        Z = None
        pca_y = mu_e = std_e = None

    scaler_x = StandardScaler()

    pos_oof = np.zeros_like(pos_truth)
    ampS_oof = np.zeros_like(ampS_truth)
    if args.use_pca_fusion:
        Z_oof = np.zeros_like(Z)

    # ==== 置信区间 OOF 累加器 ====
    if args.save_ci_files and ci_qs is not None:
        pos_q_low_oof = np.zeros_like(pos_truth)
        pos_q_high_oof = np.zeros_like(pos_truth)
        amp_q_low_oof = np.zeros_like(ampS_truth)
        amp_q_high_oof = np.zeros_like(ampS_truth)
    else:
        pos_q_low_oof = pos_q_high_oof = None
        amp_q_low_oof = amp_q_high_oof = None

    r2_list, cos_list, mae_list = [], [], []
    alpha_best_list = []  # 记录每次 repeat 的最佳 alpha

    for rep in range(args.repeats):
        rep_seed = args.seed + rep * 11
        kf = KFold(n_splits=args.cv, shuffle=True, random_state=rep_seed)
        logger.info("[Repeat %d/%d] seed=%d", rep + 1, args.repeats, rep_seed)

        pos_rep = np.zeros_like(pos_truth)
        ampS_rep = np.zeros_like(ampS_truth)
        if args.use_pca_fusion:
            Z_rep = np.zeros_like(Z)

        if args.save_ci_files and ci_qs is not None:
            pos_q_low_rep = np.zeros_like(pos_truth)
            pos_q_high_rep = np.zeros_like(pos_truth)
            amp_q_low_rep = np.zeros_like(ampS_truth)
            amp_q_high_rep = np.zeros_like(ampS_truth)
        else:
            pos_q_low_rep = pos_q_high_rep = None
            amp_q_low_rep = amp_q_high_rep = None

        for fold, (tr, va) in enumerate(kf.split(X_main), start=1):
            Xtr = scaler_x.fit_transform(X_main[tr])
            Xva = scaler_x.transform(X_main[va])

            # ===== Peak-head targets =====
            pos_off_tr = pos_truth[tr] - centroids[tr, None]
            amp_log_tr = amp_transform(ampS_truth[tr])

            sc_pos = StandardScaler()
            sc_amp = StandardScaler()
            pos_tr_s = sc_pos.fit_transform(pos_off_tr)
            amp_tr_s = sc_amp.fit_transform(amp_log_tr)

            # 位置：带/不带 CI
            if args.save_ci_files and ci_qs is not None:
                pos_va_s, pos_q_low_s, pos_q_high_s = fit_predict_tabpfn_multi_output(
                    Xtr, pos_tr_s, Xva, args, quantiles=ci_qs
                )
            else:
                pos_va_s = fit_predict_tabpfn_multi_output(Xtr, pos_tr_s, Xva, args)

            # 幅度：带/不带 CI
            if args.save_ci_files and ci_qs is not None:
                amp_va_s, amp_q_low_s, amp_q_high_s = fit_predict_tabpfn_multi_output(
                    Xtr, amp_tr_s, Xva, args, quantiles=ci_qs
                )
            else:
                amp_va_s = fit_predict_tabpfn_multi_output(Xtr, amp_tr_s, Xva, args)

            pos_off_va = sc_pos.inverse_transform(pos_va_s)
            amp_log_va = sc_amp.inverse_transform(amp_va_s)

            pos_va = centroids[va, None] + pos_off_va
            ampS_va = amp_inverse(amp_log_va)

            pos_va = np.clip(pos_va, emin_all, emax_all)

            pos_rep[va] = pos_va
            ampS_rep[va] = ampS_va

            if args.save_ci_files and ci_qs is not None:
                # 位置 CI
                pos_q_low_off_va = sc_pos.inverse_transform(pos_q_low_s)
                pos_q_high_off_va = sc_pos.inverse_transform(pos_q_high_s)
                pos_q_low_va = centroids[va, None] + pos_q_low_off_va
                pos_q_high_va = centroids[va, None] + pos_q_high_off_va

                # 幅度 CI
                amp_q_low_log_va = sc_amp.inverse_transform(amp_q_low_s)
                amp_q_high_log_va = sc_amp.inverse_transform(amp_q_high_s)
                amp_q_low_va = amp_inverse(amp_q_low_log_va)
                amp_q_high_va = amp_inverse(amp_q_high_log_va)

                pos_q_low_rep[va] = pos_q_low_va
                pos_q_high_rep[va] = pos_q_high_va
                amp_q_low_rep[va] = amp_q_low_va
                amp_q_high_rep[va] = amp_q_high_va

            # ===== PCA-head ridge regression (global shape) =====
            if args.use_pca_fusion:
                sc_z = StandardScaler()
                Ztr_s = sc_z.fit_transform(Z[tr])

                ridge = MultiOutputRegressor(Ridge(alpha=args.ridge_alpha), n_jobs=args.n_jobs)
                ridge.fit(Xtr, Ztr_s)
                Zva_s = ridge.predict(Xva)
                Zva = sc_z.inverse_transform(Zva_s)

                Z_rep[va] = Zva

        # ----- reconstruct Peak spectrum -----
        Y_peak_rep = render_spectrum_from_peaks(
            pos_rep, ampS_rep, energies, sigma_eV=args.gauss_sigma
        )
        Y_peak_rep = align_global_sign(Y_peak_rep, energies, 3.0, 6.0)

        if args.save_ci_files and ci_qs is not None:
            pos_q_low_oof  # 只是确保变量存在，逻辑上已在上面处理

        if args.use_pca_fusion:
            # ----- reconstruct PCA spectrum -----
            Yw_rep = pca_y.inverse_transform(Z_rep)
            Y_pca_rep = dewhiten(Yw_rep, mu_e, std_e)
            Y_pca_rep = normalize(Y_pca_rep, norm="l2", axis=1)
            Y_pca_rep = align_global_sign(Y_pca_rep, energies, 3.0, 6.0)

            # alpha 网格搜索（若未提供则退化为 args.alpha_pca）
            if args.alpha_grid is None:
                alpha_grid = np.linspace(0.0, 1.0, 11)
            else:
                alpha_grid = np.array(
                    [float(x) for x in args.alpha_grid.split(",")],
                    dtype=float,
                )

            cos_by_alpha = []
            for a in alpha_grid:
                Y_fuse_a = normalize(
                    a * Y_pca_rep + (1 - a) * Y_peak_rep,
                    norm="l2",
                    axis=1,
                )
                Y_fuse_a = align_global_sign(Y_fuse_a, energies, 3.0, 6.0)
                cos_a = float(np.mean(np.sum(Y_shape * Y_fuse_a, axis=1)))
                cos_by_alpha.append(cos_a)

            best_i = int(np.argmax(cos_by_alpha))
            alpha = float(alpha_grid[best_i])
            alpha_best_list.append(alpha)
            logger.info(
                "  [AlphaGrid] best alpha=%.3f, cos=%.4f",
                alpha,
                cos_by_alpha[best_i],
            )
            Y_final_rep = normalize(
                alpha * Y_pca_rep + (1 - alpha) * Y_peak_rep,
                norm="l2",
                axis=1,
            )
        else:
            Y_final_rep = Y_peak_rep

        r2_rep = float(r2_score(Y_shape, Y_final_rep, multioutput="variance_weighted"))
        cos_rep = float(np.mean(np.sum(Y_shape * Y_final_rep, axis=1)))
        mae_rep = float(np.mean(np.abs(Y_shape - Y_final_rep)))

        r2_list.append(r2_rep)
        cos_list.append(cos_rep)
        mae_list.append(mae_rep)
        pos_oof += pos_rep
        ampS_oof += ampS_rep
        if args.use_pca_fusion:
            Z_oof += Z_rep

        if args.save_ci_files and ci_qs is not None:
            pos_q_low_oof += pos_q_low_rep
            pos_q_high_oof += pos_q_high_rep
            amp_q_low_oof += amp_q_low_rep
            amp_q_high_oof += amp_q_high_rep

        logger.info("  metrics: R2=%.3f cos=%.3f MAE=%.4f", r2_rep, cos_rep, mae_rep)

    pos_oof /= args.repeats
    ampS_oof /= args.repeats

    if args.save_ci_files and ci_qs is not None:
        pos_q_low_oof /= args.repeats
        pos_q_high_oof /= args.repeats
        amp_q_low_oof /= args.repeats
        amp_q_high_oof /= args.repeats

    Y_peak_oof = render_spectrum_from_peaks(
        pos_oof, ampS_oof, energies, sigma_eV=args.gauss_sigma
    )
    Y_peak_oof = align_global_sign(Y_peak_oof, energies, 3.0, 6.0)

    if args.use_pca_fusion:
        Z_oof /= args.repeats
        Yw_oof = pca_y.inverse_transform(Z_oof)
        Y_pca_oof = dewhiten(Yw_oof, mu_e, std_e)
        Y_pca_oof = normalize(Y_pca_oof, norm="l2", axis=1)
        Y_pca_oof = align_global_sign(Y_pca_oof, energies, 3.0, 6.0)

        alpha = float(np.mean(alpha_best_list)) if len(alpha_best_list) > 0 else float(
            args.alpha_pca
        )
        logger.info("[OOF] use alpha=%.3f", alpha)
        Yshape_oof = normalize(
            alpha * Y_pca_oof + (1 - alpha) * Y_peak_oof,
            norm="l2",
            axis=1,
        )
    else:
        Yshape_oof = Y_peak_oof

    # 最终预测谱做一次全局符号对齐
    Yshape_oof = align_global_sign(Yshape_oof, energies, 3.0, 6.0)

    # save peaks (+ CI)
    pos_cols, ampS_cols = col_groups
    out_peaks = pd.DataFrame({"id": ids})

    for j, c in enumerate(pos_cols):
        out_peaks[c] = pos_oof[:, j]
        if pos_q_low_oof is not None:
            base = c.replace("_pos", "")
            out_peaks[base + "_pos_q_low"] = pos_q_low_oof[:, j]
            out_peaks[base + "_pos_q_high"] = pos_q_high_oof[:, j]

    for j, c in enumerate(ampS_cols):
        out_peaks[c] = ampS_oof[:, j]
        out_peaks[c.replace("_amp_signed", "_amp")] = np.abs(ampS_oof[:, j])
        out_peaks[c.replace("_amp_signed", "_sign")] = np.where(ampS_oof[:, j] >= 0, 1, -1)

        if amp_q_low_oof is not None:
            base = c.replace("_amp_signed", "")
            out_peaks[base + "_amp_signed_q_low"] = amp_q_low_oof[:, j]
            out_peaks[base + "_amp_signed_q_high"] = amp_q_high_oof[:, j]

    out_peaks.to_csv(outdir / "pred_oof_peaks.csv", index=False)
    pd.DataFrame(np.c_[ids, Yshape_oof], columns=["id"] + spec_cols).to_csv(
        outdir / "pred_oof_shape.csv", index=False
    )

    # ==== 计算 R2/cos/MAE 的 95% CI ====
    def ci95(x_list):
        x = np.array(x_list, dtype=float)
        m = float(x.mean())
        if len(x) <= 1:
            return {
                "mean": m,
                "low": m,
                "high": m,
            }
        se = float(x.std(ddof=1) / np.sqrt(len(x)))
        low = m - 1.96 * se
        high = m + 1.96 * se
        return {"mean": m, "low": float(low), "high": float(high)}

    r2_ci = ci95(r2_list)
    cos_ci = ci95(cos_list)
    mae_ci = ci95(mae_list)

    meta = {
        "oof_avg": {
            "R2": r2_ci["mean"],
            "cosine": cos_ci["mean"],
            "MAE": mae_ci["mean"],
        },
        "oof_ci95": {
            "R2": {"low": r2_ci["low"], "high": r2_ci["high"]},
            "cosine": {"low": cos_ci["low"], "high": cos_ci["high"]},
            "MAE": {"low": mae_ci["low"], "high": mae_ci["high"]},
        },
        "repeat_each": [
            {
                "R2": float(r2_list[i]),
                "cosine": float(cos_list[i]),
                "MAE": float(mae_list[i]),
            }
            for i in range(len(r2_list))
        ],
        "K_truth": int(K),
        "mace_dim": int(X_mace.shape[1]),
        "use_compact3d": bool(args.X_compact3d),
        "use_pca_fusion": bool(args.use_pca_fusion),
        "alpha_pca": float(args.alpha_pca),
        "alpha_best_repeats": alpha_best_list,
        "pca_k": int(args.pca_k),
        "notes": "fusion v3: Peak(TabPFN pos_offset+log_amp) + optional PCA-head(Ridge on Z) + optional compact3d, fused in shape space. CI 来自 TabPFN quantiles.",
    }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ===== 在 main 内部画图 =====
    try:
        plot_mean_curve(energies, Y_shape, Yshape_oof, outdir / "oof_mean_compare.png")
        if args.plot_n and args.plot_n > 0:
            plot_random_samples(
                energies,
                Y_shape,
                Yshape_oof,
                ids,
                outdir / "oof_samples_compare.png",
                n=args.plot_n,
                seed=args.seed,
            )
    except Exception as e:
        logger.warning(f"Plotting failed: {e}")

    dt = time.time() - t0
    logger.info("[DONE] outputs -> %s", outdir)
    logger.info("[OOF avg] %s", meta["oof_avg"])
    logger.info("[OOF ci95] %s", meta["oof_ci95"])
    logger.info("Total wall time: %.1fs", dt)


if __name__ == "__main__":
    main()
