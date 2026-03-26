#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECD spectrum prediction — TabPFN pipeline with options to learn only shape (no amplitude learning)
plus interpretability add-ons (append-only based on your v27) and UMAP bins-by-wavelength coloring.

python predict_full_spectrum_tabpfn.py \
  --X ./out/X_compact3D.cleaned.csv \
  --Y ./out_ecd/Y_all.csv \
  --out_pred ./out/preds.csv \
  --out_metrics ./out/metrics.csv \
  --method pca \
  --pca_x_var 0.95 --pca_x_max 32 \
  --pca_var 0.99 --pca_max 16 \
  --amp_norm l2 --target_tf yeojohnson \
  --regressor tabpfn --pc_models tabpfn,ridge --pc_ridge_alpha 10 \
  --model_path "./out/tabpfn_cache/tabpfn-v2-regressor.ckpt" \
  --explain_perm --perm_repeats 5 --perm_topk 20 --explain_out ./out/perm_importance.png \
  --umap_scatter --umap_color bins --umap_bins_mode nm_auto --umap_out ./out/umap_bins_auto.png \
  --device cpu --cv 6 --log-level INFO
"""
import argparse, math, warnings, logging
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge, HuberRegressor, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

_HAS_AUTO = False
try:
    from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
    _HAS_AUTO = True
except Exception:
    _HAS_AUTO = False

# --- TabPFN ---
try:
    from tabpfn import TabPFNRegressor
except Exception as e:
    raise RuntimeError("请先安装 tabpfn：pip install tabpfn") from e


# ---------------- utils ----------------
def setup_logger(level="INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    return logging.getLogger("ecd-tabpfn-ens-shape")

def read_xy(x_path, y_path):
    X = pd.read_csv(x_path); Y = pd.read_csv(y_path)
    ids = sorted(set(X["id"]).intersection(set(Y["id"])) )
    X = X[X["id"].isin(ids)].sort_values("id").reset_index(drop=True)
    Y = Y[Y["id"].isin(ids)].sort_values("id").reset_index(drop=True)
    spec_cols = [c for c in Y.columns if c != "id"]
    return X, Y, np.array(ids), spec_cols

def e_axis_from_cols(spec_cols):
    def col_to_eV(c): return float(c.replace("y_e","" ).replace("p","."))
    return np.array([col_to_eV(c) for c in spec_cols], dtype=float)

def impute_X_numeric_median(X_df):
    Xn = X_df.drop(columns=["id"]).copy()
    for c in Xn.columns:
        if not np.issubdtype(Xn[c].dtype, np.number):
            Xn[c] = pd.Categorical(Xn[c]).codes
    med = Xn.median(numeric_only=True); Xn = Xn.fillna(med)
    return Xn.to_numpy(dtype=np.float32), med

def smooth_1d(Y, win=0, passes=1):
    if win is None or win <= 1 or passes <= 0: return Y
    k = int(win); k = max(1, k | 1)
    ker = np.ones(k, dtype=np.float32) / k
    Y2 = Y.copy()
    for _ in range(int(passes)):
        Y2 = np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), 1, Y2)
    return Y2.astype(np.float32)


# -------------- PCA X/Y --------------
def pca_fit_x_auto(X, pca_x_var=None, pca_x_max=None, seed=42):
    if (pca_x_var is None) and (pca_x_max is None):
        return None, X, None, None
    max_allowed = min(X.shape[0], X.shape[1])
    ncomp_init = max_allowed if pca_x_max is None else min(int(pca_x_max), max_allowed)
    pca_full = PCA(n_components=ncomp_init, random_state=seed).fit(X)
    evr = np.cumsum(pca_full.explained_variance_ratio_)
    K = ncomp_init
    if pca_x_var is not None:
        K = int(np.searchsorted(evr, float(pca_x_var)) + 1); K = min(K, ncomp_init)
    pca = PCA(n_components=K, random_state=seed).fit(X)
    return pca, pca.transform(X), K, float(np.sum(pca.explained_variance_ratio_))

class TargetTransformer:
    def __init__(self, method="yeojohnson"):
        self.method = method; self.tx = None
    def fit(self, Z):
        if self.method == "none": return self
        if self.method == "yeojohnson":
            self.tx = PowerTransformer(method="yeo-johnson", standardize=True)
        elif self.method == "quantile":
            n_q = max(32, min(100, Z.shape[0]))
            self.tx = QuantileTransformer(output_distribution="normal", n_quantiles=n_q, subsample=int(1e9), random_state=42)
        else:
            raise ValueError("target_tf must be: none|yeojohnson|quantile")
        self.tx.fit(Z); return self
    def transform(self, Z): return Z if self.method=="none" else self.tx.transform(Z)
    def inverse_transform(self, Zt): return Zt if self.method=="none" else self.tx.inverse_transform(Zt)

def pca_fit_y_auto(Y_shape, pca_var=0.99, pca_max=16, seed=42):
    max_allowed = min(Y_shape.shape[0], Y_shape.shape[1])
    pca_full = PCA(n_components=min(int(pca_max), max_allowed), random_state=seed).fit(Y_shape)
    evr = np.cumsum(pca_full.explained_variance_ratio_)
    K = int(np.searchsorted(evr, float(pca_var)) + 1); K = min(K, pca_full.n_components_)
    pca = PCA(n_components=K, random_state=seed).fit(Y_shape)
    return pca, pca.transform(Y_shape), K, float(np.sum(pca.explained_variance_ratio_))

def amplitude_normalize(Y, mode="l2", eps=1e-8):
    if mode == "none":
        amp = np.ones((Y.shape[0], 1), dtype=np.float32); return Y.astype(np.float32), amp
    if mode == "maxabs":
        amp = np.max(np.abs(Y), axis=1, keepdims=True) + eps
    elif mode == "l2":
        amp = (np.linalg.norm(Y, ord=2, axis=1, keepdims=True) + eps)
    else:
        raise ValueError("amp_norm must be one of: none|maxabs|l2")
    return (Y/amp).astype(np.float32), amp.astype(np.float32)


# -------- metrics --------
def metric_table(y_true, y_pred):
    r2 = r2_score(y_true.ravel(), y_pred.ravel())
    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    rmse = math.sqrt(mean_squared_error(y_true.ravel(), y_pred.ravel()))
    return dict(global_R2=r2, global_MAE=mae, global_RMSE=rmse)

def shape_metrics_table(y_true, y_pred, sign_invariant=True):
    """
    shape-only 评估（每条谱 L2 归一化）
    - cos_raw：原始点积（符号敏感）
    - cos_signinv：|cos_raw|（符号不敏感，真正的 shape 相似度）
    """
    y_true_s, _ = amplitude_normalize(y_true, mode="l2")
    y_pred_s, _ = amplitude_normalize(y_pred, mode="l2")
    r2s = r2_score(y_true_s.ravel(), y_pred_s.ravel())

    cos_raw = np.sum(y_true_s * y_pred_s, axis=1)
    cos_use = np.abs(cos_raw) if sign_invariant else cos_raw

    return dict(
        shape_R2=float(r2s),
        shape_cos_mean=float(cos_use.mean()),
        shape_cos_p05=float(np.quantile(cos_use, 0.05)),
        shape_cos_p50=float(np.quantile(cos_use, 0.50)),
        shape_cos_p95=float(np.quantile(cos_use, 0.95)),
        shape_cos_raw_mean=float(cos_raw.mean()),
        shape_cos_raw_p05=float(np.quantile(cos_raw, 0.05)),
        shape_cos_raw_p50=float(np.quantile(cos_raw, 0.50)),
        shape_cos_raw_p95=float(np.quantile(cos_raw, 0.95)),
    )

# -------- amp helpers --------
def clip_logA_by_quantile(logA_pred, logA_ref, q_low=0.05, q_high=0.95):
    logA_pred = np.asarray(logA_pred, dtype=np.float32); ref = np.asarray(logA_ref, dtype=np.float32)
    lo = float(np.quantile(ref, q_low)); hi = float(np.quantile(ref, q_high)); med = float(np.median(ref))
    logA_pred = np.nan_to_num(logA_pred, nan=med, posinf=hi, neginf=lo)
    return np.clip(logA_pred, lo, hi), (lo, hi)

class AmpCalibrator:
    def __init__(self): self.lr = LinearRegression(); self.fitted = False
    def fit(self, logA_pred_tr, logA_true_tr):
        x = np.asarray(logA_pred_tr).reshape(-1,1); y = np.asarray(logA_true_tr).reshape(-1)
        self.lr.fit(x, y); self.fitted = True; return self
    def predict(self, logA_pred):
        if not self.fitted: return np.asarray(logA_pred).reshape(-1)
        x = np.asarray(logA_pred).reshape(-1,1); return self.lr.predict(x).reshape(-1)

# -------- base estimators (PC & AMP) --------
def make_base_estimator(kind="tabpfn", model_path=None, device="cpu"):
    if kind == "auto_tabpfn" and _HAS_AUTO:
        return AutoTabPFNRegressor()
    if model_path:
        return TabPFNRegressor(device=device, model_path=model_path)
    return TabPFNRegressor(device=device)

def make_amp_estimator(kind="ridge", model_path=None, device="cpu"):
    if kind == "ridge": return Ridge(alpha=1.0)
    if kind == "huber": return HuberRegressor(alpha=0.0, epsilon=1.35)
    if kind == "tabpfn": return make_base_estimator(kind="tabpfn", model_path=model_path, device=device)
    raise ValueError("amp_model must be one of: ridge|huber|tabpfn")

def build_pc_models(regressor, pc_ridge_alpha, model_path, device, seed):
    models = {}
    base = make_base_estimator(kind=regressor, model_path=model_path, device=device)
    models["tabpfn"] = MultiOutputRegressor(base, n_jobs=-1)
    models["ridge"] = Ridge(alpha=float(pc_ridge_alpha))
    gbdt = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=seed)
    models["gbdt"] = MultiOutputRegressor(gbdt, n_jobs=-1)
    models["mlp"] = MLPRegressor(hidden_layer_sizes=(128,), activation="relu", alpha=1e-2, learning_rate_init=1e-3,
                                   max_iter=800, early_stopping=True, random_state=seed)
    return models

def parse_pc_blend(pc_models, pc_blend):
    if pc_blend is None:
        if pc_models == ["tabpfn","ridge"]: return {"tabpfn":0.5,"ridge":0.5}
        w = 1.0/len(pc_models); return {m:w for m in pc_models}
    out = {}
    for seg in pc_blend.split(","):
        name, w = seg.split(":"); out[name.strip()] = float(w)
    s = sum(out.values())
    if s <= 0: raise ValueError("pc_blend 权重和必须>0")
    for k in out: out[k] /= s
    return {k:v for k,v in out.items() if k in pc_models}

# -------------- Y whiten --------------
class PerEnergyWhiten:
    def __init__(self): self.mu=None; self.std=None
    def fit(self, Y_shape):
        self.mu = np.mean(Y_shape, axis=0, keepdims=True).astype(np.float32)
        self.std = np.std(Y_shape, axis=0, keepdims=True).astype(np.float32) + 1e-8
        return self
    def transform(self, Y_shape): return ((Y_shape - self.mu) / self.std).astype(np.float32)
    def inverse_transform(self, Yw): return (Yw * self.std + self.mu).astype(np.float32)


# ---------------- main training (in-sample) ----------------
def train_predict_pca_two_stage(
    X_df, Y_df, spec_cols,
    amp_norm="l2",
    pca_x_var=None, pca_x_max=None,
    pca_var=0.99, pca_max=16, target_tf="yeojohnson",
    regressor="tabpfn", amp_model="ridge",
    pc_models=("tabpfn","ridge"), pc_blend=None,
    pc_ridge_alpha=10.0,
    amp_strategy="learn", amp_clip=(0.05,0.95), amp_calib=True,
    y_whiten="none",
    smooth_win=0, smooth_passes=1,
    model_path=None, device="cpu", seed=42,
    use_tqdm=False, logger=None,
):
    if logger is None: logger = logging.getLogger("ecd-tabpfn-ens-shape")
    logger.info("[STEP] X-PCA + AMP归一 + (可选)Y白化 + Y-PCA + PC多模型 + AMP策略(%s) + RENORM + (可选)平滑", amp_strategy)

    # X
    X_raw, _ = impute_X_numeric_median(X_df)
    scaler_x = StandardScaler().fit(X_raw)
    Xs = scaler_x.transform(X_raw)
    pca_x, X, Kx, evr_x = pca_fit_x_auto(Xs, pca_x_var=pca_x_var, pca_x_max=pca_x_max, seed=seed)
    if pca_x is not None: logger.info(f"[X-PCA] 选取主成分 Kx={Kx} | EVR={evr_x:.4f}")
    else: logger.info("[X-PCA] 未启用")

    # Y -> shape + amp
    Y = Y_df[spec_cols].to_numpy(dtype=np.float32)
    Y_shape, A = amplitude_normalize(Y, mode=amp_norm)
    logger.info(f"[AMP] 归一化方式: {amp_norm}")

    ywhite = None
    if y_whiten == "per_energy":
        ywhite = PerEnergyWhiten().fit(Y_shape)
        Y_shape_for_pca = ywhite.transform(Y_shape)
        logger.info("[Y-WHITEN] per-energy 标准化已启用")
    else:
        Y_shape_for_pca = Y_shape

    # Y-PCA
    pca_y, Z, Ky, evr_y = pca_fit_y_auto(Y_shape_for_pca, pca_var=pca_var, pca_max=pca_max, seed=seed)
    logger.info(f"[Y-PCA] 选取主成分 Ky={Ky} | EVR_y={evr_y:.4f}")

    # target transform & build PC models
    tt = TargetTransformer(target_tf).fit(Z)
    Zt = tt.transform(Z)
    logger.info(f"[TF] 目标变换: {target_tf}")

    models = build_pc_models(regressor, pc_ridge_alpha, model_path, device, seed)
    weights = parse_pc_blend(list(pc_models), pc_blend)

    # Train each PC model and fuse
    Zt_hat_sum = None
    for name in pc_models:
        mdl = models[name]
        mdl.fit(X, Zt)
        Zt_hat = mdl.predict(X)
        if Zt_hat_sum is None: Zt_hat_sum = np.zeros_like(Zt_hat)
        Zt_hat_sum += weights.get(name, 0.0) * Zt_hat
        logger.info(f"[PC] 训练完成：{name}")

    Z_hat = tt.inverse_transform(Zt_hat_sum)
    Y_shape_hat = pca_y.inverse_transform(Z_hat)
    if ywhite is not None: Y_shape_hat = ywhite.inverse_transform(Y_shape_hat)
    shape_norms = np.linalg.norm(Y_shape_hat, ord=2, axis=1, keepdims=True) + 1e-8
    Y_shape_hat = Y_shape_hat / shape_norms

    # AMP strategy
    n = Y_shape_hat.shape[0]
    if amp_norm == "none":
        A_hat = np.ones((n,1), dtype=np.float32)
    else:
        if amp_strategy in ("unit",):
            A_hat = np.ones((n,1), dtype=np.float32)
            logger.info("[AMP] 使用 unit：预测幅度恒为 1（只输出单位范数谱形）")
        elif amp_strategy in ("const_median", "fold_median"):
            A_const = float(np.median(A))
            A_hat = np.full((n,1), A_const, dtype=np.float32)
            logger.info(f"[AMP] 使用 const：全体中位幅度 A={A_const:.6f}")
        else:
            logger.info("[AMP] 训练幅度回归器（log-amp）")
            reg_amp = make_amp_estimator(kind=amp_model, model_path=model_path, device=device)
            logAtr = np.log(A.ravel() + 1e-8).astype(np.float32)
            reg_amp.fit(X, logAtr)
            cal = AmpCalibrator().fit(reg_amp.predict(X), logAtr)
            logA_pred = cal.predict(reg_amp.predict(X)).astype(np.float32)
            logA_pred, bounds = clip_logA_by_quantile(logA_pred, logAtr, q_low=amp_clip[0], q_high=amp_clip[1])
            logger.info(f"[AMP] 分位裁剪区间(logA)：[{bounds[0]:.3f}, {bounds[1]:.3f}]")
            A_hat = np.exp(logA_pred).reshape(-1,1).astype(np.float32)

    Y_hat = (Y_shape_hat * A_hat).astype(np.float32)
    Y_hat = smooth_1d(Y_hat, win=int(smooth_win), passes=int(smooth_passes))
    if not np.all(np.isfinite(Y_hat)):
        y_mean = np.nanmean(Y, axis=0, keepdims=True).astype(np.float32)
        Y_hat = np.where(np.isfinite(Y_hat), Y_hat, y_mean)

    meta = dict(
        method="pca_two_stage_shape_amp_strategy",
        X_PCA=dict(enabled=(pca_x is not None), Kx=Kx, EVR_x=evr_x),
        Y_PCA=dict(Ky=Ky, EVR_y=evr_y), amp_norm=amp_norm, target_tf=target_tf,
        pc_models=list(pc_models), pc_blend=weights, pc_ridge_alpha=float(pc_ridge_alpha),
        amp_strategy=amp_strategy, amp_clip=list(amp_clip),
        y_whiten=y_whiten, smooth_win=int(smooth_win), smooth_passes=int(smooth_passes),
    )
    return (Y, Y_hat, models, pca_y, tt, meta)


# ---------------- OOF（原样保留） ----------------
def oof_pca_two_stage_plus(
    X_df, Y_df, ids, spec_cols, folds=6, seed=42,
    amp_norm="l2",
    pca_x_var=None, pca_x_max=None,
    pca_var=0.99, pca_max=16, target_tf="yeojohnson",
    regressor="tabpfn", amp_model="ridge",
    pc_models=("tabpfn","ridge"), pc_blend=None, pc_ridge_alpha=10.0,
    amp_strategy="learn", amp_clip=(0.05,0.95),
    y_whiten="none", smooth_win=0, smooth_passes=1,
    model_path=None, device="cpu", use_tqdm=False, logger=None,
):
    X_all_raw, _ = impute_X_numeric_median(X_df)
    Y_all = Y_df[spec_cols].to_numpy(dtype=np.float32)
    n, m = Y_all.shape
    Y_oof = np.full((n, m), np.nan, dtype=np.float32); fold_idx = np.full(n, -1, dtype=int)

    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fi, (tr, va) in enumerate(kf.split(X_all_raw), 1):
        Xtr_raw, Xva_raw = X_all_raw[tr], X_all_raw[va]
        Ytr, Yva = Y_all[tr], Y_all[va]

        scaler_x = StandardScaler().fit(Xtr_raw)
        Xtr_s = scaler_x.transform(Xtr_raw); Xva_s = scaler_x.transform(Xva_raw)
        pca_x, Xtr, Kx, evr_x = pca_fit_x_auto(Xtr_s, pca_x_var=pca_x_var, pca_x_max=pca_x_max, seed=seed)
        Xva = pca_x.transform(Xva_s) if pca_x is not None else Xva_s

        Ytr_shape, Atr = amplitude_normalize(Ytr, mode=amp_norm)
        if y_whiten == "per_energy":
            ywhite = PerEnergyWhiten().fit(Ytr_shape)
            Ytr_pca_in = ywhite.transform(Ytr_shape)
        else:
            ywhite = None; Ytr_pca_in = Ytr_shape

        pca_y, Ztr, Ky, evr_y = pca_fit_y_auto(Ytr_pca_in, pca_var=pca_var, pca_max=pca_max, seed=seed)
        tt = TargetTransformer(target_tf).fit(Ztr)
        Ztr_t = tt.transform(Ztr)

        models = build_pc_models(regressor, pc_ridge_alpha, model_path, device, seed)
        weights = parse_pc_blend(list(pc_models), pc_blend)

        Zva_t_sum = None
        for name in pc_models:
            mdl = models[name]
            mdl.fit(Xtr, Ztr_t)
            Zva_t_hat = mdl.predict(Xva)
            if Zva_t_sum is None: Zva_t_sum = np.zeros_like(Zva_t_hat)
            Zva_t_sum += weights.get(name, 0.0) * Zva_t_hat

        Zva_hat = tt.inverse_transform(Zva_t_sum)
        Yva_shape_hat = pca_y.inverse_transform(Zva_hat)
        if ywhite is not None: Yva_shape_hat = ywhite.inverse_transform(Yva_shape_hat)
        sn = np.linalg.norm(Yva_shape_hat, ord=2, axis=1, keepdims=True) + 1e-8
        Yva_shape_hat = Yva_shape_hat / sn

        if amp_norm == "none":
            Ava_hat = np.ones((Yva_shape_hat.shape[0], 1), dtype=np.float32)
        else:
            if amp_strategy == "unit":
                Ava_hat = np.ones((Yva_shape_hat.shape[0], 1), dtype=np.float32)
            elif amp_strategy == "fold_median":
                A_const = float(np.median(Atr))
                Ava_hat = np.full((Yva_shape_hat.shape[0], 1), A_const, dtype=np.float32)
            elif amp_strategy == "const_median":
                A_const = float(np.median(Y_all))
                Ava_hat = np.full((Yva_shape_hat.shape[0], 1), A_const, dtype=np.float32)
            else:
                reg_amp = make_amp_estimator(kind=amp_model, model_path=model_path, device=device)
                logAtr = np.log(Atr.ravel() + 1e-8).astype(np.float32)
                reg_amp.fit(Xtr, logAtr)
                cal = AmpCalibrator().fit(reg_amp.predict(Xtr), logAtr)
                logA_va = cal.predict(reg_amp.predict(Xva)).astype(np.float32)
                Ava_hat = np.exp(np.clip(logA_va, np.quantile(logAtr,0.05), np.quantile(logAtr,0.95))).reshape(-1, 1).astype(np.float32)

        Yva_hat = (Yva_shape_hat * Ava_hat).astype(np.float32)
        Y_oof[va] = Yva_hat; fold_idx[va] = fi
        logger.info(f"[VAL PRED - ENS] fold {fi}/{folds} | rows={len(va)} | cols={m}")
    return Y_oof, fold_idx


# ---------------- PCA scatter helpers（原样保留） ----------------
def plot_pca_scatters(Z_embed, Y_shape_for_pca, ids, out_path="pca_scatter.png", dpi=300, seed=42, labels=None):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    pca_e = PCA(n_components=2, random_state=seed)
    Xe = pca_e.fit_transform(np.asarray(Z_embed, dtype=np.float32))
    pca_o = PCA(n_components=2, random_state=seed)
    Xo = pca_o.fit_transform(np.asarray(Y_shape_for_pca, dtype=np.float32))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
    def _scatter(ax, X, title):
        if labels is not None:
            uniq = pd.unique(labels)
            palette = plt.get_cmap("tab20")
            color_map = {lab: palette(i % 20) for i, lab in enumerate(uniq)}
            for lab in uniq:
                mask = (labels == lab)
                ax.scatter(X[mask,0], X[mask,1], s=26, alpha=0.9, lw=0, label=str(lab),
                           color=color_map[lab])
            ax.legend(frameon=False, loc="best", fontsize=9)
        else:
            c = np.linspace(0, 1, X.shape[0])
            ax.scatter(X[:,0], X[:,1], s=24, alpha=0.9, lw=0, c=c, cmap="viridis")
        ax.set_title(title, fontsize=13, weight="bold")
        ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")
        for spine in ["top","right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(1.2); ax.spines["bottom"].set_linewidth(1.2)
        ax.tick_params(direction="out", width=1.0, length=4)
    _scatter(axes[0], Xe, "Embedded data + PCA")
    _scatter(axes[1], Xo, "Original data + PCA")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def pca_dr_scatter_with_residual(Y_shape_for_pca, pca_y, Ky, out_png="out/pca_dr_scatter.png", dpi=300, seed=42):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    Y_in = np.asarray(Y_shape_for_pca, dtype=np.float32)
    pca2 = PCA(n_components=2, random_state=seed).fit(Y_in)
    X2 = pca2.transform(Y_in); evr2 = pca2.explained_variance_ratio_.sum()
    Z = pca_y.transform(Y_in)
    Y_rec = pca_y.inverse_transform(Z)
    num = np.linalg.norm(Y_in - Y_rec, axis=1); den = np.linalg.norm(Y_in, axis=1) + 1e-8
    rel_err = (num / den)
    plt.figure(figsize=(7.2, 5.6))
    sc = plt.scatter(X2[:,0], X2[:,1], c=rel_err, cmap="viridis", s=28, alpha=0.9, linewidths=0)
    cbar = plt.colorbar(sc); cbar.set_label(f"Relative reconstruction error  (K={Ky})", fontsize=10)
    plt.title(f"PCA scatter (coords=PC1/PC2, cum={evr2*100:.1f}%)\ncolor = error after projecting to K={Ky}",
              fontsize=12, weight="bold")
    plt.xlabel("PC 1"); plt.ylabel("PC 2")
    for sp in ["top","right"]:
        plt.gca().spines[sp].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.2); plt.gca().spines["bottom"].set_linewidth(1.2)
    plt.tick_params(direction="out", width=1.0, length=4)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white"); plt.close()

def shepard_diagram(Y_shape_for_pca, pca_y, out_png="out/shepard_pcaK.png", dpi=300):
    import matplotlib.pyplot as plt
    from sklearn.metrics import pairwise_distances
    Y_in = np.asarray(Y_shape_for_pca, dtype=np.float32)
    Z = pca_y.transform(Y_in)
    Y_rec = pca_y.inverse_transform(Z)
    d0 = pairwise_distances(Y_in, metric="euclidean")
    dK = pairwise_distances(Y_rec, metric="euclidean")
    iu = np.triu_indices_from(d0, k=1)
    x, y = d0[iu], dK[iu]
    plt.figure(figsize=(5.6,5.2))
    plt.scatter(x, y, s=8, alpha=0.6)
    lim = (0, max(x.max(), y.max()))
    plt.plot(lim, lim, "r--", lw=1.2, label="y=x (perfect preservation)")
    plt.xlim(lim); plt.ylim(lim)
    plt.xlabel("Original pairwise distance"); plt.ylabel("Reconstructed (K PCs) distance")
    plt.title("Shepard diagram: distance preservation after PCA-K"); plt.legend(frameon=False)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=dpi, bbox_inches="tight"); plt.close()

def _weighted_pc_predict(models, weights, X):
    Zt_hat_sum = None
    for name, mdl in models.items():
        if name not in weights or weights[name] <= 0: continue
        Zt_hat = mdl.predict(X)
        if Zt_hat_sum is None: Zt_hat_sum = np.zeros_like(Zt_hat)
        Zt_hat_sum += float(weights[name]) * Zt_hat
    return Zt_hat_sum

def permutation_importance_pc(models, weights, X, Zt_true, n_repeats=5, seed=42):
    rng = np.random.RandomState(seed)
    Zt_pred = _weighted_pc_predict(models, weights, X)
    base_mse = np.mean((Zt_true - Zt_pred)**2)
    d = X.shape[1]
    imp = np.zeros((n_repeats, d), dtype=np.float32)
    Xc = X.copy()
    for r in range(n_repeats):
        for j in range(d):
            backup = Xc[:, j].copy()
            rng.shuffle(Xc[:, j])
            Zt_perm = _weighted_pc_predict(models, weights, Xc)
            imp[r, j] = np.mean((Zt_true - Zt_perm)**2) - base_mse
            Xc[:, j] = backup
    return imp.mean(axis=0), imp.std(axis=0)

def plot_perm_importance(feature_names, imp_mean, imp_std, topk=20, out_png="out/perm_importance.png", title="Permutation importance (PC predictors)"):
    import matplotlib.pyplot as plt
    idx = np.argsort(imp_mean)[::-1][:topk]
    fm = imp_mean[idx]; fs = imp_std[idx]; names = [feature_names[i] for i in idx]
    y = np.arange(len(idx))
    plt.figure(figsize=(8, 0.5*len(idx)+1.5))
    plt.barh(y, fm, xerr=fs, alpha=0.85)
    plt.gca().invert_yaxis()
    plt.yticks(y, names, fontsize=10)
    plt.xlabel("ΔMSE after permutation"); plt.title(title, fontsize=12, weight="bold")
    for sp in ["top","right"]:
        plt.gca().spines[sp].set_visible(False)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()

def _parse_nm_bins_string(bins_str):
    out = []
    for seg in bins_str.split(","):
        a,b = seg.strip().split("-")
        out.append((float(a), float(b)))
    return out

def _label_by_nm_bins(E_eV_axis, Y_shape, nm_bins):
    lam_axis = 1240.0 / np.asarray(E_eV_axis, dtype=np.float64)
    peak_idx = np.argmax(np.abs(Y_shape), axis=1)
    peak_nm = lam_axis[peak_idx]
    labels = []
    for v in peak_nm:
        lab = "NA"
        for lo,hi in nm_bins:
            if (v >= lo) and (v < hi):
                lab = f"{int(lo)}–{int(hi)} nm"; break
        labels.append(lab)
    return np.array(labels, dtype=object)

def umap_or_tsne_scatter(Y_shape_for_pca, color_mode="peak", ids=None, bins_csv=None,
                         out_png="out/umap_scatter.png", n_neighbors=15, min_dist=0.1,
                         tsne_fallback=False, seed=42,
                         e_axis_eV=None, bins_mode="none", bins_nm=None):
    import matplotlib.pyplot as plt
    Y = np.asarray(Y_shape_for_pca, dtype=np.float32)
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist), random_state=seed, metric="euclidean")
        X2 = reducer.fit_transform(Y); subtitle = f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
    except Exception:
        if not tsne_fallback:
            raise RuntimeError("未安装 umap-learn（pip install umap-learn）。如用 t-SNE 兜底请加 --tsne_fallback。")
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=30, random_state=seed, learning_rate='auto', init='pca')
        X2 = reducer.fit_transform(Y); subtitle = "t-SNE (perplexity=30)"

    plt.figure(figsize=(7.2,5.6))
    if color_mode == "bins":
        labels = None
        if bins_mode == "csv" and bins_csv is not None and ids is not None:
            bins_df = pd.read_csv(bins_csv)
            if set(["id","bin_label"]).issubset(bins_df.columns):
                lab_map = dict(zip(bins_df["id"].astype(str), bins_df["bin_label"].astype(str)))
                labels = np.array([lab_map.get(str(i), "NA") for i in ids], dtype=object)
        elif bins_mode in ("nm_auto","nm_custom"):
            if e_axis_eV is None:
                raise RuntimeError("需要 e_axis_eV 来计算主峰波长。")
            if bins_mode == "nm_auto":
                nm_bins_use = [(80,136),(154,210),(228,284),(302,358),(376,450)]
            else:
                nm_bins_use = _parse_nm_bins_string(bins_nm) if (bins_nm and len(bins_nm)>0) else []
                if not nm_bins_use:
                    raise RuntimeError("nm_custom 需提供 --umap_bins_nm 'a-b,c-d,...'")
            labels = _label_by_nm_bins(e_axis_eV, Y, nm_bins_use)

        if labels is not None:
            uniq = pd.unique(labels); palette = plt.get_cmap("tab20")
            for k, lab in enumerate(uniq):
                mask = (labels == lab)
                plt.scatter(X2[mask,0], X2[mask,1], s=26, alpha=0.95, lw=0, label=str(lab), color=palette(k%20))
            plt.legend(frameon=False, fontsize=10, title="Band")
        else:
            nE = Y.shape[1]
            peak_idx = np.argmax(np.abs(Y), axis=1)
            peak_val = peak_idx.astype(np.float32) / max(1, nE-1)
            sc = plt.scatter(X2[:,0], X2[:,1], c=peak_val, cmap="viridis", s=26, alpha=0.95, lw=0)
            cb = plt.colorbar(sc); cb.set_label("Normalized main-peak energy index", fontsize=10)
    else:
        nE = Y.shape[1]
        peak_idx = np.argmax(np.abs(Y), axis=1)
        peak_val = peak_idx.astype(np.float32) / max(1, nE-1)
        sc = plt.scatter(X2[:,0], X2[:,1], c=peak_val, cmap="viridis", s=26, alpha=0.95, lw=0)
        cb = plt.colorbar(sc); cb.set_label("Normalized main-peak energy index", fontsize=10)

    plt.title(f"Spectral-shape embedding — {subtitle}", fontsize=12, weight="bold")
    plt.xlabel("dim-1"); plt.ylabel("dim-2")
    for sp in ["top","right"]:
        plt.gca().spines[sp].set_visible(False)
    plt.gca().spines["left"].set_linewidth(1.2); plt.gca().spines["bottom"].set_linewidth(1.2)
    plt.tick_params(direction="out", width=1.0, length=4)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()


# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True); ap.add_argument("--Y", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--out_metrics", default=None); ap.add_argument("--out_meta", default=None)
    ap.add_argument("--method", choices=["pca","direct"], default="pca")

    ap.add_argument("--pca_x_var", type=float, default=None)
    ap.add_argument("--pca_x_max", type=int, default=None)

    ap.add_argument("--pca_var", type=float, default=0.99)
    ap.add_argument("--pca_max", type=int, default=16)
    ap.add_argument("--amp_norm", choices=["none","maxabs","l2"], default="l2")
    ap.add_argument("--target_tf", choices=["none","yeojohnson","quantile"], default="yeojohnson")
    ap.add_argument("--y_whiten", choices=["none","per_energy"], default="none")

    ap.add_argument("--regressor", choices=["tabpfn","auto_tabpfn"], default="tabpfn")
    ap.add_argument("--pc_models", type=str, default="tabpfn,ridge")
    ap.add_argument("--pc_blend", type=str, default=None)
    ap.add_argument("--pc_ridge_alpha", type=float, default=10.0)

    ap.add_argument("--amp_model", choices=["ridge","huber","tabpfn"], default="ridge")
    ap.add_argument("--amp_clip", type=str, default="0.05,0.95")
    ap.add_argument("--amp_strategy", choices=["learn","const_median","fold_median","unit"], default="learn")

    ap.add_argument("--shape_metrics", action="store_true")

    ap.add_argument("--smooth_win", type=int, default=0)
    ap.add_argument("--smooth_passes", type=int, default=1)

    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--out_oof", type=str, default=None)
    ap.add_argument("--out_folds", type=str, default=None)

    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--model_path", type=str, default=None)

    ap.add_argument("--pca_scatter", action="store_true")
    ap.add_argument("--pca_scatter_out", type=str, default="pca_scatter.png")
    ap.add_argument("--pca_scatter_dpi", type=int, default=300)
    ap.add_argument("--pca_color", choices=["none","bins"], default="none")
    ap.add_argument("--color_bins_csv", type=str, default=None)

    ap.add_argument("--pca_dr_scatter", action="store_true")
    ap.add_argument("--pca_dr_out", type=str, default="out/pca_dr_scatter.png")
    ap.add_argument("--pca_dr_dpi", type=int, default=300)
    ap.add_argument("--pca_dr_shepard", action="store_true")

    ap.add_argument("--explain_perm", action="store_true")
    ap.add_argument("--perm_repeats", type=int, default=5)
    ap.add_argument("--perm_topk", type=int, default=20)
    ap.add_argument("--explain_out", type=str, default="out/perm_importance.png")

    ap.add_argument("--umap_scatter", action="store_true")
    ap.add_argument("--umap_color", choices=["peak","bins"], default="peak")
    ap.add_argument("--umap_out", type=str, default="out/umap_scatter.png")
    ap.add_argument("--umap_neighbors", type=int, default=15)
    ap.add_argument("--umap_min_dist", type=float, default=0.1)
    ap.add_argument("--tsne_fallback", action="store_true")
    ap.add_argument("--umap_bins_mode", choices=["none","csv","nm_auto","nm_custom"], default="none")
    ap.add_argument("--umap_bins_nm", type=str, default="")

    ap.add_argument("--log-level", choices=["DEBUG","INFO","WARNING","ERROR"], default="INFO")
    ap.add_argument("--progress", action="store_true")

    args = ap.parse_args()
    logger = setup_logger(args.log_level)
    use_tqdm = bool(args.progress and _HAS_TQDM)
    if args.progress and not _HAS_TQDM:
        logger.warning("未检测到 tqdm，已关闭进度条（pip install tqdm 可启用）")
    if args.regressor == "auto_tabpfn" and not _HAS_AUTO:
        logger.warning("未检测到 tabpfn-extensions，已回退为 tabpfn。"); args.regressor = "tabpfn"

    ql, qh = (float(x) for x in args.amp_clip.split(","))
    pc_models = [s.strip() for s in args.pc_models.split(",") if s.strip()]

    X_df, Y_df, ids, spec_cols = read_xy(args.X, args.Y)
    E_axis_eV = e_axis_from_cols(spec_cols)
    logger.info(f"读取完成：样本数={len(ids)} | 特征维={X_df.shape[1]-1} | 能量点数={len(spec_cols)}")

    if args.method == "pca":
        Y_true, Y_pred, models, pca_y, tt, meta = train_predict_pca_two_stage(
            X_df, Y_df, spec_cols,
            amp_norm=args.amp_norm,
            pca_x_var=args.pca_x_var, pca_x_max=args.pca_x_max,
            pca_var=args.pca_var, pca_max=args.pca_max, target_tf=args.target_tf,
            regressor=args.regressor, amp_model=args.amp_model,
            pc_models=pc_models, pc_blend=args.pc_blend, pc_ridge_alpha=args.pc_ridge_alpha,
            amp_strategy=args.amp_strategy, amp_clip=(ql,qh), amp_calib=True,
            y_whiten=args.y_whiten, smooth_win=args.smooth_win, smooth_passes=args.smooth_passes,
            model_path=args.model_path, device=args.device, seed=42,
            use_tqdm=use_tqdm, logger=logger,
        )
    else:
        logger.info("[STEP] direct 点估计")
        X_raw, _ = impute_X_numeric_median(X_df)
        X = StandardScaler().fit_transform(X_raw)
        Y_true = Y_df[spec_cols].to_numpy(dtype=np.float32)
        Y_pred = np.zeros_like(Y_true)
        iterable = range(Y_true.shape[1])
        if use_tqdm and _HAS_TQDM: iterable = tqdm(iterable, desc="[FIT-ALL]")
        for j in iterable:
            reg = make_base_estimator(kind=args.regressor, model_path=args.model_path, device=args.device)
            reg.fit(X, Y_true[:, j]); Y_pred[:, j] = reg.predict(X)
        meta = dict(method="direct_point")
        pca_y = None
        models = None

    P = pd.DataFrame(Y_pred, columns=spec_cols); P.insert(0, "id", ids)
    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)
    P.to_csv(args.out_pred, index=False); logger.info(f"[SAVE] 预测写入：{args.out_pred}")

    if args.out_metrics:
        M = metric_table(Y_true, Y_pred)
        if args.shape_metrics:
            M.update(shape_metrics_table(Y_true, Y_pred, sign_invariant=True))
        pd.DataFrame([M]).to_csv(args.out_metrics, index=False)
        logger.info(f"[SAVE] 指标写入：{args.out_metrics}")

    Y_all = Y_df[spec_cols].to_numpy(dtype=np.float32)
    Y_shape_all, _ = amplitude_normalize(Y_all, mode=args.amp_norm)
    Y_shape_for_pca = Y_shape_all

    if args.method == "pca" and args.pca_scatter and (pca_y is not None):
        Z_embed = pca_y.transform(Y_shape_for_pca)
        labels = None
        if args.pca_color == "bins" and args.color_bins_csv:
            try:
                bins_df = pd.read_csv(args.color_bins_csv)
                if set(["id","bin_label"]).issubset(bins_df.columns):
                    lab_map = dict(zip(bins_df["id"].astype(str), bins_df["bin_label"].astype(str)))
                    labels = np.array([lab_map.get(str(i), "NA") for i in ids], dtype=object)
            except Exception as e:
                logger.warning(f"[PCA-PLOT] 读取 bins CSV 失败：{e}")
        Path(args.pca_scatter_out).parent.mkdir(parents=True, exist_ok=True)
        plot_pca_scatters(Z_embed, Y_shape_for_pca, ids,
                          out_path=args.pca_scatter_out,
                          dpi=int(args.pca_scatter_dpi),
                          labels=labels)
        logger.info(f"[PCA-PLOT] 已保存：{args.pca_scatter_out}")

    if args.method == "pca" and args.pca_dr_scatter and (pca_y is not None):
        pca_dr_scatter_with_residual(
            Y_shape_for_pca=Y_shape_for_pca,
            pca_y=pca_y,
            Ky=int(getattr(pca_y, "n_components_", 0) or Y_shape_for_pca.shape[1]),
            out_png=args.pca_dr_out,
            dpi=int(args.pca_dr_dpi),
            seed=42
        )
        logger.info(f"[PCA-DR] 已保存：{args.pca_dr_out}")
        if args.pca_dr_shepard:
            out_sh = str(Path(args.pca_dr_out).with_name(Path(args.pca_dr_out).stem + "_shepard.png"))
            shepard_diagram(Y_shape_for_pca, pca_y, out_png=out_sh, dpi=int(args.pca_dr_dpi))
            logger.info(f"[PCA-DR] Shepard 图已保存：{out_sh}")

    if args.method == "pca" and args.explain_perm and (models is not None):
        X_raw, _ = impute_X_numeric_median(X_df)
        Xs = StandardScaler().fit_transform(X_raw)
        pca_x, X_mat, _, _ = pca_fit_x_auto(Xs, pca_x_var=args.pca_x_var, pca_x_max=args.pca_x_max, seed=42)
        feature_names = [c for c in X_df.columns if c != "id"]
        if pca_x is not None:
            feature_names = [f"PCX_{i+1}" for i in range(X_mat.shape[1])]
        Z_true = pca_y.transform(Y_shape_for_pca)
        tt_tmp = TargetTransformer(args.target_tf).fit(Z_true)
        Zt_true = tt_tmp.transform(Z_true)
        weights = parse_pc_blend([s.strip() for s in args.pc_models.split(",") if s.strip()], args.pc_blend)
        imp_mean, imp_std = permutation_importance_pc(models, weights, X_mat, Zt_true, n_repeats=int(args.perm_repeats), seed=42)
        plot_perm_importance(feature_names, imp_mean, imp_std, topk=int(args.perm_topk), out_png=args.explain_out)
        logger.info(f"[EXPLAIN] 置换重要度已保存：{args.explain_out}")

    if args.umap_scatter:
        umap_or_tsne_scatter(
            Y_shape_for_pca,
            color_mode=args.umap_color,
            ids=ids, bins_csv=args.color_bins_csv,
            out_png=args.umap_out,
            n_neighbors=int(args.umap_neighbors),
            min_dist=float(args.umap_min_dist),
            tsne_fallback=bool(args.tsne_fallback),
            seed=42,
            e_axis_eV=E_axis_eV,
            bins_mode=args.umap_bins_mode,
            bins_nm=args.umap_bins_nm
        )
        logger.info(f"[UMAP] 已保存：{args.umap_out}")

    if args.cv and int(args.cv) > 1:
        folds = int(args.cv); logger.info(f"[CV] Start {folds}-fold OOF …")
        Y_oof, fold_idx = oof_pca_two_stage_plus(
            X_df, Y_df, ids, spec_cols, folds=folds, seed=42,
            amp_norm=args.amp_norm,
            pca_x_var=args.pca_x_var, pca_x_max=args.pca_x_max,
            pca_var=args.pca_var, pca_max=args.pca_max, target_tf=args.target_tf,
            regressor=args.regressor, amp_model=args.amp_model,
            pc_models=pc_models, pc_blend=args.pc_blend, pc_ridge_alpha=args.pc_ridge_alpha,
            amp_strategy=args.amp_strategy, amp_clip=(ql,qh),
            y_whiten=args.y_whiten, smooth_win=args.smooth_win, smooth_passes=args.smooth_passes,
            model_path=args.model_path, device=args.device, use_tqdm=use_tqdm, logger=logger,
        )
        if args.out_oof:
            oof_df = pd.DataFrame(Y_oof, columns=spec_cols); oof_df.insert(0, "id", ids)
            Path(args.out_oof).parent.mkdir(parents=True, exist_ok=True)
            oof_df.to_csv(args.out_oof, index=False); logger.info(f"[SAVE] OOF写入：{args.out_oof}")
        if args.out_folds:
            folds_df = pd.DataFrame({"id": ids, "fold": fold_idx})
            Path(args.out_folds).parent.mkdir(parents=True, exist_ok=True)
            folds_df.to_csv(args.out_folds, index=False); logger.info(f"[SAVE] 折号写入：{args.out_folds}")

if __name__ == "__main__":
    main()
