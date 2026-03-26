#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# XGB
python predict_full_spectrum_xgb_cat.py \
  --model xgb \
  --X ./out/X_compact3D.cleaned.csv \
  --Y ./out_ecd/Y_all.csv \
  --out_pred preds_oof_xgb_shape_pca.csv \
  --out_oof preds_oof_xgb_shape_pca.csv \
  --out_metrics metrics_xgb.csv \
  --method pca --cv 6

# CAT
python predict_full_spectrum_xgb_cat.py \
  --model cat \
  --X ./out/X_compact3D.cleaned.csv \
  --Y ./out_ecd/Y_all.csv \
  --out_pred preds_oof_cat_shape_pca.csv \
  --out_oof preds_oof_cat_shape_pca.csv \
  --out_metrics metrics_cat.csv \
  --method pca --cv 6
ECD 光谱多输出回归（只学形状；L2 行归一化）
- 仅使用一种模型：--model {xgb, cat}
- 支持 method=direct / method=pca
- 评估：R2 / MAE / RMSE（形状空间）+ 平均余弦相似度（raw & sign-invariant）
- I/O 与此前一致：X.csv(id, f1..), Y.csv(id, y_*)

依赖：numpy, pandas, scikit-learn, xgboost, catboost
"""

import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------- utils ----------
def cosine_sim_rows(A, B, eps=1e-12):
    """逐样本余弦相似度（形状指标）"""
    A = A.copy(); B = B.copy()
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    return np.sum(A * B, axis=1)

def rmse_compat(y_true, y_pred):
    """兼容旧版 sklearn 的 RMSE 计算"""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return mean_squared_error(y_true, y_pred) ** 0.5

def load_xy(X_path, Y_path):
    X = pd.read_csv(X_path)
    Y = pd.read_csv(Y_path)
    assert "id" in X.columns and "id" in Y.columns, "X/Y 必须包含 id 列"
    df = X.merge(Y, on="id", how="inner")
    y_cols = [c for c in df.columns if c.startswith("y_")]
    f_cols = [c for c in df.columns if c not in (["id"] + y_cols)]
    ids   = df["id"].values
    Xmat  = df[f_cols].values.astype(np.float64)
    Ymat  = df[y_cols].values.astype(np.float64)
    return ids, f_cols, y_cols, Xmat, Ymat

def l2_row_normalize(Y, eps=1e-12):
    """对每条谱做 L2 行归一化（只学形状）"""
    Y = Y.copy()
    s = np.linalg.norm(Y, axis=1, keepdims=True) + eps
    Y /= s
    return Y

# ---------- core ----------
def fit_predict_cv_single(model_name, base_estimator, X, Y, cv=5,
                          method="direct", pca_var=0.999, pca_max=32,
                          random_state=42, verbose=True):
    """
    单模型（xgb 或 cat）KFold 训练/验证。
    输入 Y 已是形状化后的矩阵。
    返回：
      oof:  OOF 预测（形状空间）
      pred_full: 全量训练预测（形状空间）
    """
    assert model_name in ("xgb", "cat")
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    n, m = Y.shape
    oof = np.zeros_like(Y, dtype=float)

    for fold, (tr, va) in enumerate(kf.split(X), start=1):
        Xtr, Xva = X[tr], X[va]
        Ytr, Yva = Y[tr], Y[va]

        if method == "direct":
            est = MultiOutputRegressor(base_estimator)
            est.fit(Xtr, Ytr)
            pred_va = est.predict(Xva)

        elif method == "pca":
            _pca = PCA(n_components=min(pca_max, min(len(tr), m-1)), svd_solver="full").fit(Ytr)
            if pca_var < 1.0:
                cumsum = np.cumsum(_pca.explained_variance_ratio_)
                k = int(np.searchsorted(cumsum, pca_var) + 1)
                k = min(k, pca_max)
                _pca = PCA(n_components=k, svd_solver="full").fit(Ytr)
            Ztr = _pca.transform(Ytr)
            est = MultiOutputRegressor(base_estimator)
            est.fit(Xtr, Ztr)
            pred_va = _pca.inverse_transform(est.predict(Xva))
        else:
            raise ValueError(f"Unknown method={method}")

        oof[va] = pred_va

        if verbose:
            r2 = r2_score(Yva, oof[va], multioutput="uniform_average")
            mae = mean_absolute_error(Yva, oof[va])
            rmse = rmse_compat(Yva, oof[va])

            cos_raw_each = cosine_sim_rows(Yva, oof[va])
            cos_raw = float(cos_raw_each.mean())
            cos_signinv = float(np.abs(cos_raw_each).mean())

            print(f"[{model_name.upper()} | CV {fold}/{cv}] "
                  f"R2={r2:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f} | "
                  f"raw cos={cos_raw:.4f} | sign-inv cos={cos_signinv:.4f}")

    # 全量重训
    if method == "direct":
        full_est = MultiOutputRegressor(base_estimator)
        full_est.fit(X, Y)
        pred_full = full_est.predict(X)
    else:
        pca_full = PCA(n_components=min(pca_max, min(len(X), m-1)), svd_solver="full").fit(Y)
        if pca_var < 1.0:
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            k = int(np.searchsorted(cumsum, pca_var) + 1)
            k = min(k, pca_max)
            pca_full = PCA(n_components=k, svd_solver="full").fit(Y)
        Z = pca_full.transform(Y)
        full_est = MultiOutputRegressor(base_estimator)
        full_est.fit(X, Z)
        pred_full = pca_full.inverse_transform(full_est.predict(X))

    return oof, pred_full

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["xgb","cat"], required=True,
                    help="选择本次训练/预测使用的模型：xgb 或 cat")
    ap.add_argument("--X", required=True)
    ap.add_argument("--Y", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--out_oof", required=True)
    ap.add_argument("--out_metrics", required=True)
    ap.add_argument("--out_meta", default=None)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--method", choices=["direct","pca"], default="direct")
    ap.add_argument("--pca_var", type=float, default=0.999)
    ap.add_argument("--pca_max", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)

    # XGBoost 超参
    ap.add_argument("--xgb_n_estimators", type=int, default=500)
    ap.add_argument("--xgb_learning_rate", type=float, default=0.05)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    ap.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    ap.add_argument("--xgb_tree_method", default="hist")

    # CatBoost 超参
    ap.add_argument("--cat_iterations", type=int, default=800)
    ap.add_argument("--cat_learning_rate", type=float, default=0.05)
    ap.add_argument("--cat_depth", type=int, default=6)
    ap.add_argument("--cat_l2_leaf_reg", type=float, default=6.0)
    ap.add_argument("--cat_rsm", type=float, default=0.8)
    ap.add_argument("--cat_subsample", type=float, default=0.8)
    ap.add_argument("--cat_loss_function", default="RMSE")
    ap.add_argument("--cat_verbose", type=int, default=0)

    args = ap.parse_args()

    ids, f_cols, y_cols, X, Y_raw = load_xy(args.X, args.Y)

    # 只学形状：L2 行归一化
    Y = l2_row_normalize(Y_raw)
    print(f"[INFO] model={args.model} | method={args.method} | cv={args.cv} | shape-only (L2) enabled")

    # 构造所选模型
    if args.model == "xgb":
        base = XGBRegressor(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            reg_lambda=args.xgb_reg_lambda,
            min_child_weight=args.xgb_min_child_weight,
            tree_method=args.xgb_tree_method,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
        )
    else:
        base =  CatBoostRegressor(
                task_type="CPU",
                thread_count=4,
                iterations=args.cat_iterations,
                learning_rate=args.cat_learning_rate,
                depth=args.cat_depth,
                l2_leaf_reg=args.cat_l2_leaf_reg,
                rsm=args.cat_rsm,
                subsample=args.cat_subsample,
                loss_function=args.cat_loss_function,
                random_seed=args.seed,
                verbose=args.cat_verbose,
                allow_writing_files=False,
            )

    # 训练 + OOF
    oof, pred_full = fit_predict_cv_single(
        args.model, base, X, Y,
        cv=args.cv, method=args.method,
        pca_var=args.pca_var, pca_max=args.pca_max,
        random_state=args.seed, verbose=True
    )

    # 评估（在形状空间）
    r2 = r2_score(Y, oof)
    mae = mean_absolute_error(Y, oof)
    rmse = rmse_compat(Y, oof)

    cos_raw_each = cosine_sim_rows(Y, oof)
    cos_raw = float(cos_raw_each.mean())
    cos_signinv = float(np.abs(cos_raw_each).mean())

    metrics = {
        "mode": "shape-only (L2 row-normalized)",
        "model": args.model,
        "R2_oof": float(r2),
        "MAE_oof": float(mae),
        "RMSE_oof": float(rmse),

        # 论文/排名用这个
        "MeanCosine_oof_signinv": float(cos_signinv),

        # 辅助观察
        "MeanCosine_oof_raw": float(cos_raw),

        "method": args.method,
        "cv": args.cv,
        "pca_var": args.pca_var,
        "pca_max": args.pca_max,
        "xgb_params": {k: getattr(args, k) for k in vars(args) if k.startswith("xgb_")},
        "cat_params": {k: getattr(args, k) for k in vars(args) if k.startswith("cat_")},
    }
    print(f"[FINAL] OOF ({args.model} / shape-only): raw cos={cos_raw:.4f} | sign-inv cos={cos_signinv:.4f}")

    # 输出（均为形状化光谱）
    Path(args.out_pred).parent.mkdir(parents=True, exist_ok=True)

    df_pred = pd.DataFrame(pred_full, columns=y_cols)
    df_pred.insert(0, "id", ids)
    df_pred.to_csv(args.out_pred, index=False)

    df_oof = pd.DataFrame(oof, columns=y_cols)
    df_oof.insert(0, "id", ids)
    df_oof.to_csv(args.out_oof, index=False)

    pd.DataFrame([metrics]).to_csv(args.out_metrics, index=False)

    if args.out_meta:
        with open(args.out_meta, "w") as f:
            json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
