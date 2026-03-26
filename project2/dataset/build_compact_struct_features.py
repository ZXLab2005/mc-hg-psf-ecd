#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python build_compact_struct_features_3d.py
在原紧凑特征基础上，加入三维不变特征：回转张量、库仑矩阵特征值、RDF 直方图、
以及 Ag 中心的几何(τ4/τ5/八面体畸变/手性度)。
输出: ./out/X_compact3D_1513033.csv
"""

import re
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# ---------------- 配置（可按需精简以控维） ----------------
ID_STR = "1513033"
EL_SET = ["H","C","N","O","Ag"]  # 只保留必要元素，避免多余列
PAIR_FOCUS = [("Ag","N"),("Ag","O"),("Ag","C"),
              ("C","C"),("C","N"),("C","O"),("N","O"),
              ("C","H"),("N","H"),("O","H")]          # 无 S/P/卤素
CENTER_EL_ANGLE = ["C","N","O","Ag"]
CENTER_PAIR_TORS = [("C","C"),("C","N"),("C","O"),("N","O"),
                    ("Ag","N"),("Ag","O")]

# 3D 相关
COULOMB_EIG_K = 16                        # 取前 K 个库仑矩阵特征值（越小越省维）
RDF_PAIRS = [("Ag","N"),("Ag","O"),("C","C"),("C","N"),("C","O")]
RDF_BINS = np.arange(2.0, 10.0+1e-9, 1.0)  # [2,3,...,10] Å 8 桶

# 原子序号（CM 用）；缺省给 0
ZMAP = {"H":1,"C":6,"N":7,"O":8,"S":16,"P":15,"F":9,"Cl":17,"Br":35,"I":53,"Ag":47}

# ---------------- 小工具 ----------------
def robust_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    num = s.str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', expand=False)
    return pd.to_numeric(num, errors="coerce")

def parse_label(lbl: str):
    m = re.match(r'([A-Za-z]+)(\d+)$', str(lbl).strip())
    return (m.group(1), int(m.group(2))) if m else ("X", None)

def find_one(root: Path, key: str):
    exact = root / f"{key}.csv"
    if exact.exists(): return exact
    cands = sorted(root.glob(f"*{key}*.csv"))
    return cands[0] if cands else None

def find_numeric_col(df: pd.DataFrame, patterns):
    for c in df.columns:
        if any(re.search(p, c, re.I) for p in patterns):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or robust_numeric(df[c]).notna().any():
            return c
    return None

def basic_stats(vals: pd.Series):
    v = pd.to_numeric(vals, errors="coerce").dropna().values
    if v.size == 0:
        return dict(mean=0.0, std=0.0, min=0.0, p10=0.0, p50=0.0, p90=0.0, max=0.0, n=0)
    return dict(mean=float(v.mean()), std=float(v.std()),
                min=float(v.min()), p10=float(np.quantile(v,0.10)),
                p50=float(np.quantile(v,0.50)), p90=float(np.quantile(v,0.90)),
                max=float(v.max()), n=int(v.size))

def circular_stats(vals_deg: pd.Series):
    x = pd.to_numeric(vals_deg, errors="coerce").dropna().values
    if x.size == 0:
        return dict(mean=0.0, std=0.0, r=0.0, sin_mean=0.0, cos_mean=1.0, n=0)
    rad = np.deg2rad(x); c, s = np.cos(rad), np.sin(rad)
    return dict(mean=float(np.mean(x)),
                std=float(np.std(x)),
                r=float(np.hypot(c.mean(), s.mean())),
                sin_mean=float(s.mean()), cos_mean=float(c.mean()),
                n=int(x.size))

# ---------------- 解析 Atoms 坐标 ----------------
def read_atoms_with_coords(p_atoms: Path):
    if p_atoms is None or not p_atoms.exists():
        return None
    A = pd.read_csv(p_atoms)
    # 找元素列（Label/Element/Symbol/Type）
    el_col = next((c for c in A.columns if re.search(r'(Label|Element|Symbol|Type)$', c, re.I)), None)
    # 坐标列（常见：x,y,z 或 X,Y,Z 或 Cartn_X 等）
    def _find_xyz(cols):
        lower = {c.lower():c for c in cols}
        for cand in [("x","y","z"), ("cartn_x","cartn_y","cartn_z"), ("fract_x","fract_y","fract_z")]:
            if all(k in lower for k in cand):
                return lower[cand[0]], lower[cand[1]], lower[cand[2]]
        # 退而求其次：找名字中含 x/y/z 的三列
        xs = [c for c in cols if re.search(r'\bx\b', c, re.I)]
        ys = [c for c in cols if re.search(r'\by\b', c, re.I)]
        zs = [c for c in cols if re.search(r'\bz\b', c, re.I)]
        if xs and ys and zs:
            return xs[0], ys[0], zs[0]
        return None, None, None
    xcol, ycol, zcol = _find_xyz(list(A.columns))
    if el_col is None or xcol is None:
        return None
    lab = A[el_col].astype(str).str.strip()
    el = lab.str.replace(r'(\d+)$','', regex=True).map(lambda s: re.match(r'[A-Za-z]+', s).group(0) if re.match(r'[A-Za-z]+', s) else "X")
    X = robust_numeric(A[xcol]); Y = robust_numeric(A[ycol]); Z = robust_numeric(A[zcol])
    df = pd.DataFrame({"label": lab, "el": el, "x": X, "y": Y, "z": Z}).dropna()
    return df

# ---------------- 3D: 回转张量 ----------------
def gyration_features(coords_df):
    R = coords_df[["x","y","z"]].values.astype(float)
    if len(R)==0:
        return {"Rg":0.0,"asphericity":0.0,"acylindricity":0.0,"kappa2":0.0}
    R = R - R.mean(axis=0, keepdims=True)
    G = (R.T @ R) / len(R)                  # 3x3
    w = np.linalg.eigvalsh(G)               # 升序: g1<=g2<=g3
    g1,g2,g3 = float(w[0]), float(w[1]), float(w[2])
    Rg = float(np.sqrt(g1+g2+g3))
    asph = g3 - 0.5*(g2+g1)
    acyl = g2 - g1
    k2 = 1.0 - 3.0*(g1*g2 + g2*g3 + g3*g1) / ((g1+g2+g3)**2 + 1e-12)
    return {"Rg":Rg, "asphericity":float(asph), "acylindricity":float(acyl), "kappa2":float(k2)}

# ---------------- 3D: 库仑矩阵特征值 ----------------
def coulomb_eigs(coords_df, K=16):
    if coords_df is None or len(coords_df)==0:
        return [0.0]*K
    xyz = coords_df[["x","y","z"]].values.astype(float)
    els = coords_df["el"].astype(str).map(ZMAP).fillna(0).values.astype(float)
    n = len(coords_df)
    M = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i==j:
                M[i,i] = 0.5 * (els[i] ** 2.4)
            else:
                rij = np.linalg.norm(xyz[i]-xyz[j]) + 1e-12
                M[i,j] = els[i]*els[j]/rij
    w = np.linalg.eigvalsh(M)
    w = np.sort(w)[::-1]                 # 降序
    if len(w) < K: w = np.pad(w, (0, K-len(w)))
    return [float(v) for v in w[:K]]

# ---------------- 3D: RDF 直方图 ----------------
def rdf_hist(coords_df, pair=("Ag","N"), bins=RDF_BINS):
    if coords_df is None: return [0]* (len(bins)-1)
    A = coords_df[coords_df["el"]==pair[0]]
    B = coords_df[coords_df["el"]==pair[1]]
    if len(A)==0 or len(B)==0:
        return [0]*(len(bins)-1)
    dists = []
    for _, ra in A[["x","y","z"]].iterrows():
        va = ra.values
        vb = B[["x","y","z"]].values
        ds = np.linalg.norm(vb - va, axis=1)
        dists.extend(ds.tolist())
    hist, _ = np.histogram(np.array(dists), bins=bins)
    return [int(v) for v in hist.tolist()]

# ---------------- 金属中心 3D 几何 ----------------
def metal_3d_features(coords_df, bonds_df, metal="Ag"):
    feats = {}
    if coords_df is None or bonds_df is None or "Atom1" not in bonds_df.columns: 
        # 占位
        keys = ["Ag_centers","Ag_CN_mean","Ag_tau4_mean","Ag_tau5_mean",
                "Ag_oct_dev90","Ag_oct_dev180","Ag_chiral_abs","Ag_chiral_signed"]
        return {k:0.0 for k in keys}
    # 坐标查表
    pos = {r["label"]: r[["x","y","z"]].values for _,r in coords_df.iterrows()}
    ele = {r["label"]: r["el"] for _,r in coords_df.iterrows()}

    # 构邻接
    adj = defaultdict(list)
    for _,r in bonds_df.iterrows():
        u,v = str(r["Atom1"]).strip(), str(r["Atom2"]).strip()
        adj[u].append(v); adj[v].append(u)

    # 只看金属中心
    agg_tau4, agg_tau5 = [], []
    dev90, dev180 = [], []
    chir_abs, chir_signed = [], []
    centers = 0

    for lbl, e in ele.items():
        if e != metal or lbl not in pos: 
            continue
        nbrs = [n for n in adj.get(lbl, []) if n in pos]   # 有坐标的邻居
        CN = len(nbrs)
        if CN==0: 
            continue
        centers += 1

        # 向量
        vecs = [pos[n]-pos[lbl] for n in nbrs]
        # 所有夹角
        angles = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                vi, vj = vecs[i], vecs[j]
                ni = np.linalg.norm(vi)+1e-12; nj = np.linalg.norm(vj)+1e-12
                ang = np.degrees(np.arccos(np.clip(np.dot(vi,vj)/(ni*nj), -1.0, 1.0)))
                angles.append(ang)
        # τ4/τ5
        if CN==4 and len(angles)>=2:
            alpha, beta = sorted(angles, reverse=True)[:2]
            tau4 = (360.0 - (alpha + beta)) / 141.0
            agg_tau4.append(float(tau4))
        if CN==5 and len(angles)>=2:
            alpha, beta = sorted(angles, reverse=True)[:2]
            tau5 = (alpha - beta) / 60.0
            agg_tau5.append(float(tau5))
        # 八面体畸变近似（CN>=6）：统计与 90/180 的偏差
        if CN>=6 and len(angles)>0:
            dev90.append(float(np.mean(np.abs(np.array(angles)-90.0))))
            dev180.append(float(np.min(np.abs(np.array(angles)-180.0))))  # 最接近 180 的成对近似

        # 手性度：三重积归一化
        if CN>=3:
            tvals = []
            m = len(vecs)
            for i in range(m):
                for j in range(i+1,m):
                    for k in range(j+1,m):
                        a,b,c = vecs[i], vecs[j], vecs[k]
                        vol = np.dot(a, np.cross(b,c))
                        denom = (np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(c) + 1e-12)
                        tvals.append(vol/denom)
            if tvals:
                chir_abs.append(float(np.mean(np.abs(tvals))))
                chir_signed.append(float(np.sum(np.sign(tvals))))

    feats["Ag_centers"] = float(centers)
    feats["Ag_CN_mean"] = float(np.mean([len(adj[l]) for l,e in ele.items() if e==metal])) if centers>0 else 0.0
    feats["Ag_tau4_mean"] = float(np.mean(agg_tau4)) if agg_tau4 else 0.0
    feats["Ag_tau5_mean"] = float(np.mean(agg_tau5)) if agg_tau5 else 0.0
    feats["Ag_oct_dev90"] = float(np.mean(dev90)) if dev90 else 0.0
    feats["Ag_oct_dev180"] = float(np.mean(dev180)) if dev180 else 0.0
    feats["Ag_chiral_abs"] = float(np.mean(chir_abs)) if chir_abs else 0.0
    feats["Ag_chiral_signed"] = float(np.mean(chir_signed)) if chir_signed else 0.0
    return feats

# ---------------- 主流程 ----------------
def main():
    root = Path("./1513033"); out = Path("./out"); out.mkdir(parents=True, exist_ok=True)
    p_atoms  = find_one(root, "Atoms")
    p_bonds  = find_one(root, "Bonds")
    p_angles = find_one(root, "AllAngles")
    p_tors   = find_one(root, "AllTorsions")

    A = pd.read_csv(p_atoms)  if p_atoms  else None
    B = pd.read_csv(p_bonds)  if p_bonds  else None
    G = pd.read_csv(p_angles) if p_angles else None
    T = pd.read_csv(p_tors)   if p_tors   else None

    feats = {"id": ID_STR}

    # 元素计数（同原脚本，略）
    if A is not None:
        el_col = next((c for c in A.columns if re.search(r'(Label|Element|Symbol|Type)$', c, re.I)), None)
        if el_col:
            els = A[el_col].astype(str).str.replace(r'(\d+)$','', regex=True).str.strip()
            for el in EL_SET:
                feats[f"n_{el}"] = int((els==el).sum())
            feats["n_atoms"] = int(len(els)); feats["n_heavy_atoms"] = int((els!="H").sum())

    # 读取坐标
    AC = read_atoms_with_coords(p_atoms)
    # -------- 3D：回转张量 + 库仑矩阵特征值 + RDF --------
    if AC is not None and len(AC)>0:
        feats.update({f"gyr_{k}": v for k,v in gyration_features(AC).items()})
        eigs = coulomb_eigs(AC, K=COULOMB_EIG_K)
        for i,v in enumerate(eigs, 1):
            feats[f"cm_eig_{i}"] = v
        for (a,b) in RDF_PAIRS:
            hist = rdf_hist(AC, (a,b), bins=RDF_BINS)
            for i,h in enumerate(hist, 1):
                feats[f"rdf_{a}_{b}_bin{i}"] = h
    else:
        # 占位
        feats.update({f"gyr_{k}":0.0 for k in ["Rg","asphericity","acylindricity","kappa2"]})
        for i in range(COULOMB_EIG_K): feats[f"cm_eig_{i+1}"]=0.0
        for (a,b) in RDF_PAIRS:
            for i in range(len(RDF_BINS)-1): feats[f"rdf_{a}_{b}_bin{i+1}"]=0

    # -------- 金属中心 3D 几何摘要 --------
    if AC is not None and B is not None:
        feats.update(metal_3d_features(AC, B, metal="Ag"))
    else:
        feats.update({k:0.0 for k in ["Ag_centers","Ag_CN_mean","Ag_tau4_mean","Ag_tau5_mean","Ag_oct_dev90","Ag_oct_dev180","Ag_chiral_abs","Ag_chiral_signed"]})

    # -------- 原有 2D/角/扭转摘要（保留，维度很小） --------
    # 键长全局 & 精选元素对
    if B is not None:
        lc = find_numeric_col(B, [r'length', r'distance'])
        if lc:
            L = robust_numeric(B[lc]); feats.update({f"bond_len_{k}": v for k,v in basic_stats(L).items()})
            e1 = B["Atom1"].astype(str).apply(lambda s: parse_label(s)[0]).values if "Atom1" in B.columns else None
            e2 = B["Atom2"].astype(str).apply(lambda s: parse_label(s)[0]).values if "Atom2" in B.columns else None
            if e1 is not None and e2 is not None:
                L = L.values
                for (a,b) in PAIR_FOCUS:
                    mask = [(tuple(sorted((e1[i],e2[i])))==(a,b)) and np.isfinite(L[i]) for i in range(len(B))]
                    vals = pd.Series([L[i] for i,m in enumerate(mask) if m])
                    s = basic_stats(vals)
                    feats[f"pair_{a}_{b}_count"] = int(s["n"])
                    for k in ["mean","std","min","p10","p50","p90","max"]:
                        feats[f"pair_{a}_{b}_len_{k}"] = float(s[k])
        else:
            feats.update({f"bond_len_{k}":0.0 for k in ["mean","std","min","p10","p50","p90","max","n"]})

    # 角
    if G is not None:
        ang_col = find_numeric_col(G, [r'angle', r'theta', r'deg'])
        if ang_col:
            st = circular_stats(G[ang_col]); 
            for k,v in st.items(): feats[f"angle_global_{k}"]=v
            vv = pd.to_numeric(G[ang_col], errors="coerce").dropna().values
            hist,_ = np.histogram(vv, bins=[0,60,120,180])
            for i,h in enumerate(hist): feats[f"angle_hist_bin{i}"]=int(h)
        a1 = next((c for c in G.columns if re.fullmatch(r'(Atom1|A|i|Idx1|Index1)', c, flags=re.I)), None)
        a2 = next((c for c in G.columns if re.fullmatch(r'(Atom2|B|j|Idx2|Index2)', c, flags=re.I)), None)
        if a1 and a2 and ang_col:
            center_el = G[a2].astype(str).apply(lambda s: parse_label(s)[0])
            vv = pd.to_numeric(G[ang_col], errors="coerce")
            for el in CENTER_EL_ANGLE:
                st = circular_stats(vv[center_el==el])
                feats[f"angle_center_{el}_mean"]=st["mean"]; feats[f"angle_center_{el}_std"]=st["std"]
                feats[f"angle_center_{el}_r"]=st["r"]; feats[f"angle_center_{el}_n"]=st["n"]

    # 扭转
    if T is not None:
        tor_col = find_numeric_col(T, [r'torsion', r'dihedral', r'phi'])
        if tor_col:
            st = circular_stats(T[tor_col])
            for k in ["mean","std","r","sin_mean","cos_mean","n"]:
                feats[f"torsion_global_{k}"]=st[k]
        t2 = next((c for c in T.columns if re.fullmatch(r'(Atom2|B)', c, flags=re.I)), None)
        t3 = next((c for c in T.columns if re.fullmatch(r'(Atom3|C)', c, flags=re.I)), None)
        if t2 and t3 and tor_col:
            Bc = T[t2].astype(str).apply(lambda s: parse_label(s)[0])
            Cc = T[t3].astype(str).apply(lambda s: parse_label(s)[0])
            vv = pd.to_numeric(T[tor_col], errors="coerce")
            for (eB,eC) in CENTER_PAIR_TORS:
                mask = (Bc==eB)&(Cc==eC) | (Bc==eC)&(Cc==eB)
                st = circular_stats(vv[mask])
                feats[f"torsion_center_{eB}_{eC}_r"]=st["r"]; feats[f"torsion_center_{eB}_{eC}_n"]=st["n"]

    # ---- 写出 ----
    X = pd.DataFrame([feats]).fillna(0.0)
    out_path = out / "X_compact3D_1513033.csv"
    X.to_csv(out_path, index=False)
    print(f"[DONE] {out_path}  shape={X.shape}")

if __name__ == "__main__":
    main()
