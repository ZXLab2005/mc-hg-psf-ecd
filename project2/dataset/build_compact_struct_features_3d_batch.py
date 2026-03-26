#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_compact_struct_features_3d_batch_xyz.py
批处理 ATOM_BOND/<id>/ 四表，若 Atoms.csv 无可用 XYZ 坐标，则回退读取同目录下的 *.xyz。
输出一张总表: ./out/X_compact3D.csv
"""

import re, argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# ---------------- 配置（与你现有一致） ----------------
EL_SET = ["H","C","N","O","Ag"]
PAIR_FOCUS = [("Ag","N"),("Ag","O"),("Ag","C"),
              ("C","C"),("C","N"),("C","O"),("N","O"),
              ("C","H"),("N","H"),("O","H")]
CENTER_EL_ANGLE = ["C","N","O","Ag"]
CENTER_PAIR_TORS = [("C","C"),("C","N"),("C","O"),("N","O"),
                    ("Ag","N"),("Ag","O")]
COULOMB_EIG_K = 16
RDF_PAIRS = [("Ag","N"),("Ag","O"),("C","C"),("C","N"),("C","O")]
RDF_BINS = np.arange(2.0, 10.0 + 1e-9, 1.0)  # 2–10 Å, 步长 1 Å
ZMAP = {"H":1, "C":6, "N":7, "O":8, "Ag":47}

# ---------------- 小工具 ----------------
def robust_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    # 允许 D 科学计数与逗号
    s = s.str.replace('D', 'E', regex=False).str.replace('d', 'E', regex=False).str.replace(',', '', regex=False)
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

# ========= 新增：坐标强制数值化 & 字符串转浮点（支持 D 科学计数/逗号） =========
def _to_float(v):
    if isinstance(v, str):
        v = v.replace('D','E').replace('d','E').replace(',', '')
    return float(v)

def _ensure_numeric_xyz(df: pd.DataFrame) -> pd.DataFrame:
    """把 x/y/z 列强制转 float，无法解析的置 NaN 后丢弃整行。"""
    if df is None or df.empty:
        return df
    for c in ("x","y","z"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace('D','E').str.replace('d','E').str.replace(',', ''), errors="coerce")
    df = df.dropna(subset=["x","y","z"]).copy()
    df[["x","y","z"]] = df[["x","y","z"]].astype(float)
    return df

# -------- 读取 Atoms.csv 的坐标（有则优先用） --------
def read_atoms_xyz_from_csv(p_atoms: Path):
    if p_atoms is None or not p_atoms.exists(): return None
    A = pd.read_csv(p_atoms)
    el_col = next((c for c in A.columns if re.search(r'(Label|Element|Symbol|Type)$', c, re.I)), None)

    def _find_xyz(cols):
        lower = {c.lower(): c for c in cols}
        for cand in [("x","y","z"), ("cartn_x","cartn_y","cartn_z"), ("fract_x","fract_y","fract_z")]:
            if all(k in lower for k in cand):
                return lower[cand[0]], lower[cand[1]], lower[cand[2]], ("fract" in cand[0])
        # 宽松匹配：含独立 x/y/z
        xs = [c for c in cols if re.search(r'\bx\b', c, re.I)]
        ys = [c for c in cols if re.search(r'\by\b', c, re.I)]
        zs = [c for c in cols if re.search(r'\bz\b', c, re.I)]
        if xs and ys and zs: return xs[0], ys[0], zs[0], False
        return None, None, None, False

    if el_col is None: return None
    xcol,ycol,zcol,is_frac = _find_xyz(list(A.columns))
    if xcol is None: return None

    lab = A[el_col].astype(str).str.strip()
    el = lab.str.replace(r'(\d+)$','', regex=True).map(
        lambda s: re.match(r'[A-Za-z]+', s).group(0) if re.match(r'[A-Za-z]+', s) else "X")
    X = robust_numeric(A[xcol]); Y = robust_numeric(A[ycol]); Z = robust_numeric(A[zcol])
    df = pd.DataFrame({"label": lab, "el": el, "x": X, "y": Y, "z": Z}).dropna()

    df = _ensure_numeric_xyz(df)
    return df if df is not None and len(df) else None

# -------- 读取 .xyz（回退方案：标准 xyz，含元素符号） --------
def read_atoms_from_xyz(sample_dir: Path):
    files = sorted(sample_dir.glob("*.xyz"))
    if not files: return None
    p = files[0]
    lines = p.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    rows = []
    # 尝试标准 XYZ（第一行是原子数）
    try:
        n = int(lines[0].strip()); start = 2
        for i in range(start, start+n):
            parts = lines[i].split()
            if len(parts) < 4: continue
            el, x, y, z = parts[0], _to_float(parts[1]), _to_float(parts[2]), _to_float(parts[3])
            rows.append((el, x, y, z))
    except Exception:
        # 宽松解析：逐行提取 el x y z
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 4 and re.match(r'^[A-Za-z]+$', parts[0]):
                try:
                    rows.append((parts[0], _to_float(parts[-3]), _to_float(parts[-2]), _to_float(parts[-1])))
                except:
                    pass
    if not rows:
        return None
    # 生成 label 如 C1, H2...
    ctr = Counter(); lab=[]
    for el,x,y,z in rows:
        ctr[el]+=1; lab.append(f"{el}{ctr[el]}")
    df = pd.DataFrame(rows, columns=["el","x","y","z"])
    df.insert(0, "label", lab)
    df = _ensure_numeric_xyz(df)
    return df if df is not None and len(df) else None

# ===================== 仅坐标的 *.xyz（无元素） =====================
def read_xyz_coords_only(sample_dir: Path):
    files = sorted(sample_dir.glob("*.xyz"))
    if not files: return None
    p = files[0]
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    coords = []
    # 尝试标准 XYZ
    try:
        n = int(lines[0].strip()); start = 2
        for i in range(start, min(start+n, len(lines))):
            toks = lines[i].split()
            if len(toks) >= 4 and re.match(r'^[A-Za-z]+$', toks[0]):
                x,y,z = _to_float(toks[1]), _to_float(toks[2]), _to_float(toks[3])
                coords.append((x,y,z))
            elif len(toks) >= 3:
                x,y,z = _to_float(toks[-3]), _to_float(toks[-2]), _to_float(toks[-1])
                coords.append((x,y,z))
    except Exception:
        # 宽松：直接取每行最后三列数字
        for ln in lines:
            toks = ln.split()
            if len(toks) >= 3:
                try:
                    x,y,z = _to_float(toks[-3]), _to_float(toks[-2]), _to_float(toks[-1])
                    coords.append((x,y,z))
                except:
                    pass
    return coords if coords else None

# ===================== 与 Atoms 元素/标签按行对齐融合 =====================
def fuse_atoms_and_xyz(p_atoms: Path, coords):
    """coords 可以是 [(x,y,z), ...] 或 DataFrame 带 x/y/z 列；按行覆盖 label/el。"""
    if p_atoms is None or not p_atoms.exists() or coords is None:
        return None
    if isinstance(coords, pd.DataFrame):
        coords = coords[["x","y","z"]].to_numpy().tolist()
    A = pd.read_csv(p_atoms)
    el_col = next((c for c in A.columns if re.search(r'(Label|Element|Symbol|Type)$', c, re.I)), None)
    if el_col is None:
        return None
    lab = A[el_col].astype(str).str.strip()
    el  = lab.str.replace(r'(\d+)$','', regex=True).map(
        lambda s: re.match(r'[A-Za-z]+', s).group(0) if re.match(r'[A-Za-z]+', s) else "X")
    if len(coords) != len(lab):
        print(f"[WARN] XYZ 坐标数({len(coords)})与 Atoms 行数({len(lab)})不一致，跳过对齐")
        return None
    X = [_to_float(c[0]) for c in coords]; Y = [_to_float(c[1]) for c in coords]; Z = [_to_float(c[2]) for c in coords]
    df = pd.DataFrame({"label": lab.values, "el": el.values, "x": X, "y": Y, "z": Z})
    df = _ensure_numeric_xyz(df)
    return df if df is not None and len(df) else None

# -------- 统一接口：优先 Atoms.csv，其次 *.xyz；在行数一致时**强制用 Atoms 的 label/el 对齐** --------
def read_coords_any(sample_dir: Path):
    p_atoms = find_one(sample_dir, "Atoms")
    # 1) CSV 优先
    df = read_atoms_xyz_from_csv(p_atoms)
    if df is not None and len(df):
        print(f"[XYZ] {sample_dir.name}: 使用 Atoms.csv 坐标，原子数={len(df)}")
        return df
    # 2) 含元素的 xyz → 若与 Atoms 行数一致，强制用 Atoms 的 label/el 覆盖（逐行一一对齐）
    df = read_atoms_from_xyz(sample_dir)
    if df is not None and len(df):
        fused = fuse_atoms_and_xyz(p_atoms, df)  # ★ 关键改动：即使 xyz 含元素也对齐
        if fused is not None:
            print(f"[XYZ] {sample_dir.name}: 使用 *.xyz 回退（已按 Atoms 标签/元素对齐），原子数={len(fused)}")
            return fused
        print(f"[XYZ] {sample_dir.name}: 使用 *.xyz 回退（无法与 Atoms 对齐），原子数={len(df)}")
        return df
    # 3) 仅坐标的 xyz + Atoms 对齐
    coords = read_xyz_coords_only(sample_dir)
    if coords is not None:
        fused = fuse_atoms_and_xyz(p_atoms, coords)
        if fused is not None:
            print(f"[XYZ] {sample_dir.name}: 使用 *.xyz 坐标 + Atoms 元素对齐，原子数={len(fused)}")
            return fused
    print(f"[WARN] {sample_dir.name}: 未找到可用坐标，3D 特征将为 0")
    return None

# -------- 3D 特征 --------
def gyration_features(coords_df):
    R = coords_df[["x","y","z"]].to_numpy(dtype=float)
    if len(R)==0:
        return {"Rg":0.0,"asphericity":0.0,"acylindricity":0.0,"kappa2":0.0}
    R = R - R.mean(axis=0, keepdims=True)
    G = (R.T @ R) / len(R)
    w = np.linalg.eigvalsh(G)
    g1,g2,g3 = float(w[0]), float(w[1]), float(w[2])
    Rg = float(np.sqrt(g1+g2+g3))
    asph = g3 - 0.5*(g2+g1)
    acyl = g2 - g1
    k2 = 1.0 - 3.0*(g1*g2 + g2*g3 + g3*g1) / ((g1+g2+g3)**2 + 1e-12)
    return {"Rg":Rg, "asphericity":float(asph), "acylindricity":float(acyl), "kappa2":float(k2)}

def coulomb_eigs(coords_df, K=16):
    if coords_df is None or len(coords_df)==0: return [0.0]*K
    xyz = coords_df[["x","y","z"]].to_numpy(dtype=float)
    els = coords_df["el"].astype(str).map(ZMAP).fillna(0).to_numpy(dtype=float)
    unknown = set(coords_df["el"].astype(str).unique()) - set(ZMAP.keys())
    if unknown: print(f"[WARN] 未在 ZMAP 中的元素将按 0 处理: {sorted(unknown)}")
    n = len(coords_df)
    M = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i==j: M[i,i] = 0.5 * (els[i] ** 2.4)
            else:
                rij = np.linalg.norm(xyz[i]-xyz[j]) + 1e-12
                M[i,j] = els[i]*els[j]/rij
    w = np.linalg.eigvalsh(M)
    w = np.sort(w)[::-1]
    if len(w) < K: w = np.pad(w, (0, K-len(w)))
    return [float(v) for v in w[:K]]

def rdf_hist(coords_df, pair=("Ag","N"), bins=RDF_BINS):
    if coords_df is None: return [0]*(len(bins)-1)
    A = coords_df[coords_df["el"]==pair[0]]
    B = coords_df[coords_df["el"]==pair[1]]
    if len(A)==0 or len(B)==0: return [0]*(len(bins)-1)
    dists = []
    for _, ra in A[["x","y","z"]].iterrows():
        va = ra.values.astype(float)
        vb = B[["x","y","z"]].to_numpy(dtype=float)
        dists.extend(np.linalg.norm(vb - va, axis=1).tolist())
    hist, _ = np.histogram(np.array(dists, dtype=float), bins=bins)
    return [int(v) for v in hist.tolist()]

def metal_3d_features(coords_df, bonds_df, metal="Ag"):
    feats = {}
    if coords_df is None or bonds_df is None or "Atom1" not in bonds_df.columns:
        keys = ["Ag_centers","Ag_CN_mean","Ag_tau4_mean","Ag_tau5_mean",
                "Ag_oct_dev90","Ag_oct_dev180","Ag_chiral_abs","Ag_chiral_signed"]
        return {k:0.0 for k in keys}
    pos = {r["label"]: r[["x","y","z"]].to_numpy(dtype=float) for _,r in coords_df.iterrows()}
    ele = {r["label"]: r["el"] for _,r in coords_df.iterrows()}
    adj = defaultdict(list)
    for _,r in bonds_df.iterrows():
        u,v = str(r["Atom1"]).strip(), str(r["Atom2"]).strip()
        adj[u].append(v); adj[v].append(u)

    agg_tau4, agg_tau5, dev90, dev180, chir_abs, chir_signed, centers = [], [], [], [], [], [], 0
    for lbl, e in ele.items():
        if e != metal or lbl not in pos: continue
        nbrs = [n for n in adj.get(lbl, []) if n in pos]
        CN = len(nbrs)
        if CN==0: continue
        centers += 1
        vecs = [pos[n]-pos[lbl] for n in nbrs]
        angles = []
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                vi, vj = vecs[i], vecs[j]
                ni = np.linalg.norm(vi)+1e-12; nj = np.linalg.norm(vj)+1e-12
                ang = np.degrees(np.arccos(np.clip(np.dot(vi,vj)/(ni*nj), -1.0, 1.0)))
                angles.append(ang)
        if CN==4 and len(angles)>=2:
            alpha, beta = sorted(angles, reverse=True)[:2]
            agg_tau4.append((360.0 - (alpha + beta)) / 141.0)
        if CN==5 and len(angles)>=2:
            alpha, beta = sorted(angles, reverse=True)[:2]
            agg_tau5.append((alpha - beta) / 60.0)
        if CN>=6 and len(angles)>0:
            dev90.append(float(np.mean(np.abs(np.array(angles, dtype=float)-90.0))))
            dev180.append(float(np.min(np.abs(np.array(angles, dtype=float)-180.0))))
        if CN>=3:
            tvals=[]
            m=len(vecs)
            for i in range(m):
                for j in range(i+1,m):
                    for k in range(j+1,m):
                        a,b,c = vecs[i], vecs[j], vecs[k]
                        vol = float(np.dot(a, np.cross(b,c)))
                        denom = (np.linalg.norm(a)*np.linalg.norm(b)*np.linalg.norm(c) + 1e-12)
                        tvals.append(vol/denom)
            if tvals:
                chir_abs.append(float(np.mean(np.abs(tvals))))
                chir_signed.append(float(np.sum(np.sign(tvals))))
    return {
        "Ag_centers": float(centers),
        "Ag_CN_mean": float(np.mean([len(adj[l]) for l,e in ele.items() if e==metal])) if centers>0 else 0.0,
        "Ag_tau4_mean": float(np.mean(agg_tau4)) if agg_tau4 else 0.0,
        "Ag_tau5_mean": float(np.mean(agg_tau5)) if agg_tau5 else 0.0,
        "Ag_oct_dev90": float(np.mean(dev90)) if dev90 else 0.0,
        "Ag_oct_dev180": float(np.mean(dev180)) if dev180 else 0.0,
        "Ag_chiral_abs": float(np.mean(chir_abs)) if chir_abs else 0.0,
        "Ag_chiral_signed": float(np.mean(chir_signed)) if chir_signed else 0.0,
    }

# -------- 单样本提取（其余不变） --------
def extract_one(sample_dir: Path, sample_id: str) -> dict:
    p_atoms  = find_one(sample_dir, "Atoms")
    p_bonds  = find_one(sample_dir, "Bonds")
    p_angles = find_one(sample_dir, "AllAngles")
    p_tors   = find_one(sample_dir, "AllTorsions")

    A = pd.read_csv(p_atoms)  if p_atoms  else None
    B = pd.read_csv(p_bonds)  if p_bonds  else None
    G = pd.read_csv(p_angles) if p_angles else None
    T = pd.read_csv(p_tors)   if p_tors   else None

    feats = {"id": str(sample_id)}

    # 组成
    if A is not None:
        el_col = next((c for c in A.columns if re.search(r'(Label|Element|Symbol|Type)$', c, re.I)), None)
        if el_col:
            els = A[el_col].astype(str).str.replace(r'(\d+)$','', regex=True).str.strip()
            for el in EL_SET: feats[f"n_{el}"] = int((els==el).sum())
            feats["n_atoms"] = int(len(els)); feats["n_heavy_atoms"] = int((els!="H").sum())

    # 坐标（CSV 优先，失败则回退 .xyz / 仅坐标 xyz + Atoms 融合）
    AC = read_coords_any(sample_dir)

    # 3D 特征
    if AC is not None and len(AC)>0:
        feats.update({f"gyr_{k}": v for k,v in gyration_features(AC).items()})
        eigs = coulomb_eigs(AC, K=COULOMB_EIG_K)
        for i,v in enumerate(eigs, 1): feats[f"cm_eig_{i}"] = v
        for (a,b) in RDF_PAIRS:
            hist = rdf_hist(AC, (a,b), bins=RDF_BINS)
            for i,h in enumerate(hist, 1): feats[f"rdf_{a}_{b}_bin{i}"] = h
        feats.update(metal_3d_features(AC, B, metal="Ag"))
    else:
        feats.update({f"gyr_{k}":0.0 for k in ["Rg","asphericity","acylindricity","kappa2"]})
        for i in range(COULOMB_EIG_K): feats[f"cm_eig_{i+1}"]=0.0
        for (a,b) in RDF_PAIRS:
            for i in range(len(RDF_BINS)-1): feats[f"rdf_{a}_{b}_bin{i+1}"]=0
        feats.update({k:0.0 for k in ["Ag_centers","Ag_CN_mean","Ag_tau4_mean","Ag_tau5_mean",
                                      "Ag_oct_dev90","Ag_oct_dev180","Ag_chiral_abs","Ag_chiral_signed"]})

    # 键长 & 重点配对
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
            st = circular_stats(G[ang_col])
            for k,v in st.items(): feats[f"angle_global_{k}"]=v
            vv = pd.to_numeric(G[ang_col], errors="coerce").dropna().values
            hist,_ = np.histogram(vv, bins=[0,60,120,180])
            for i,h in enumerate(hist): feats[f"angle_hist_bin{i}"]=int(h)
        a2 = next((c for c in G.columns if re.fullmatch(r'(Atom2|B|j|Idx2|Index2)', c, flags=re.I)), None)
        if a2 and ang_col:
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

    return feats

# ---------------- 主程序 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./ATOM_BOND")
    ap.add_argument("--out",  type=str, default="./out/X_compact3D.csv")
    args = ap.parse_args()

    root = Path(args.root); out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    subdirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not subdirs:
        raise SystemExit(f"在 {root} 下没有找到子文件夹")
    for d in subdirs:
        sid = d.name
        try:
            row = extract_one(d, sid)
            rows.append(row)
            print(f"[OK] {sid}: features={len(row)-1}")
        except Exception as e:
            print(f"[WARN] {sid} 处理失败：{e}")

    all_keys = set().union(*[r.keys() for r in rows])
    cols = ["id"] + sorted(k for k in all_keys if k!="id")
    df = pd.DataFrame(rows, columns=cols).fillna(0.0)
    df.to_csv(out_path, index=False)
    print(f"[DONE] 保存 {out_path}  形状={df.shape}")

if __name__ == "__main__":
    main()
