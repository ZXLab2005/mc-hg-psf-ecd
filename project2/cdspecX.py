#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_itemwise_X_single.py

读取 ./1513033/ 目录下的四个表：
  - Bonds.csv（或 *Bonds*.csv）
  - AllAngles.csv（或 *AllAngles*.csv）
  - AllTorsions.csv（或 *AllTorsions*.csv）
  - Atoms.csv（或 *Atoms*.csv，可选）

将“每一条 键长/键角/扭转角”按元素模式逐项展开为固定列：
  - 键：元素对 (A,B) -> 为每个对分配 k_bond 个槽位，填入升序的键长；不足 0 填充；另给 *_count
  - 角：元素三元 (A,B,C) -> 分配 k_angle 个槽位，填入升序角度；另给 *_count
  - 扭转：元素四元 (A,B,C,D) -> 分配 k_torsion 个槽位，填入升序扭转角，
           同时写入对应的 sin/cos；另给 *_count

输出：
  ./out/X_itemwise_1513033.csv    （一行：id=1513033）
  ./out/itemwise_schema_1513033.json  （记录本样本出现的模式，便于复现）

用法：
  python build_itemwise_X_single.py \
     --root ./1513033 --out ./out \
     --k_bond 5 --k_angle 5 --k_torsion 6
"""

import argparse, re, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import pandas as pd

# --------- 小工具 ---------
def robust_numeric(s: pd.Series) -> pd.Series:
    """从 '1.41(3)' 或 '2.0±0.1' 抽取首个浮点数；失败返回 NaN。"""
    s = s.astype(str)
    num = s.str.extract(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', expand=False)
    return pd.to_numeric(num, errors="coerce")

def parse_label(lbl: str):
    """'Ag12' -> ('Ag', 12)；失败 -> ('X', None)。"""
    m = re.match(r'([A-Za-z]+)(\d+)$', str(lbl).strip())
    return (m.group(1), int(m.group(2))) if m else ("X", None)

def find_numeric_col(df: pd.DataFrame, patterns):
    """按列名模式找一个数值列；否则回退到第一列可转数值的列。"""
    for c in df.columns:
        if any(re.search(pat, c, re.I) for pat in patterns):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or robust_numeric(df[c]).notna().any():
            return c
    return None

def fill_slots(sorted_values, K):
    """取前 K 个，不足补 0，并返回 (值列表, 实际数量)。"""
    vals = list(sorted_values[:K])
    used = min(len(sorted_values), K)
    if len(vals) < K:
        vals += [0.0]*(K - len(vals))
    return vals, used

def find_one(root: Path, key: str):
    """
    在 root 下寻找 key 对应的文件：
      key='Bonds' -> 先找 'Bonds.csv'，否则找 '*Bonds*.csv'。
    """
    exact = root / f"{key}.csv"
    if exact.exists():
        return exact
    cands = sorted(root.glob(f"*{key}*.csv"))
    return cands[0] if cands else None

# --------- 主逻辑（单样本） ---------
def build_single(root: Path, out_dir: Path, k_bond=5, k_angle=5, k_torsion=6, sample_id="1513033"):
    out_dir.mkdir(parents=True, exist_ok=True)

    p_atoms = find_one(root, "Atoms")          # 可选
    p_bonds = find_one(root, "Bonds")          # 必需（若没有则该块为空）
    p_angles = find_one(root, "AllAngles")     # 可选
    p_tors = find_one(root, "AllTorsions")     # 可选

    row = {"id": sample_id}
    schema = {"pairs": [], "triples": [], "quads": []}

    # -------- Bonds：元素对 (A,B) 的逐项槽位（键长） --------
    if p_bonds is not None and p_bonds.exists():
        B = pd.read_csv(p_bonds)
        len_col = find_numeric_col(B, [r'length', r'distance'])
        L = robust_numeric(B[len_col]) if len_col else pd.Series([], dtype=float)

        pair2vals = defaultdict(list)
        if {"Atom1","Atom2"}.issubset(B.columns) and len(L)==len(B):
            for (_, r), val in zip(B.iterrows(), L):
                e1,_ = parse_label(r["Atom1"]); e2,_ = parse_label(r["Atom2"])
                if np.isfinite(val):
                    pair2vals[tuple(sorted((e1,e2)))].append(float(val))

        # 本样本实际出现过的元素对（按频次从高到低）
        pairs_sorted = [p for p,_ in Counter({k:len(v) for k,v in pair2vals.items()}).most_common()]
        schema["pairs"] = [list(p) for p in pairs_sorted]

        for a,b in pairs_sorted:
            vals = sorted(pair2vals.get((a,b), []))
            slots, used = fill_slots(vals, k_bond)
            for j,v in enumerate(slots, 1):
                row[f"bond_{a}_{b}_len_{j}"] = v
            row[f"bond_{a}_{b}_count"] = int(used)

    # -------- Angles：元素三元 (A,B,C) 的逐项槽位（角度，度） --------
    if p_angles is not None and p_angles.exists():
        G = pd.read_csv(p_angles)
        ang_col = find_numeric_col(G, [r'angle', r'theta', r'deg'])
        avals = robust_numeric(G[ang_col]) if ang_col else pd.Series([], dtype=float)
        a1 = next((c for c in G.columns if re.fullmatch(r'(Atom1|A|i|Idx1|Index1)', c, flags=re.I)), None)
        a2 = next((c for c in G.columns if re.fullmatch(r'(Atom2|B|j|Idx2|Index2)', c, flags=re.I)), None)
        a3 = next((c for c in G.columns if re.fullmatch(r'(Atom3|C|k|Idx3|Index3)', c, flags=re.I)), None)

        triple2vals = defaultdict(list)
        if a1 and a2 and a3 and ang_col and len(avals)==len(G):
            for (_, r), val in zip(G.iterrows(), avals):
                A = parse_label(r[a1])[0]; Bc = parse_label(r[a2])[0]; C = parse_label(r[a3])[0]
                if np.isfinite(val):
                    triple2vals[(A,Bc,C)].append(float(val))

        triples_sorted = [t for t,_ in Counter({k:len(v) for k,v in triple2vals.items()}).most_common()]
        schema["triples"] = [list(t) for t in triples_sorted]

        for a,b,c in triples_sorted:
            vals = sorted(triple2vals.get((a,b,c), []))
            slots, used = fill_slots(vals, k_angle)
            for j,v in enumerate(slots, 1):
                row[f"angle_{a}_{b}_{c}_deg_{j}"] = v
            row[f"angle_{a}_{b}_{c}_count"] = int(used)

    # -------- Torsions：元素四元 (A,B,C,D) 逐项槽位（角度 + sin/cos） --------
    if p_tors is not None and p_tors.exists():
        T = pd.read_csv(p_tors)
        tor_col = find_numeric_col(T, [r'torsion', r'dihedral', r'phi'])
        tvals = robust_numeric(T[tor_col]) if tor_col else pd.Series([], dtype=float)
        t1 = next((c for c in T.columns if re.fullmatch(r'(Atom1|A)', c, flags=re.I)), None)
        t2 = next((c for c in T.columns if re.fullmatch(r'(Atom2|B)', c, flags=re.I)), None)
        t3 = next((c for c in T.columns if re.fullmatch(r'(Atom3|C)', c, flags=re.I)), None)
        t4 = next((c for c in T.columns if re.fullmatch(r'(Atom4|D)', c, flags=re.I)), None)

        quad2vals = defaultdict(list)
        if t1 and t2 and t3 and t4 and tor_col and len(tvals)==len(T):
            for (_, r), val in zip(T.iterrows(), tvals):
                A = parse_label(r[t1])[0]; Bc = parse_label(r[t2])[0]
                C = parse_label(r[t3])[0]; D  = parse_label(r[t4])[0]
                if np.isfinite(val):
                    quad2vals[(A,Bc,C,D)].append(float(val))

        quads_sorted = [q for q,_ in Counter({k:len(v) for k,v in quad2vals.items()}).most_common()]
        schema["quads"] = [list(q) for q in quads_sorted]

        for a,b,c,d in quads_sorted:
            vals = sorted(quad2vals.get((a,b,c,d), []))
            slots, used = fill_slots(vals, k_torsion)
            for j,v in enumerate(slots, 1):
                row[f"torsion_{a}_{b}_{c}_{d}_deg_{j}"] = v
                if v == 0.0:
                    row[f"torsion_{a}_{b}_{c}_{d}_sin_{j}"] = 0.0
                    row[f"torsion_{a}_{b}_{c}_{d}_cos_{j}"] = 1.0
                else:
                    rad = np.deg2rad(v)
                    row[f"torsion_{a}_{b}_{c}_{d}_sin_{j}"] = float(np.sin(rad))
                    row[f"torsion_{a}_{b}_{c}_{d}_cos_{j}"] = float(np.cos(rad))
            row[f"torsion_{a}_{b}_{c}_{d}_count"] = int(used)

    # ---- 可选：Atoms 做个简要元素计数（不改变“逐项展开”的主体设计） ----
    if p_atoms is not None and p_atoms.exists():
        A = pd.read_csv(p_atoms)
        el_col = None
        # 常见列名：Label（如 Ag12）、Element、Symbol...
        for c in A.columns:
            if re.search(r'(Label|Element|Symbol|Type)$', c, re.I):
                el_col = c; break
        if el_col is not None:
            els = A[el_col].astype(str).str.replace(r'(\d+)$','', regex=True).str.strip()
            cnt = els.value_counts()
            for e, n in cnt.items():
                row[f"atoms_n_{e}"] = int(n)
            row["atoms_n"] = int(len(els))
            row["atoms_n_heavy"] = int((els!="H").sum())

    # ---- 保存 ----
    df = pd.DataFrame([row]).fillna(0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "X_itemwise_1513033.csv", index=False)

    # 记录本样本出现过的模式，便于复现
    (out_dir / "itemwise_schema_1513033.json").write_text(
        json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[DONE] {out_dir/'X_itemwise_1513033.csv'}  shape={df.shape}")
    print(f"[INFO] 模式记录：{out_dir/'itemwise_schema_1513033.json'}")

# --------- CLI ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./1513033", help="包含四张表的目录")
    ap.add_argument("--out",  type=str, default="./out", help="输出目录")
    ap.add_argument("--k_bond",     type=int, default=5, help="每个元素对的键长槽位数")
    ap.add_argument("--k_angle",    type=int, default=5, help="每个元素三元的角度槽位数")
    ap.add_argument("--k_torsion",  type=int, default=6, help="每个元素四元的扭转槽位数")
    ap.add_argument("--id",         type=str, default="1513033", help="该样本的 id（写入输出表）")
    args = ap.parse_args()

    build_single(Path(args.root), Path(args.out),
                 k_bond=args.k_bond, k_angle=args.k_angle, k_torsion=args.k_torsion,
                 sample_id=args.id)

if __name__ == "__main__":
    main()
