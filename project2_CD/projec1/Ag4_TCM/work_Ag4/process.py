#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_y_highres.py
把 (E, intensity) 光谱文件批量转换为高分辨率 TabPFN 标签：Y_highres.csv + energy_grid.txt

支持：
- 单文件：--input file.dat
- 批量目录：--indir raw_dir  (自动收集 .dat / .csv)
- 自动判别能量单位（eV 或 nm），也可手动 --unit {auto,eV,nm}
- 2–8 eV 网格，默认步长 0.02 eV（共 301 点），可用 --step 覆盖
- 允许 .dat 的注释行以 '#' 开头
- .csv/.dat 都需至少两列：(E, intensity)

输出：
- energy_grid.txt
- Y_highres.csv  (行=id=文件名、列=E_*.***eV)
- 每个样本的裁剪后原始数据：<id>_clipped.csv (E_eV,intensity)

用法：
  python process.py --input "rot_spec-COM-length.dat" --outdir ./out
  python make_y_highres.py --indir ./raw_spectra --outdir ./out
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

def to_eV(E, unit="auto"):
    if unit == "eV":
        return E
    if unit == "nm":
        return 1240.0 / E
    # auto 判别
    med = float(np.median(E))
    if 20 < med < 2000:
        return 1240.0 / E
    return E

def load_xy(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        # 取前两列为 E, intensity
        if df.shape[1] < 2:
            raise ValueError(f"{p} 至少需要两列")
        E = pd.to_numeric(df.iloc[:,0], errors="coerce").values
        Y = pd.to_numeric(df.iloc[:,1], errors="coerce").values
        m = ~(np.isnan(E) | np.isnan(Y))
        return E[m], Y[m]
    else:
        # .dat or others -> 尝试 numpy.loadtxt，忽略 # 注释
        arr = np.loadtxt(p, comments="#")
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"{p} 不是两列数值的光谱文件")
        return arr[:,0], arr[:,1]

def process(files, outdir, emin=2.0, emax=8.0, step=0.02, unit="auto"):
    outdir.mkdir(parents=True, exist_ok=True)
    E_grid = np.arange(emin, emax + 1e-12, step)
    np.savetxt(outdir / "energy_grid.txt", E_grid, fmt="%.6f")
    rows = []
    for f in files:
        try:
            E, Y = load_xy(f)
            E = to_eV(E, unit=unit)
            # 裁剪、排序、去重
            m = (E >= emin) & (E <= emax)
            E, Y = E[m], Y[m]
            if E.size < 2:
                Y_grid = np.zeros_like(E_grid)
                E_u, Y_u = np.array([]), np.array([])
            else:
                o = np.argsort(E)
                E, Y = E[o], Y[o]
                E_u, idx = np.unique(E, return_index=True)
                Y_u = Y[idx]
                Y_grid = np.interp(E_grid, E_u, Y_u, left=0.0, right=0.0)
            rid = Path(f).stem
            rows.append((rid, Y_grid))
            pd.DataFrame({"E_eV":E_u, "intensity":Y_u}).to_csv(outdir / f"{rid}_clipped.csv", index=False)
        except Exception as e:
            print(f"[WARN] 跳过 {f}: {e}", file=sys.stderr)
    # 汇总为 Y_highres.csv
    ycols = [f"E_{e:0.3f}eV" for e in E_grid]
    Y_df = pd.DataFrame({rid: y for rid, y in rows}).T
    Y_df.columns = ycols
    Y_df.index.name = "id"
    Y_df.to_csv(outdir / "Y_highres.csv")
    print(f"完成：{outdir}/Y_highres.csv 形状 {Y_df.shape}；能量网格 {len(E_grid)} 点")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="单个光谱文件路径 (.dat/.csv)")
    ap.add_argument("--indir", type=str, help="包含多条光谱的目录")
    ap.add_argument("--outdir", type=str, default="./out_hr")
    ap.add_argument("--unit", type=str, default="auto", choices=["auto","eV","nm"])
    ap.add_argument("--emin", type=float, default=2.0)
    ap.add_argument("--emax", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.02)
    args = ap.parse_args()

    if args.input:
        files = [args.input]
    elif args.indir:
        d = Path(args.indir)
        files = sorted([str(p) for p in list(d.glob("*.dat")) + list(d.glob("*.csv"))])
        if not files:
            print("目录下没有 .dat / .csv 文件", file=sys.stderr); sys.exit(1)
    else:
        print("请提供 --input 或 --indir", file=sys.stderr); sys.exit(1)

    process(files, Path(args.outdir), emin=args.emin, emax=args.emax, step=args.step, unit=args.unit)

if __name__ == "__main__":
    main()
