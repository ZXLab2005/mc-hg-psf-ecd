#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cif2xyz_ase.py —— 批量将 cod/ 文件夹下的所有 CIF 转换为 XYZ
依赖：ASE (pip install ase)
"""

import glob
from pathlib import Path
from ase.io import read, write

# 输入文件夹
in_root = Path("COD")
# 输出文件夹
out_root = Path("xyz_out")

def main():
    cif_files = glob.glob(str(in_root / "**" / "*.cif"), recursive=True)
    if not cif_files:
        print("[INFO] cod/ 文件夹下未找到任何 .cif 文件")
        return

    for cif in cif_files:
        cif_path = Path(cif)
        rel = cif_path.relative_to(in_root)
        out_path = out_root / rel.with_suffix(".xyz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            at = read(cif)        # ASE 自动处理分数坐标 -> 笛卡尔坐标
            write(out_path, at, "xyz")
            print(f"[OK] {cif_path} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {cif_path}: {e}")

if __name__ == "__main__":
    main()
