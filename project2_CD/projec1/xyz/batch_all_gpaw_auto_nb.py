#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_all_gpaw_auto_nb.py — 批量执行 GS→rt-TDDFT→ECD，并为每个分子**自适应 nbands**

关键修复：
  - 对 XYZ 分子：atoms.pbc=False 且 atoms.center(vacuum=...)
  - 自适应 nbands 时，用临时 LCAO 计算器且 nbands='nao'（确保初始化不会因带数过小报错）

自适应 nbands 思路：
  - 初始化一次 GPAW（不做 SCF）获得 valence 电子数 nelec_v；
  - nocc = ceil(nelec_v / 2)（闭壳层假设）；
  - nbands = nocc + max(min_extra, int(safety*nocc), per_atom_extra*Natoms)，再向上取整到 8 的倍数。
依赖：gsrun.py / td_calc.py / cdspecX.py 与 GPAW/ASE。

示例：
  python batch_all_gpaw_auto_nb.py
  python batch_all_gpaw_auto_nb.py --ranks 64 --basis dzp --nsteps 4000 --safety 0.8 --min_extra 30 --per_atom_extra 3
"""

import os
import sys
import glob
import shlex
import subprocess
import math
import argparse
from pathlib import Path


def run(cmd, cwd=None):
    print("[RUN]", cmd)
    ret = subprocess.run(cmd, shell=True, cwd=cwd)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed ({ret.returncode}): {cmd}")


def suggest_nbands(
    xyz_path,
    mode="lcao",
    basis="dzp",
    charge=0,
    safety=0.6,
    min_extra=20,
    per_atom_extra=2,
    round_to=8,
    vacuum=8.0,
):
    """Return (nbands, nocc, nelec_valence, Natoms).
    - safety: 按 nocc 的比例增加虚带（如 0.6 表示 +60% nocc）
    - min_extra: 至少增加的虚带数
    - per_atom_extra: 每个原子再加的虚带数（大体系给更多）
    - vacuum: 非周期分子盒子的真空厚度（Å）
    """
    from ase.io import read
    from gpaw import GPAW
    from gpaw.occupations import FermiDirac

    atoms = read(xyz_path)
    # 非周期分子：需要显式设置非周期并加真空盒，避免 “requires 3 lattice vectors” 报错
    atoms.pbc = False
    atoms.center(vacuum=float(vacuum))

    # --- 用临时 LCAO 计算器只为拿电子数（与实际 mode 无关） ---
    # 使用 nbands='nao' 确保“带数>=原子轨道数”，初始化不会因带数不足而失败
    try:
        tmp_calc = GPAW(mode="lcao", basis=basis,
                        occupations=FermiDirac(0.01),
                        nbands='nao', txt=None)
    except Exception:
        # 个别旧版不支持 'nao'，退回到一个“大到离谱”的整数以保证通过初始化
        tmp_calc = GPAW(mode="lcao", basis=basis,
                        occupations=FermiDirac(0.01),
                        nbands=2048, txt=None)

    atoms.calc = tmp_calc
    tmp_calc.initialize(atoms)
    nelec_v = tmp_calc.get_number_of_electrons() - charge  # 价电子总数（PAW valence）
    nocc = int(math.ceil(nelec_v / 2.0))                   # 闭壳层假设
    Nat = len(atoms)

    # 释放临时计算器（以免占内存）
    atoms.calc = None
    try:
        tmp_calc.close()  # 新版GPAW支持
    except Exception:
        pass

    extra = max(min_extra, int(safety * nocc), per_atom_extra * Nat)
    nbands = int(nocc + extra)
    if round_to > 1:
        nbands = int(math.ceil(nbands / round_to) * round_to)
    return nbands, nocc, nelec_v, Nat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", "-P", type=int, default=int(os.environ.get("RANKS", "96")))
    ap.add_argument(
        "--mode", default=os.environ.get("MODE", "lcao"), choices=["lcao", "fd"]
    )
    ap.add_argument("--basis", default=os.environ.get("BASIS", "dzp"))
    ap.add_argument(
        "--nbands",
        type=int,
        default=int(os.environ.get("NBANDS", "-1")),
        help=">0 固定；<=0 则自适应",
    )
    ap.add_argument("--charge", type=int, default=int(os.environ.get("CHARGE", "0")))
    ap.add_argument("--dt_as", type=float, default=float(os.environ.get("DT_AS", "20.0")))
    ap.add_argument("--nsteps", type=int, default=int(os.environ.get("NSTEPS", "1600")))
    ap.add_argument(
        "--gauge",
        default=os.environ.get("GAUGE", "length"),
        choices=["length", "velocity"],
    )
    ap.add_argument("--origin", default=os.environ.get("ORIGIN", "COM"))
    ap.add_argument(
        "--vacuum",
        type=float,
        default=float(os.environ.get("VACUUM", "8.0")),
        help="分子盒子真空厚度（Å），用于XYZ体系",
    )
    ap.add_argument("--xyz_root", default=os.environ.get("XYZ_ROOT", "xyz"))
    ap.add_argument("--kicks", default=os.environ.get("KICKS", "xyz"))
    # 自适应参数
    ap.add_argument(
        "--safety", type=float, default=float(os.environ.get("NB_SAFETY", "0.6"))
    )
    ap.add_argument(
        "--min_extra", type=int, default=int(os.environ.get("NB_MINEXTRA", "20"))
    )
    ap.add_argument(
        "--per_atom_extra", type=int, default=int(os.environ.get("NB_PERATOM", "2"))
    )
    args = ap.parse_args()

    xyz_files = sorted(glob.glob(str(Path(args.xyz_root) / "*.xyz")))
    if not xyz_files:
        print("[INFO] 未在", args.xyz_root, "找到 .xyz 文件")
        return

    for xyz in xyz_files:
        name = Path(xyz).stem
        work = Path(f"work_{name}")
        work.mkdir(exist_ok=True)

        # 计算/决定 nbands
        if args.nbands > 0:
            nb = args.nbands
            print(f"[NBAND] 固定 nbands={nb} (用户指定)")
        else:
            nb, nocc, nelec_v, Nat = suggest_nbands(
                xyz,
                mode=args.mode,
                basis=args.basis,
                charge=args.charge,
                safety=args.safety,
                min_extra=args.min_extra,
                per_atom_extra=args.per_atom_extra,
                vacuum=args.vacuum,
            )
            print(
                f"[NBAND] {name}: Nat={Nat}, nelec(val)≈{nelec_v:.1f}, nocc≈{nocc} → nbands={nb}"
            )

        # 1) 基态（用你自己的 gsrun.py）
        gs_cmd = (
            f"gpaw -P {args.ranks} python gsrun.py -- "
            f"--xyz {shlex.quote(xyz)} --mode {args.mode} --basis {shlex.quote(str(args.basis))} "
            f"--nbands {nb} --vacuum {args.vacuum} --outdir {shlex.quote(str(work))}"
        )
        run(gs_cmd)
        gpw = work / f"{name}.gpw"
        if not gpw.exists():
            raise FileNotFoundError(f"{gpw} not found after GS step.")

        # 2) rt-TDDFT 指定方向（默认 xyz）
        for k in list(args.kicks):
            td_cmd = (
                f"gpaw -P {args.ranks} python td_calc.py -- "
                f"--gpw {shlex.quote(str(gpw))} --mode {args.mode} --gauge {args.gauge} "
                f"--kick {k} --dt_as {args.dt_as} --nsteps {args.nsteps} "
                f"--origin {args.origin} --outdir {shlex.quote(str(work))}"
            )
            run(td_cmd)

        # 3) ECD 光谱（聚合 mm-*.dat）
        spec_cmd = (
            f"python cdspecX.py --prefix {name} --gauge {args.gauge} "
            f"--origin {args.origin} --outdir {shlex.quote(str(work))}"
        )
        run(spec_cmd)

    print("\n[ALL DONE] 结果位于 ./work_<name>/ 目录。")


if __name__ == "__main__":
    main()
