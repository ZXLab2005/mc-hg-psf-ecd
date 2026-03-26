#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_chiral_ag.py
一键生成一批简单手性 Ag(I) 配合物的 XYZ 坐标（原子数≪100），并自动输出 R/S 对映体。

默认分子骨架：四配位四面体 Ag(I)，四种不同配体 ⇒ 天然手性
示例组合（都很小）：
  [Ag(SCH3)(PH3)(NH3)Cl]
  [Ag(SCH3)(PH3)(NH3)CN]
  [Ag(SCH3)(PH3)(H2O)Cl]
  [Ag(SCH3)(PH3)(NH3)Br]

依赖：numpy, ase   （pip install numpy ase）
用法：直接运行
  python generate_chiral_ag.py
输出：在 ./ag_dataset_xyz 目录写出若干 <name>_R.xyz 与 <name>_S.xyz

说明：
- 这些几何是“物理合理”的构型原型，适合 ML 训练/预训练。
- 若要做 DFT/TDDFT（尤其是 ECD），建议再做一次快速几何优化（如 GFN2-xTB/DFTB 或低精度 DFT）
  以微调键长/角；但对“从结构学习光谱”的深度模型来说，这些初始几何已足够作为训练样本。
"""
import os, math, numpy as np
from dataclasses import dataclass
from typing import List
from ase import Atoms
from ase.io import write

# ===================== 工具函数 =====================

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def rand_orthonormal(u, rng):
    """返回与 u 垂直的两个单位向量（随机旋转过）"""
    a = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(a, u)) > 0.85:
        a = np.array([0.0, 1.0, 0.0])
    v1 = np.cross(u, a); v1 = unit(v1)
    v2 = np.cross(u, v1); v2 = unit(v2)
    theta = rng.uniform(0, 2*math.pi)
    v1r = math.cos(theta)*v1 + math.sin(theta)*v2
    v2r = -math.sin(theta)*v1 + math.cos(theta)*v2
    return v1r, v2r

def tetrahedron_dirs():
    """原点处正四面体四个顶点方向"""
    V = np.array([[ 1,  1,  1],
                  [ 1, -1, -1],
                  [-1,  1, -1],
                  [-1, -1,  1]], dtype=float)
    return np.array([unit(v) for v in V])

# ============= 配体构造（沿指定方向放置）==============

def place_SCH3(anchor_pos, dir_u, d_XC=1.82, d_CH=1.09, rng=None):
    """在“锚点 X”(S) 的 +u 方向放一个 CH3，形成 X-CH3；返回 C 与三 H 的坐标"""
    rng = rng or np.random.default_rng()
    C = anchor_pos + dir_u * d_XC
    v1, v2 = rand_orthonormal(dir_u, rng)
    angles = [0, 2*math.pi/3, 4*math.pi/3]
    Hs = []
    for ang in angles:
        d = unit(-0.8*dir_u + math.cos(ang)*v1 + math.sin(ang)*v2)
        Hs.append(C + d_CH * d)
    return C, np.array(Hs)

def place_XH3(anchor_pos, dir_u, d_XH, rng=None):
    """在“锚点 X”(P/N/O) 周围放三个 H（近似三角锥/金字塔）"""
    rng = rng or np.random.default_rng()
    v1, v2 = rand_orthonormal(dir_u, rng)
    angles = [0, 2*math.pi/3, 4*math.pi/3]
    Hs = []
    for ang in angles:
        d = unit(0.05*dir_u + math.cos(ang)*v1 + math.sin(ang)*v2)
        Hs.append(anchor_pos + d_XH * d)
    return np.array(Hs)

def place_CN(Ag_pos, dir_u, d_AgC=2.05, d_CN=1.16):
    """CN-：Ag–C≡N，C 在 +u，N 再沿 +u"""
    C = Ag_pos + dir_u * d_AgC
    N = C + dir_u * d_CN
    return C, N

def place_H2O(anchor_pos, dir_u, d_OH=0.96, angle_HOH=104.5, rng=None):
    """在 O 周围放两个 H，HOH ≈ 104.5°，平面近似垂直于 dir_u"""
    rng = rng or np.random.default_rng()
    v1, v2 = rand_orthonormal(dir_u, rng)
    # 两个氢在 v1 方向两侧
    half = math.radians(angle_HOH/2.0)
    H1 = anchor_pos + d_OH * unit( math.cos(half)*v1 + math.sin(half)*v2 )
    H2 = anchor_pos + d_OH * unit( math.cos(half)*v1 - math.sin(half)*v2 )
    return H1, H2

# =================== 体系参数 ===================

@dataclass
class BondParams:
    AgS: float = 2.45
    AgP: float = 2.50
    AgN: float = 2.10
    AgCl: float = 2.30
    AgBr: float = 2.45
    AgC_CN: float = 2.05
    SC: float  = 1.82
    CH: float  = 1.09
    PH: float  = 1.42
    NH: float  = 1.02
    OH: float  = 0.96
    CN: float  = 1.16

BP = BondParams()

# ============== 构建手性复合物（四面体）==============

def build_chiral_Ag_tetra(ligands: List[str], handedness='R', noise=0.02, seed=0):
    """
    ligands: 长度=4，来自 {'SMe','PH3','NH3','Cl','Br','CN','H2O'}
    handedness: 'R' 或 'S'，通过交换顶点实现对映体
    """
    rng = np.random.default_rng(seed)
    Ag = np.zeros(3)
    D = tetrahedron_dirs()

    # 随机整体旋转，增加多样性
    A = rng.normal(size=(3,3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0: Q[:,0] *= -1
    D = (Q @ D.T).T

    # R/S：交换两个顶点以构造镜像
    order = [0,1,2,3] if handedness.upper()=='R' else [0,2,1,3]

    symbols = ['Ag']
    positions = [Ag.copy()]

    for k, lig in enumerate(ligands):
        u = unit(D[order[k]] + rng.normal(scale=noise, size=3))
        if lig == 'SMe':
            S = Ag + u * BP.AgS
            C, Hs = place_SCH3(S, u, d_XC=BP.SC, d_CH=BP.CH, rng=rng)
            symbols += ['S','C'] + ['H']*3
            positions += [S, C] + [h for h in Hs]
        elif lig == 'PH3':
            P = Ag + u * BP.AgP
            Hp = place_XH3(P, u, BP.PH, rng=rng)
            symbols += ['P'] + ['H']*3
            positions += [P] + [h for h in Hp]
        elif lig == 'NH3':
            N = Ag + u * BP.AgN
            Hn = place_XH3(N, u, BP.NH, rng=rng)
            symbols += ['N'] + ['H']*3
            positions += [N] + [h for h in Hn]
        elif lig == 'H2O':
            # 先放 O，再放两个 H
            O = Ag + u * (BP.AgN + 0.10)  # 近似 Ag–O 距离（与 Ag–N 相近）
            H1, H2 = place_H2O(O, u, d_OH=BP.OH, rng=rng)
            symbols += ['O','H','H']
            positions += [O, H1, H2]
        elif lig == 'Cl':
            Cl = Ag + u * BP.AgCl
            symbols += ['Cl']; positions += [Cl]
        elif lig == 'Br':
            Br = Ag + u * BP.AgBr
            symbols += ['Br']; positions += [Br]
        elif lig == 'CN':
            C, N = place_CN(Ag, u, d_AgC=BP.AgC_CN, d_CN=BP.CN)
            symbols += ['C','N']; positions += [C, N]
        else:
            raise ValueError(f'未知配体: {lig}')

    atoms = Atoms(symbols, positions=np.array(positions))
    atoms.center(vacuum=8.0)
    return atoms

# ============== 轻量“去碰撞”修饰（非优化，仅做排斥修正） ==============

def steric_push(atoms, min_d=1.15, step=0.03, iters=60):
    """
    极简“排斥”迭代：发现非键合原子间距 < min_d 时，沿连线各推一步。
    仅用于避免偶发重叠；不替代量化优化。
    """
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    N = len(atoms)
    for _ in range(iters):
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                # 忽略相邻共价对（非常粗糙：H-与近邻、Ag-锚点等）
                pair = {sym[i], sym[j]}
                if 'Ag' in pair and ('S' in pair or 'P' in pair or 'N' in pair or 'O' in pair or 'C' in pair or 'Cl' in pair or 'Br' in pair):
                    continue
                if 'C' in pair and 'H' in pair:  # CH 内键
                    continue
                rij = pos[j] - pos[i]
                d = np.linalg.norm(rij)
                if d < 1e-8: continue
                if d < min_d:
                    delta = step * rij / d
                    pos[i] -= 0.5*delta
                    pos[j] += 0.5*delta
                    moved = True
        if not moved:
            break
    atoms.set_positions(pos)
    atoms.center(vacuum=8.0)

# ============== 主程序：批量生成 =====================

def main():
    outdir = 'ag_dataset_xyz'
    os.makedirs(outdir, exist_ok=True)

    # 预设若干非常小的“手性四配位”组合（四个配体各不同 ⇒ 手性）
    combos = [
        ('Ag_SMe_PH3_NH3_Cl', ['SMe','PH3','NH3','Cl']),
        ('Ag_SMe_PH3_NH3_CN', ['SMe','PH3','NH3','CN']),
        ('Ag_SMe_PH3_H2O_Cl', ['SMe','PH3','H2O','Cl']),
        ('Ag_SMe_PH3_NH3_Br', ['SMe','PH3','NH3','Br']),
    ]

    seeds = list(range(1, 6))   # 每种组合生成 5 个随机变体（R/S 各一）

    total = 0
    for name, ligs in combos:
        for s in seeds:
            for hand in ['R','S']:
                atoms = build_chiral_Ag_tetra(ligs, handedness=hand, noise=0.02, seed=1000*s)
                steric_push(atoms, min_d=1.15, step=0.03, iters=60)
                fname = os.path.join(outdir, f'{name}_seed{s}_{hand}.xyz')
                write(fname, atoms)
                total += 1
                # 打印一次原子数，确认远小于 100
                if s == seeds[0]:
                    print(f'[{name} {hand}] atoms = {len(atoms)} → {fname}')
    print(f'\n[OK] 写出 {total} 个 XYZ 到 ./{outdir}/  （每条都≪100 原子）')

if __name__ == '__main__':
    main()
