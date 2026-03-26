#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_density_TCM_draw.py —— 三方向 FDM → 各向性平均 TCM（含覆盖校验）
python plot_density_TCM_draw.py --dirs --sigma 0.10 --de 0.01 --occ -6 0.2 --uno -0.2 6 --sample-id Ag4

增强项：
- 全局字体放大加粗SS
- 在对角线上标注公式 E_a - E_i = ħω
- 支持 --sample-id 并在主图内标注 ID
- 兼容无 plotter.ax 的 GPAW 版本（用返回的 AxesImage 或 plt.gca() 兜底）
"""

import os, csv, argparse, numpy as np, matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.lcaotddft.tcm import TCMPlotter

# --------- 全局字体（放大加粗） ---------
plt.rcParams.update({
    'font.size': 16,
    'font.weight': 'bold',
    'axes.labelsize': 16,
    'axes.titlesize': 20,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

ORANGE = '#ff7f0e'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--unocc', default='unocc.gpw', help='unocc.gpw 路径（用于 KSD 与兜底取本征值）')
    p.add_argument('--ksd',   default='ksd.ulm',   help='ksd.ulm 路径')
    p.add_argument('--fdm-x', default='fdm_x.ulm'); p.add_argument('--fdm-y', default='fdm_y.ulm'); p.add_argument('--fdm-z', default='fdm_z.ulm')
    p.add_argument('--sigma', type=float, default=0.10)
    p.add_argument('--de',    type=float, default=0.01)
    p.add_argument('--occ',   nargs=2, type=float, default=(-6.0, 0.2))
    p.add_argument('--uno',   nargs=2, type=float, default=(-0.2, 6.0))
    p.add_argument('--max-cells', type=int, default=8_000_000)
    p.add_argument('--prefix', default='tcm_iso')
    p.add_argument('--dirs', action='store_true', help='是否单独画出 x/y/z 三方向 TCM')
    p.add_argument('--minweight', type=float, default=0.10)
    # 覆盖校验兜底来源
    p.add_argument('--energies-csv', default='ks_energies.csv', help='可选：occ/unocc 能级 CSV（type,index,energy）')
    p.add_argument('--unocc-gpw',    default=None, help='可选：若与 --unocc 不同，可指定另一份 gpw 供兜底取本征值')
    p.add_argument('--skip-cov', action='store_true', help='跳过覆盖校验（只画图）')
    # 新增：样本 ID 标注
    p.add_argument('--sample-id', default='', help='在图内标注的样本 ID（例如 7202365）')
    return p.parse_args()

def build_grids_with_limit(eomin, eomax, eumin, eumax, de_init, max_cells, sigma):
    de = float(de_init)
    while True:
        eo = np.arange(eomin, eomax + 1e-12, de)
        eu = np.arange(eumin, eumax + 1e-12, de)
        cells = eo.size * eu.size
        if cells <= max_cells:
            print(f"[window/manual] occ:[{eomin:.2f},{eomax:.2f}] eV | unocc:[{eumin:.2f},{eumax:.2f}] eV | de={de:.3f}, sigma={sigma:.2f} | cells={cells/1e6:.2f}M")
            return eo, eu, de
        de *= 1.5

def load_fdm(calc, dmat, fn):
    if not os.path.exists(fn): raise FileNotFoundError(f'缺少 FDM：{fn}')
    return FrequencyDensityMatrix(calc, dmat, fn)

def check_freq_consistency(fdm_dict):
    freq_lists = {tag: np.array([fw.freq * au_to_eV for fw in fdm.freq_w]) for tag, (fdm, _) in fdm_dict.items()}
    ref, tol = freq_lists['x'], 1e-3
    for tag, arr in freq_lists.items():
        if len(arr) != len(ref) or np.max(np.abs(arr - ref)) > tol:
            raise ValueError(f'FDM 频点不一致：x vs {tag}')
    return ref

# ---------- 能级提取 & 量纲对齐 ----------
def _robust_scale(x):
    x = np.asarray(x, float).ravel(); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    m = np.median(x); return np.median(np.abs(x - m))

def align_unocc_like_occ(unocc_e, occ_e):
    ue = np.array(unocc_e, float).ravel(); oe = np.array(occ_e, float).ravel()
    su, so = _robust_scale(ue), _robust_scale(oe)
    if np.isfinite(su) and np.isfinite(so) and su > 0 and so > 0:
        r = su / so
        if 20.0 < r < 35.0:   # ~27.2 Ha→eV
            return ue * au_to_eV
        if 1/35.0 < r < 1/20.0:
            return ue / au_to_eV
    return ue

def try_occ_unocc_from_ksd(ksd):
    occ = uno = None
    for name in ('eig_o','eig_n','eig_occ','occ_eigs'):
        a = getattr(ksd, name, None)
        if a is not None:
            try:
                occ = np.array(a, float).ravel(); break
            except: pass
    for name in ('eig_u','eig_m','eig_p','eig_unocc','unocc_eigs'):
        a = getattr(ksd, name, None)
        if a is not None:
            try:
                uno = np.array(a, float).ravel(); break
            except: pass
    if occ is None or uno is None:
        try:
            ret = ksd.get_eig_n()
            if isinstance(ret,(list,tuple)) and len(ret)>=2:
                if occ is None: occ = np.array(ret[0], float).ravel()
                if uno is None: uno = np.array(ret[1], float).ravel()
        except: pass
    return occ, uno

def try_occ_unocc_from_csv(csv_path):
    if not (csv_path and os.path.exists(csv_path)): return None, None
    occ_list, uno_list = [], []
    with open(csv_path,'r',encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for row in rd:
            t = (row.get('type') or row.get('kind') or '').strip().lower()
            e = row.get('energy')
            if e is None: continue
            try: val = float(e)
            except: continue
            if t in ('occ','occupied'): occ_list.append(val)
            elif t in ('unocc','unocc_aligned','unoccupied'): uno_list.append(val)
    occ = np.array(occ_list,float) if occ_list else None
    uno = np.array(uno_list,float) if uno_list else None
    return occ, uno

def try_occ_unocc_from_gpw(gpw_path):
    if not (gpw_path and os.path.exists(gpw_path)): return None, None
    calc = GPAW(gpw_path, txt=None)
    e = calc.get_eigenvalues(kpt=0)  # eV
    Ef = calc.get_fermi_level()      # eV
    occ = np.array(sorted(e[e <= Ef + 1e-6]), float)
    uno = np.array(sorted(e[e >  Ef + 1e-6]), float)
    return occ, uno

def get_occ_unocc_for_coverage(ksd, csv_path, gpw_path, ia_p):
    occ, uno = try_occ_unocc_from_ksd(ksd)
    def finite_cnt(x): return int(np.isfinite(x).sum()) if isinstance(x,np.ndarray) else 0
    if not isinstance(occ,np.ndarray) or finite_cnt(occ)==0 or occ.size<=int(np.max(ia_p[:,0])): occ = None
    if not isinstance(uno,np.ndarray) or finite_cnt(uno)==0 or uno.size<=int(np.max(ia_p[:,1])): uno = None
    src = 'KSD'
    if occ is None or uno is None:
        occ_csv, uno_csv = try_occ_unocc_from_csv(csv_path)
        if occ is None and isinstance(occ_csv,np.ndarray) and finite_cnt(occ_csv)>0: occ = occ_csv; src='CSV'
        if uno is None and isinstance(uno_csv,np.ndarray) and finite_cnt(uno_csv)>0: uno = uno_csv; src='CSV'
    if occ is None or uno is None:
        occ_gpw, uno_gpw = try_occ_unocc_from_gpw(gpw_path or ksd.calc.wfs.gd.filename)
        if occ is None and isinstance(occ_gpw,np.ndarray) and occ_gpw.size>0: occ = occ_gpw; src='GPW'
        if uno is None and isinstance(uno_gpw,np.ndarray) and uno_gpw.size>0: uno = uno_gpw; src='GPW'
    i_max = int(np.max(ia_p[:,0])); a_max = int(np.max(ia_p[:,1]))
    if occ is None: occ = np.full(i_max+1, np.nan)
    if uno is None: uno = np.full(a_max+1, np.nan)
    uno = align_unocc_like_occ(uno, occ)
    if occ.size <= i_max: occ = np.pad(occ, (0, i_max+1-occ.size), constant_values=np.nan)
    if uno.size <= a_max: uno = np.pad(uno, (0, a_max+1-uno.size), constant_values=np.nan)
    print(f"[cov-source] energies from {src} | finite(occ)={int(np.isfinite(occ).sum())}/{occ.size}, finite(unocc)={int(np.isfinite(uno).sum())}/{uno.size}")
    return occ, uno

# ---------- 样式与标注 ----------
def beautify_all_axes(fig):
    for ax in fig.get_axes():
        ax.tick_params(axis='both', which='major', width=1.6, length=6)
        for item in (ax.title, ax.xaxis.label, ax.yaxis.label):
            try:
                item.set_fontweight('bold')
            except Exception:
                pass

def annotate_diagonal(ax, eomin, eomax, eumin, eumax, omega):
    """在对角线附近标公式：无描边框，半透明白底可读（可去掉 facecolor=... 就完全透明）。"""
    # 选取对角线上较靠左的点，如果超界就回退到 20% 高度
    x = eomin + 0.18 * (eomax - eomin)
    y = x + omega
    if not (eumin <= y <= eumax):
        y = eumin + 0.20 * (eumax - eumin)
        x = y - omega

    ax.text(x, y, r'$E_a - E_i = \hbar\omega$',
            rotation=45, ha='center', va='center',
            fontsize=15, fontweight='bold',
            # 无黑色边框：edgecolor='none'；若想完全没有底色可去掉 bbox 参数
            bbox=dict(boxstyle='round,pad=0.15',
                      facecolor='white', alpha=0.55, edgecolor='none'))

def annotate_sample_id(ax, eomin, eomax, eumin, eumax, sample_id):
    """在主图上方居中标出样本 ID：无黑边，文字在白底框中居中，避免遮挡左上角色码。"""
    if not sample_id:
        return
    # 放到主图上缘中间位置，离顶留 4% 高度，不挡左上角色码
    x = eomin + 0.50 * (eomax - eomin)   # 水平居中
    y = eumin + 0.96 * (eumax - eumin)   # 顶部下方一点
    ax.text(x, y, f'ID: {sample_id}',
            ha='center', va='center',
            fontsize=15, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.25',
                      facecolor='white', alpha=0.75, edgecolor='none'))

# ---------- 取 TCM 主轴（兼容不同版本） ----------
def get_tcm_axes(plotter, im_after_plot_tcm):
    # 优先取新版属性
    ax = getattr(plotter, 'ax', None)
    if ax is not None:
        return ax
    # 再试 AxesImage 返回值
    try:
        ax = im_after_plot_tcm.axes
        if ax is not None:
            return ax
    except Exception:
        pass
    # 最后兜底：当前轴
    return plt.gca()

# ---------- 绘图/表格 ----------
def draw_one_tcm(ksd, energy_o, energy_u, sigma, w_vec, freq_eV, title, path, sample_id):
    fig = plt.figure(figsize=(6.8, 6.6))
    plotter = TCMPlotter(ksd, energy_o, energy_u, sigma=sigma)
    im = plotter.plot_TCM(w_vec)
    plotter.plot_DOS(fill={'color': ORANGE}, line={'color': 'k'})
    plotter.plot_TCM_diagonal(freq_eV, color='k')
    plotter.set_title(title)
    beautify_all_axes(fig)
    # 关键修复：稳健获取 TCM 主轴
    ax_main = get_tcm_axes(plotter, im)
    annotate_diagonal(ax_main, energy_o.min(), energy_o.max(), energy_u.min(), energy_u.max(), freq_eV)
    annotate_sample_id(ax_main, energy_o.min(), energy_o.max(), energy_u.min(), energy_u.max(), sample_id)
    plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()

def save_contrib_csv(ksd, w_vec, csv_path, minweight=0.10):
    ia_p = ksd.ia_p; w = np.array(w_vec, float).ravel(); W = float(np.sum(w))
    keep = np.where(np.abs(w) >= minweight)[0]
    if keep.size == 0: keep = np.argsort(np.abs(w))[-50:]
    occ, uno = try_occ_unocc_from_ksd(ksd)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['pair_index','i','a','Ei_eV','Ea_eV','weight','percent'])
        for p in keep:
            i = int(ia_p[p,0]); a = int(ia_p[p,1])
            Ei = float(occ[i]) if isinstance(occ,np.ndarray) and i < len(occ) else float('nan')
            Ea = float(uno[a]) if isinstance(uno,np.ndarray) and a < len(uno) else float('nan')
            wp = float(w[p]); perc = 100.0*wp/W if W!=0.0 else 0.0
            wr.writerow([p,i,a,Ei,Ea,wp,perc])

# ---------- 主流程 ----------
def main():
    args = parse_args()
    for f in [args.unocc, args.ksd, args.fdm_x, args.fdm_y, args.fdm_z]:
        if not os.path.exists(f): raise FileNotFoundError(f'未找到：{f}')

    calc = GPAW(args.unocc, txt=None)
    ksd  = KohnShamDecomposition(calc, args.ksd)
    dmat = DensityMatrix(calc)

    fdm_dict = {
        'x': (load_fdm(calc, dmat, args.fdm_x), 0),
        'y': (load_fdm(calc, dmat, args.fdm_y), 1),
        'z': (load_fdm(calc, dmat, args.fdm_z), 2),
    }
    freq_eV_list = check_freq_consistency(fdm_dict)

    eomin, eomax = args.occ; eumin, eumax = args.uno
    energy_o, energy_u, de_used = build_grids_with_limit(eomin, eomax, eumin, eumax, args.de, args.max_cells, args.sigma)

    ia = ksd.ia_p
    print(f"[pairs] ia_p: i_max={int(np.max(ia[:,0]))}, a_max={int(np.max(ia[:,1]))}")

    for widx, freq_eV in enumerate(map(float, freq_eV_list)):
        w_sum = None; per_dir_total = []
        for tag, (fdm, comp) in fdm_dict.items():
            rho_uMM = fdm.FReDrho_wuMM[widx]
            freq    = fdm.freq_w[widx]
            rho_up  = ksd.transform(rho_uMM)
            dmrho_vp = ksd.get_dipole_moment_contributions(rho_up)
            w_p = (2.0 * freq.freq / np.pi) * dmrho_vp[comp].imag / au_to_eV * 1e5
            per_dir_total.append((tag, float(np.sum(w_p))))
            w_sum = w_p if w_sum is None else (w_sum + w_p)
            if args.dirs:
                draw_one_tcm(ksd, energy_o, energy_u, args.sigma, w_p, freq_eV,
                             f'{tag}-pol absorption TCM at {freq_eV:.2f} eV',
                             f'{args.prefix}_{tag}_{freq_eV:.2f}eV.png',
                             args.sample_id)

        w_iso = w_sum/3.0; iso_total = float(np.sum(w_iso))
        print(f"\nFreq {freq_eV:.2f} eV | per-dir totals: " + ', '.join([f'{t}:{s:.3f}' for t,s in per_dir_total]) + f' | iso-avg total: {iso_total:.3f} eV^-1')

        # 表格
        table_txt = ksd.get_contributions_table(w_iso, minweight=args.minweight)
        with open(f'{args.prefix}_table_{freq_eV:.2f}eV.txt','w',encoding='utf-8') as f:
            f.write(f'Frequency: {freq_eV:.2f} eV\n')
            f.write('Totals per direction: ' + ', '.join([f'{t}:{s:.3f}' for t,s in per_dir_total]) + '\n')
            f.write(f'Isotropic total: {iso_total:.6f} eV^-1\n')
            f.write(table_txt)
        save_contrib_csv(ksd, w_iso, f'{args.prefix}_table_{freq_eV:.2f}eV.csv', minweight=args.minweight)

        # 各向性 TCM（加粗、公式与ID；同样兼容无 ax 的版本）
        fig = plt.figure(figsize=(6.8, 6.6))
        plotter = TCMPlotter(ksd, energy_o, energy_u, sigma=args.sigma)
        im = plotter.plot_TCM(w_iso)
        plotter.plot_DOS(fill={'color': ORANGE}, line={'color': 'k'})
        plotter.plot_TCM_diagonal(freq_eV, color='k')
        plotter.set_title(f'Isotropic absorption TCM at {freq_eV:.2f} eV')
        beautify_all_axes(fig)
        ax_main = get_tcm_axes(plotter, im)
        annotate_diagonal(ax_main, energy_o.min(), energy_o.max(), energy_u.min(), energy_u.max(), freq_eV)
        annotate_sample_id(ax_main, energy_o.min(), energy_o.max(), energy_u.min(), energy_u.max(), args.sample_id)
        plt.savefig(f'{args.prefix}_{freq_eV:.2f}eV.png', dpi=300, bbox_inches='tight'); plt.close()

        # 覆盖校验（逻辑不变）
        if args.skip_cov:
            continue
        tcm_ou = ksd.get_TCM(w_iso, (plotter.eig_n if hasattr(plotter,'eig_n') else ksd.get_eig_n()[0]),
                             energy_o, energy_u, sigma=args.sigma)
        tcm_abs = float(np.sum(tcm_ou) * (energy_o[1]-energy_o[0]) * (energy_u[1]-energy_u[0]))
        print(f"tcmmax {float(np.max(np.abs(tcm_ou)))}")

        occ_cov, uno_cov = get_occ_unocc_for_coverage(
            ksd,
            csv_path=args.energies_csv if args.energies_csv else None,
            gpw_path=(args.unocc_gpw if args.unocc_gpw else args.unocc),
            ia_p=ia
        )
        P = ia.shape[0]
        Ei = np.full(P, np.nan, float); Ea = np.full(P, np.nan, float)
        i_valid = (ia[:,0]>=0) & (ia[:,0] < len(occ_cov))
        a_valid = (ia[:,1]>=0) & (ia[:,1] < len(uno_cov))
        valid = i_valid & a_valid
        Ei[valid] = occ_cov[ia[valid,0]]; Ea[valid] = uno_cov[ia[valid,1]]

        try:
            print(f"  [range] Ei in [{np.nanmin(Ei):.2f},{np.nanmax(Ei):.2f}] eV, Ea in [{np.nanmin(Ea):.2f},{np.nanmax(Ea):.2f}] eV; "
                  f"occ_win=[{energy_o.min():.2f},{energy_o.max():.2f}], uno_win=[{energy_u.min():.2f},{energy_u.max():.2f}]")
        except ValueError:
            print("  [range] no finite Ei/Ea values to summarize")

        cover_mask = valid & (Ei>=energy_o.min()) & (Ei<=energy_o.max()) & (Ea>=energy_u.min()) & (Ea<=energy_u.max())
        n_cov = int(np.nansum(cover_mask.astype(int)))
        covered_sum = float(np.nansum(w_iso[cover_mask])) if n_cov>0 else 0.0
        rel_err_cov = 0.0 if covered_sum==0 else abs(tcm_abs - covered_sum)/abs(covered_sum)
        print(f'  [check/cov] grid-integral ≈ {tcm_abs:.3f} eV^-1  | covered-sum ≈ {covered_sum:.3f} eV^-1 | covered_pairs={n_cov} | rel.err={100*rel_err_cov:.2f}%')

if __name__ == '__main__':
    main()
