# python plot_density_Kohn-sham.py --force --nbands nao --csv ks_energies.csv
# 功能：基于基态 gs.gpw 生成 unocc.gpw（含足够未占据态）与 ksd.ulm（KS e–h 基），并做能级自检
import os
import argparse
import numpy as np
from gpaw import GPAW
from gpaw.tddft.units import au_to_eV
from gpaw.lcaotddft.ksdecomposition import KohnShamDecomposition

def parse_args():
    p = argparse.ArgumentParser(description='Prepare unocc.gpw & ksd.ulm robustly (for TCM).')
    p.add_argument('--gs', default='Ag4.gpw', help='基态 GPW 文件（默认 gs.gpw）')
    p.add_argument('--unocc', default='unocc.gpw', help='输出：含未占据态的 GPW（默认 unocc.gpw）')
    p.add_argument('--ksd', default='ksd.ulm', help='输出：KS 分解文件（默认 ksd.ulm）')
    p.add_argument('--nbands', default='nao',
                   help="未占据带设置：'nao' 或 整数；也可用 'occ+N'（如 'occ+300'）")
    p.add_argument('--extra', type=int, default=300,
                   help="当 nbands='nao' 失败时的回退：占据数 + extra（默认 300）")
    p.add_argument('--force', action='store_true', help='若已存在文件则强制重建')
    p.add_argument('--csv', default='ks_energies.csv', help='可选：导出占据/未占据能级 CSV')
    return p.parse_args()

def robust_scale(x):
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))

def align_unit_like_occ(unocc_e, occ_e):
    """根据稳健尺度判断是否需 Ha↔eV 转换，使未占据与占据的量纲一致。"""
    ue = np.array(unocc_e, float).ravel()
    oe = np.array(occ_e, float).ravel()
    su, so = robust_scale(ue), robust_scale(oe)
    if np.isfinite(su) and np.isfinite(so) and su > 0 and so > 0:
        r = su / so
        if 20.0 < r < 35.0:      # ~27.2 ≈ Ha→eV
            return ue * au_to_eV
        if 1/35.0 < r < 1/20.0:  # eV→Ha
            return ue / au_to_eV
    return ue

def guess_occ_bands(calc_gs):
    """粗略估计占据带数：Ne/2（闭壳层近似）。若失败，回退 0。"""
    try:
        ne = calc_gs.get_number_of_electrons()
        return int(np.ceil(float(ne) / 2.0))
    except Exception:
        return 0

def make_unocc_gpw(gs_path, unocc_path, nbands_spec, extra, force):
    if os.path.exists(unocc_path) and not force:
        print(f'[skip] 已存在 {unocc_path}')
        return

    if not os.path.exists(gs_path):
        raise FileNotFoundError(f'未找到基态文件：{gs_path}')
    print(f'[load] 读取基态：{gs_path}')
    calc_gs = GPAW(gs_path, txt='unocc.out')

    # 解析 nbands 规格
    nb = None
    nb_mode = str(nbands_spec).strip().lower()
    if nb_mode == 'nao':
        try:
            print("[try] fixed_density(nbands='nao')")
            calc_unocc = calc_gs.fixed_density(nbands='nao', txt='unocc.out')
            calc_unocc.write(unocc_path, mode='all')
            print(f'[ok] 写出：{unocc_path}  (nbands=NAO)')
            return
        except Exception as e:
            print(f'[warn] nbands="nao" 不可用或失败：{e}\n      → 回退到 occ+extra 策略')

    elif nb_mode.startswith('occ+'):
        try:
            nb_add = int(nb_mode.split('+', 1)[1])
            occ_est = guess_occ_bands(calc_gs)
            nb = max(occ_est + nb_add, occ_est + 1)
        except Exception:
            pass
    else:
        # 试图解析为整数
        try:
            nb = int(nb_mode)
        except Exception:
            pass

    # 若未能解析，采用 occ+extra 回退
    if nb is None:
        occ_est = guess_occ_bands(calc_gs)
        nb = max(occ_est + int(extra), occ_est + 1)
        print(f'[fallback] 使用回退 nbands={nb} (≈ occ({occ_est})+{extra})')
    else:
        print(f'[use] 使用 nbands={nb}')

    # 生成 unocc.gpw
    calc_unocc = calc_gs.fixed_density(nbands=nb, txt='unocc.out')
    calc_unocc.write(unocc_path, mode='all')
    print(f'[ok] 写出：{unocc_path}')

def make_ksd(unocc_path, ksd_path, force):
    if os.path.exists(ksd_path) and not force:
        print(f'[skip] 已存在 {ksd_path}')
        return
    if not os.path.exists(unocc_path):
        raise FileNotFoundError(f'未找到 {unocc_path}（请先生成 unocc.gpw）')

    print(f'[setup] 基于 {unocc_path} 生成 {ksd_path} ...')
    calc = GPAW(unocc_path, txt=None)
    ksd = KohnShamDecomposition(calc)
    ksd.initialize(calc)
    ksd.write(ksd_path)
    print(f'[ok] 写出：{ksd_path}')

def extract_occ_unocc_from_ksd(ksd):
    """
    尽可能从 KSD 中拿到与 ia_p 对齐的占据/未占据能量（1D np.array，单位未必是 eV）。
    依次尝试：显式属性 -> get_eig_n() -> 按 ia_p 补 NaN。
    """
    occ = None; uno = None
    occ_candidates = ('eig_o', 'eig_n', 'eig_occ', 'occ_eigs')
    uno_candidates = ('eig_u', 'eig_m', 'eig_p', 'eig_unocc', 'unocc_eigs')

    def _take(name):
        arr = getattr(ksd, name, None)
        if arr is None:
            return None
        try:
            a = np.array(arr, float).ravel()
            return a if a.size > 0 else None
        except Exception:
            return None

    for name in occ_candidates:
        occ = _take(name)
        if occ is not None:
            break
    for name in uno_candidates:
        uno = _take(name)
        if uno is not None:
            break

    # 再试 get_eig_n()
    if occ is None or uno is None:
        try:
            ret = ksd.get_eig_n()
            if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                if occ is None and hasattr(ret[0], '__len__'):
                    occ = np.array(ret[0], float).ravel()
                if uno is None and hasattr(ret[1], '__len__'):
                    uno = np.array(ret[1], float).ravel()
        except Exception:
            pass

    # 兜底：按 ia_p 最大索引补 NaN，保证不会报错
    ia = ksd.ia_p
    if occ is None:
        i_max = int(np.max(ia[:, 0])); occ = np.full(i_max + 1, np.nan)
    if uno is None:
        a_max = int(np.max(ia[:, 1])); uno = np.full(a_max + 1, np.nan)
    return occ, uno

def verify_and_dump(unocc_path, ksd_path, csv_path=None):
    """自检：打印/导出占据与未占据能级数量与范围，便于后续 TCM 覆盖校验不过时定位问题。"""
    calc = GPAW(unocc_path, txt=None)
    ksd = KohnShamDecomposition(calc, ksd_path)

    occ, uno = extract_occ_unocc_from_ksd(ksd)
    # 尝试把未占据能量对齐到与占据相同量纲
    uno_aligned = align_unit_like_occ(uno, occ)

    ia = ksd.ia_p
    i_max = int(np.max(ia[:, 0])); a_max = int(np.max(ia[:, 1]))
    print(f'[pairs] ia_p: i_max={i_max}, a_max={a_max} | len(occ)={len(occ)}, len(unocc)={len(uno)}')

    def rng(x):
        x = np.array(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return (np.nan, np.nan, 0)
        return (float(np.min(x)), float(np.max(x)), int(x.size))

    emin_o, emax_o, n_o = rng(occ)
    emin_u, emax_u, n_u = rng(uno_aligned)

    print(f'[range] occ(E): [{emin_o:.2f}, {emax_o:.2f}]  (finite={n_o})')
    print(f'[range] unocc(E): [{emin_u:.2f}, {emax_u:.2f}]  (finite={n_u})  (已与占据量纲对齐)')

    if csv_path:
        try:
            import csv as _csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = _csv.writer(f)
                w.writerow(['type', 'index', 'energy'])
                for i, e in enumerate(occ):
                    w.writerow(['occ', i, e])
                for a, e in enumerate(uno_aligned):
                    w.writerow(['unocc_aligned', a, e])
            print(f'[dump] KS 能级已导出：{csv_path}')
        except Exception as e:
            print(f'[warn] 导出 CSV 失败：{e}')

def main():
    args = parse_args()
    make_unocc_gpw(args.gs, args.unocc, args.nbands, args.extra, args.force)
    make_ksd(args.unocc, args.ksd, args.force)
    # 生成后立即自检（打印范围与有限元素数量），并可选导出 CSV
    verify_and_dump(args.unocc, args.ksd, csv_path=args.csv if args.csv else None)

if __name__ == '__main__':
    main()
