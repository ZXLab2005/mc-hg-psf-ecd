# plot_density_frequncy.py
# 功能：对 x/y/z 三个方向分别 replay 波函数，生成各自的 FDM（fdm_x/y/z.ulm）
import os
from gpaw.lcaotddft import LCAOTDDFT
from gpaw.lcaotddft.densitymatrix import DensityMatrix
from gpaw.lcaotddft.frequencydensitymatrix import FrequencyDensityMatrix
from gpaw.tddft.folding import frequencies as make_freqs

# ===== 配置区 =====
GS = 'Ag4.gpw'                          # 注意：用同一个基态文件
FREQS = [3.30, 3.70]                   # 需分析的频点（eV），与画谱时一致
FOLD, WIDTH = 'Gauss', 0.10            # 展宽类型与宽度（与画谱一致）
WF_FILES = {                           # 你的三方向波函数时间序列
    'x': 'wf-length_x.ulm',
    'y': 'wf-length_y.ulm',
    'z': 'wf-length_z.ulm',
}
# ==================

def build_fdm_for_dir(tag: str, wf_file: str):
    """对单个方向执行 replay 并写出 FDM。"""
    if not os.path.exists(GS):
        raise FileNotFoundError(f'未找到基态文件：{GS}')
    if not os.path.exists(wf_file):
        raise FileNotFoundError(f'未找到波函数文件：{wf_file}')

    print(f'[replay] 方向 {tag}: 使用 {wf_file}')
    td = LCAOTDDFT(GS, txt=None)
    dmat = DensityMatrix(td)
    # ★ 关键：用关键字参数 frequencies=... 传入（否则会被当成 filename 触发你看到的报错）
    fdm = FrequencyDensityMatrix(td, dmat, frequencies=make_freqs(FREQS, FOLD, WIDTH))

    # 逐方向重放（不更新哈密顿量/密度）
    td.replay(name=wf_file, update='none')

    out_ulm = f'fdm_{tag}.ulm'
    fdm.write(out_ulm)
    print(f'[ok] 方向 {tag}: 写出 {out_ulm}')

def main():
    for tag, wf in WF_FILES.items():
        build_fdm_for_dir(tag, wf)
    print('[done] 已生成 fdm_x.ulm / fdm_y.ulm / fdm_z.ulm')

if __name__ == '__main__':
    main()
