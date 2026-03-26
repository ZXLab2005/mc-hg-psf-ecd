# cdspecX.py —— Build ECD spectrum from mm files and plot
# python cdspecX.py --prefix Ag4 --gauge length --origin COM --outdir ./work_Ag4

import argparse, os
import numpy as np
import matplotlib.pyplot as plt
from gpaw.tddft.spectrum import rotatory_strength_spectrum

def build_spectrum(mm_x, mm_y, mm_z, out_dat, width, e_min, e_max, delta_e):
    rotatory_strength_spectrum([mm_x, mm_y, mm_z],
                               out_dat,
                               folding='Gauss', width=width,
                               e_min=e_min, e_max=e_max, delta_e=delta_e)
    data = np.loadtxt(out_dat)
    return data[:,0], data[:,1]  # energy (eV), R(ω)

def main(prefix, gauge, origin, outdir, width, e_min, e_max, delta_e, png):
    os.makedirs(outdir, exist_ok=True)
    mm_x = os.path.join(outdir, f'mm-{origin}-{gauge}_x.dat')
    mm_y = os.path.join(outdir, f'mm-{origin}-{gauge}_y.dat')
    mm_z = os.path.join(outdir, f'mm-{origin}-{gauge}_z.dat')

    out_dat = os.path.join(outdir, f'rot_spec-{origin}-{gauge}.dat')
    e, R = build_spectrum(mm_x, mm_y, mm_z, out_dat, width, e_min, e_max, delta_e)

    # 绘图（matplotlib）
    plt.figure(figsize=(6,4))
    plt.plot(e, R, lw=1.5)
    plt.xlabel('Energy (eV)')
    plt.ylabel('Rotatory strength (R)')
    plt.title(f'ECD: {prefix} [{gauge}, origin={origin}]')
    plt.grid(True, ls='--', alpha=0.4)
    png_path = os.path.join(outdir, png or f'rot_spec-{origin}-{gauge}.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    print('Spectrum data:', out_dat)
    print('Figure saved :', png_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="system", help="作图标题前缀 (title prefix)")
    ap.add_argument("--gauge", default="length", choices=["length","velocity"])
    ap.add_argument("--origin", default="COM")
    ap.add_argument("--outdir", default="rt_out")
    ap.add_argument("--width", type=float, default=0.2)
    ap.add_argument("--e_min", type=float, default=0.0)
    ap.add_argument("--e_max", type=float, default=10.0)
    ap.add_argument("--delta_e", type=float, default=0.01)
    ap.add_argument("--png", default=None)
    args = ap.parse_args()
    main(args.prefix, args.gauge, args.origin, args.outdir,
         args.width, args.e_min, args.e_max, args.delta_e, args.png)

