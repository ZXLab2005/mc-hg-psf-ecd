#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Only plot magnetic dipole components (mm_x / mm_y / mm_z) with a FIXED time axis:
t_fixed = np.arange(nsteps) * dt   (default nsteps=2000, dt=1.0)
No tdout reading. Raw curves are mapped from their own index to the fixed axis.

Examples:
  python plot_mm_fixed_time.py --nsteps 2000 --dt 1.0 --filter savgol --window 31 --poly 3 --show-raw
  python plot_mm_fixed_time.py --nsteps 2000 --layout row --filter moving --window 21
"""

import argparse, glob, os
import numpy as np
import matplotlib.pyplot as plt

# optional SciPy for advanced filters
try:
    from scipy.signal import savgol_filter, butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    savgol_filter = butter = filtfilt = None

def find_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort(key=lambda p: (len(p), p))
            return hits[0]
    return None

def load_two_col(path):
    arr = np.loadtxt(path)
    # 允许1列或2列；一列时仅有y
    if arr.ndim == 1 or arr.shape[1] == 1:
        y = np.asarray(arr, dtype=float).reshape(-1)
        t = np.arange(len(y), dtype=float)
    else:
        t = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
    return t, y

def get_mm_component(comp):
    pats = [
        f"mm-COM-length_{comp}*.dat",
        f"mm-length_{comp}*.dat",
        f"mm*_{comp}*.dat",
        f"*mm*{comp}*.dat",
    ]
    p = find_first(pats)
    if not p:
        return None, None, None
    t, y = load_two_col(p)
    return t, y, p

def smooth(y, method="savgol", window=31, poly=3, cutoff=0.1, order=4):
    y = np.asarray(y, float); n = len(y)
    if method == "none": return y
    if method == "moving":
        w = int(max(1, window));  w += (w % 2 == 0)
        k = min(w, n)
        if k < 2: return y.copy()
        pad = k // 2
        ypad = np.pad(y, (pad, pad), mode="reflect")
        kernel = np.ones(k) / k
        return np.convolve(ypad, kernel, mode="valid")
    if method == "savgol":
        if not HAVE_SCIPY:
            return smooth(y, method="moving", window=window)
        w = int(max(3, window));  w += (w % 2 == 0)
        p = int(max(2, poly));    p = min(p, w-1)
        return savgol_filter(y, window_length=w, polyorder=p, mode="mirror")
    if method == "butter":
        if not HAVE_SCIPY:
            return smooth(y, method="moving", window=window)
        b, a = butter(order, cutoff, btype="low")
        return filtfilt(b, a, y, method="gust")
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsteps", type=int, default=2000, help="fixed number of time steps")
    ap.add_argument("--dt", type=float, default=0.01, help="time step size for the fixed axis")
    ap.add_argument("--filter", choices=["none","moving","savgol","butter"], default="savgol")
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--poly",   type=int, default=3)
    ap.add_argument("--cutoff", type=float, default=0.1)
    ap.add_argument("--order",  type=int, default=4)
    ap.add_argument("--show-raw", action="store_true", help="overlay raw (index-based) curve in gray")
    ap.add_argument("--layout", choices=["col","row"], default="col")
    ap.add_argument("--title-y", type=float, default=0.94)
    ap.add_argument("--out", default="mm_fixed_time", help="output prefix (PNG/PDF)")
    args = ap.parse_args()

    # 固定时间轴
    t_fixed = np.arange(args.nsteps, dtype=float) * args.dt

    # 读取 mm 三分量
    comps = ["x","y","z"]
    series = {}
    for c in comps:
        t, y, p = get_mm_component(c)
        if t is None:
            raise SystemExit(f"找不到 mm {c} 分量文件（例如 mm-COM-length_{c}.dat）")
        series[c] = (t, y, p)

    # 把原始序列按其“索引时间”映射到固定时间轴：
    # 如果文件里自带时间列就以其为源时间，否则用 0..len(y)-1
    mm_resampled = {}
    for c in comps:
        t_src, y_src, _ = series[c]
        # 若 t_src 单调且长度与 y_src 一致，则用它；否则退回索引时间
        if len(t_src) != len(y_src) or np.any(np.diff(t_src) <= 0):
            t_src = np.arange(len(y_src), dtype=float)
        # 目标 t_fixed 的范围若超出源范围，np.interp 用端点外推（常值延拓）
        mm_resampled[c] = np.interp(
            np.linspace(t_src[0], t_src[-1], args.nsteps),  # 先在源索引范围等距
            t_src, y_src
        )

    # 平滑
    mm_s = {c: smooth(mm_resampled[c], method=args.filter, window=args.window,
                      poly=args.poly, cutoff=args.cutoff, order=args.order) for c in comps}

    # 画图
    if args.layout == "row":
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)

    colors = {"x":"C1","y":"C2","z":"C3"}
    for ax, c in zip(axes, comps):
        if args.show_raw:
            # 原始曲线按索引时间显示（与固定轴对不齐也无妨，仅作对比）
            _, y_src, _ = series[c]
            ax.plot(np.arange(len(y_src))*args.dt, y_src, color="0.75", lw=1.0, label="raw")
        ax.plot(t_fixed, mm_s[c], color=colors[c], lw=2.0, label=f"mm_{c}")
        ax.set_ylabel("Magnetic dipole")
        ax.set_title(f"{c}-component", pad=6)
        ax.tick_params(direction="in")
        for sp in ax.spines.values(): sp.set_linewidth(1.0)
        ax.legend(frameon=False, loc="upper right")

    axes[-1].set_xlabel("Time")
    fig.suptitle("Magnetic dipole components (filtered)", y=args.title_y)
    plt.subplots_adjust(top=args.title_y + 0.02)
    fig.tight_layout(rect=(0, 0, 1, args.title_y - 0.01))

    png = f"{args.out}.png"; pdf = f"{args.out}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved:", os.path.abspath(png), "and", os.path.abspath(pdf))

if __name__ == "__main__":
    main()
