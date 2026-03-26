#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot dipole/magnetic-dipole time series with optional smoothing.
- If --tdout is given (or td-*.out is found), use its time column as the common time axis.
- Robust tdout reader: skips non-numeric lines (e.g., '___', headers).
"""

import argparse, glob, os
import numpy as np
import matplotlib.pyplot as plt

# optional SciPy
try:
    from scipy.signal import savgol_filter, butter, filtfilt
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False
    savgol_filter = butter = filtfilt = None

# -------------------- helpers --------------------
def _find_first(patterns):
    for pat in patterns:
        hits = glob.glob(pat)
        if hits:
            hits.sort(key=lambda p: (len(p), p))
            return hits[0]
    return None

def load_xy(path):
    """Load .dat with 1 or 2 columns; if one column, return (idx, y)."""
    arr = np.loadtxt(path)
    if arr.ndim == 1 or arr.shape[1] == 1:
        t = np.arange(arr.size, dtype=float)
        y = arr.reshape(-1).astype(float)
    else:
        t = arr[:, 0].astype(float)
        y = arr[:, 1].astype(float)
    return t, y

def find_series(kind_prefix, comp):
    """kind_prefix: 'dm' or 'mm' keywords to search"""
    if kind_prefix == "dm":
        pats = [
            f"dm-length_{comp}*.dat", f"dm*_{comp}*.dat", f"*dipole*{comp}*.dat",
            f"*mu*{comp}*.dat"
        ]
    else:
        pats = [
            f"mm-COM-length_{comp}*.dat", f"mm-length_{comp}*.dat",
            f"mm*_{comp}*.dat", f"*mag*{comp}*.dat", f"*mdipole*{comp}*.dat"
        ]
    p = _find_first(pats)
    if not p:
        return None, None, None
    t, y = load_xy(p)
    return t, y, p

def find_tdout(path_hint=None):
    if path_hint and os.path.exists(path_hint):
        return path_hint
    pats = ["td-x.out", "td_y.out", "td-z.out", "td.out", "td-*.out"]
    return _find_first(pats)

def load_td_time_robust(td_path, col=0):
    """
    Robust loader: read lines, split by whitespace, collect float from column `col`.
    Skip lines that cannot be parsed (headers, '___', etc.).
    """
    times = []
    with open(td_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # comment lines to skip quickly
            if s[0] in "#!%;@":
                continue
            parts = s.replace(",", " ").split()
            if len(parts) <= col:
                continue
            try:
                t = float(parts[col])
            except Exception:
                continue  # skip non-numeric lines like '___'
            times.append(t)
    if not times:
        raise ValueError(f"No numeric time values parsed from {td_path}")
    return np.asarray(times, dtype=float)

def resample(t_src, y_src, t_target):
    return np.interp(t_target, t_src, y_src)

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

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tdout", type=str, default=None, help="tdout file for time axis")
    ap.add_argument("--td-col", type=int, default=0, help="which column in tdout is time (0-based)")
    ap.add_argument("--filter", choices=["none","moving","savgol","butter"], default="savgol")
    ap.add_argument("--window", type=int, default=31)
    ap.add_argument("--poly",   type=int, default=3)
    ap.add_argument("--cutoff", type=float, default=0.1)
    ap.add_argument("--order",  type=int, default=4)
    ap.add_argument("--show-raw", dest="show_raw", action="store_true")
    ap.add_argument("--out", default="dm_mm_filtered")
    ap.add_argument("--title-y", type=float, default=0.93)
    args = ap.parse_args()

    comps = ["x","y","z"]

    # ---- collect mm only (你也可以把 dm 部分加回去) ----
    mm = {}
    for c in comps:
        t, y, p = find_series("mm", c)
        if t is None:
            raise SystemExit(f"未找到磁偶极矩 {c} 分量数据（例如 mm-COM-length_{c}.dat）")
        mm[c] = (t, y)

    # ---- time axis from tdout (robust) ----
    td_path = find_tdout(args.tdout)
    if td_path:
        t_common = load_td_time_robust(td_path, col=args.td_col)
        print(f"[INFO] Using tdout time axis from: {td_path} (N={len(t_common)})")
    else:
        # fallback: intersect ranges & use min length
        tmins = [mm[c][0][0] for c in comps]
        tmaxs = [mm[c][0][-1] for c in comps]
        nmins = [len(mm[c][0]) for c in comps]
        tmin, tmax, nmin = max(tmins), min(tmaxs), min(nmins)
        if tmax <= tmin:
            t_common = min([mm[c][0] for c in comps], key=len).copy()
        else:
            t_common = np.linspace(tmin, tmax, nmin)
        print("[WARN] tdout not found; using intersected uniform time axis.")

    # ---- resample & smooth ----
    mm_rs = {c: resample(mm[c][0], mm[c][1], t_common) for c in comps}
    mm_sm = {c: smooth(mm_rs[c], method=args.filter, window=args.window,
                       poly=args.poly, cutoff=args.cutoff, order=args.order) for c in comps}

    # ---- plot 3x1 magnetic dipole only ----
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True)
    colors = {"x":"C1","y":"C2","z":"C3"}
    for ax, c in zip(axes, comps):
        if args.show_raw:
            ax.plot(t_common, mm_rs[c], color="0.75", lw=1.0, label="raw")
        ax.plot(t_common, mm_sm[c], color=colors[c], lw=2.0, label=f"mm_{c}")
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
    print(f"[OK] Saved: {os.path.abspath(png)} and {os.path.abspath(pdf)}")

if __name__ == "__main__":
    main()
