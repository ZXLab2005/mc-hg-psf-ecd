# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================================================
# Publication-style plotting
# =========================================================
rcParams["font.family"] = "DejaVu Sans"
rcParams["font.size"] = 16
rcParams["axes.linewidth"] = 2.0
rcParams["axes.labelweight"] = "bold"
rcParams["xtick.major.width"] = 1.8
rcParams["ytick.major.width"] = 1.8
rcParams["xtick.major.size"] = 7
rcParams["ytick.major.size"] = 7
rcParams["xtick.direction"] = "in"
rcParams["ytick.direction"] = "in"
rcParams["legend.frameon"] = False

# =========================================================
# User parameters
# =========================================================
CSV_X = "traj_mu_m_x.csv"
CSV_Y = "traj_mu_m_y.csv"
CSV_Z = "traj_mu_m_z.csv"

TARGETS = [3.30, 3.70]
WINDOW = 0.50
ZERO_PAD = 8
USE_WINDOW = True

OUT1 = "Ag4_local_mixed_3.30eV_notitle.png"
OUT2 = "Ag4_local_mixed_3.70eV_notitle.png"

# =========================================================
# Constant
# =========================================================
H_EV_FS = 4.135667696  # eV*fs

# =========================================================
# Helper functions
# =========================================================
def fft_complex(y, dt_fs, zero_pad=1, use_window=True, remove_mean=True):
    y = np.asarray(y, dtype=float).copy()
    if remove_mean:
        y = y - np.mean(y)
    if use_window:
        y = y * np.hanning(len(y))
    n = len(y)
    nfft = int(2 ** np.ceil(np.log2(n * zero_pad)))
    spec = np.fft.rfft(y, n=nfft)
    freq = np.fft.rfftfreq(nfft, d=dt_fs)
    energy = freq * H_EV_FS
    return energy, spec

def nearest_idx(x, x0):
    return int(np.argmin(np.abs(x - x0)))

def style_axes_box(ax):
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for side in ["left", "bottom", "top", "right"]:
        ax.spines[side].set_linewidth(2.0)

    ax.tick_params(axis="both", which="major",
                   direction="in", length=7, width=1.8, labelsize=15)
    ax.tick_params(axis="both", which="minor",
                   direction="in", length=4, width=1.4)

def make_single_panel(E, mixed_true_plot, mixed_pred_plot,
                      mixed_total_true_plot, mixed_total_pred_plot,
                      center, outfile):
    left = center - WINDOW
    right = center + WINDOW
    mask = (E >= left) & (E <= right)

    colors = {
        "x": "#d95f5f",
        "y": "#4c78a8",
        "z": "#54a24b",
        "tot": "#111111"
    }

    fig, ax = plt.subplots(figsize=(8.6, 6.2))

    # true
    ax.plot(E[mask], mixed_true_plot["x"][mask], color=colors["x"], lw=3.2, label="x true")
    ax.plot(E[mask], mixed_true_plot["y"][mask], color=colors["y"], lw=3.2, label="y true")
    ax.plot(E[mask], mixed_true_plot["z"][mask], color=colors["z"], lw=3.2, label="z true")

    # pred
    ax.plot(E[mask], mixed_pred_plot["x"][mask], color=colors["x"], lw=3.2, ls="--", label="x pred")
    ax.plot(E[mask], mixed_pred_plot["y"][mask], color=colors["y"], lw=3.2, ls="--", label="y pred")
    ax.plot(E[mask], mixed_pred_plot["z"][mask], color=colors["z"], lw=3.2, ls="--", label="z pred")

    # total
    ax.plot(E[mask], mixed_total_true_plot[mask], color=colors["tot"], lw=3.8, label="total true")
    ax.plot(E[mask], mixed_total_pred_plot[mask], color=colors["tot"], lw=3.4, ls="--", label="total pred")

    # reference lines
    ax.axvline(center, color="0.45", lw=1.8, ls=(0, (2, 2)))
    ax.axhline(0.0, color="0.2", lw=1.5)

    ax.set_xlim(left, right)
    ax.grid(alpha=0.18, linewidth=1.0)

    ax.set_xlabel("Energy (eV)", fontsize=22, fontweight="bold", labelpad=10)
    ax.set_ylabel(r"Im[$\mu(\omega)m^*(\omega)$] (normalized)",
                  fontsize=22, fontweight="bold", labelpad=12)

    ymin, ymax = ax.get_ylim()

    if abs(center - 3.70) < 1e-6:
        y_text = ymax - 0.40 * (ymax - ymin)
    elif abs(center - 3.30) < 1e-6:
        y_text = ymax - 0.40 * (ymax - ymin)
    else:
        y_text = ymax - 0.40 * (ymax - ymin)

    ax.text(center, y_text, f"{center:.2f} eV",
            ha="center", va="top", fontsize=18, fontweight="bold", color="0.35")
    # 图例放图外上方
    ax.legend(loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=4,
              fontsize=14,
              handlelength=2.4,
              columnspacing=1.4,
              borderaxespad=0.0)

    style_axes_box(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(outfile, dpi=500, bbox_inches="tight")
    plt.show()

# =========================================================
# Read CSVs
# =========================================================
fx = pd.read_csv(CSV_X)
fy = pd.read_csv(CSV_Y)
fz = pd.read_csv(CSV_Z)

t = fx["time_fs"].values
dt = np.median(np.diff(t))

mu_true = {
    "x": fx["mu_x_true"].values,
    "y": fy["mu_y_true"].values,
    "z": fz["mu_z_true"].values,
}
mu_pred = {
    "x": fx["mu_x_pred"].values,
    "y": fy["mu_y_pred"].values,
    "z": fz["mu_z_pred"].values,
}

m_true = {
    "x": fx["m_x_true"].values,
    "y": fy["m_y_true"].values,
    "z": fz["m_z_true"].values,
}
m_pred = {
    "x": fx["m_x_pred"].values,
    "y": fy["m_y_pred"].values,
    "z": fz["m_z_pred"].values,
}

# =========================================================
# FFT
# =========================================================
mu_fft_true, mu_fft_pred = {}, {}
m_fft_true, m_fft_pred = {}, {}

for comp in ["x", "y", "z"]:
    E_t, MU_t = fft_complex(mu_true[comp], dt, zero_pad=ZERO_PAD, use_window=USE_WINDOW)
    E_p, MU_p = fft_complex(mu_pred[comp], dt, zero_pad=ZERO_PAD, use_window=USE_WINDOW)
    E_mt, M_t = fft_complex(m_true[comp], dt, zero_pad=ZERO_PAD, use_window=USE_WINDOW)
    E_mp, M_p = fft_complex(m_pred[comp], dt, zero_pad=ZERO_PAD, use_window=USE_WINDOW)

    mu_fft_true[comp] = (E_t, MU_t)
    mu_fft_pred[comp] = (E_p, MU_p)
    m_fft_true[comp] = (E_mt, M_t)
    m_fft_pred[comp] = (E_mp, M_p)

E = E_t

# =========================================================
# Mixed-response proxy
# =========================================================
mixed_true = {}
mixed_pred = {}

for comp in ["x", "y", "z"]:
    mixed_true[comp] = -np.imag(mu_fft_true[comp][1] * np.conj(m_fft_true[comp][1]))
    mixed_pred[comp] = -np.imag(mu_fft_pred[comp][1] * np.conj(m_fft_pred[comp][1]))

mixed_total_true = mixed_true["x"] + mixed_true["y"] + mixed_true["z"]
mixed_total_pred = mixed_pred["x"] + mixed_pred["y"] + mixed_pred["z"]

scale = np.max(np.abs(mixed_total_true))
if scale == 0:
    scale = 1.0

mixed_true_plot = {k: v / scale for k, v in mixed_true.items()}
mixed_pred_plot = {k: v / scale for k, v in mixed_pred.items()}
mixed_total_true_plot = mixed_total_true / scale
mixed_total_pred_plot = mixed_total_pred / scale

# =========================================================
# Diagnostics
# =========================================================
print("\n=== Local mixed-response diagnostics (no-title version) ===")
for e0 in TARGETS:
    i = nearest_idx(E, e0)
    print(f"\nTarget peak: {e0:.2f} eV | nearest FFT grid: {E[i]:.4f} eV")
    for comp in ["x", "y", "z"]:
        print(f"  {comp}-true = {mixed_true_plot[comp][i]: .6f} | {comp}-pred = {mixed_pred_plot[comp][i]: .6f}")
    print(f"  total-true = {mixed_total_true_plot[i]: .6f} | total-pred = {mixed_total_pred_plot[i]: .6f}")

# =========================================================
# Generate two separate figures
# =========================================================
make_single_panel(
    E, mixed_true_plot, mixed_pred_plot,
    mixed_total_true_plot, mixed_total_pred_plot,
    center=3.30, outfile=OUT1
)

make_single_panel(
    E, mixed_true_plot, mixed_pred_plot,
    mixed_total_true_plot, mixed_total_pred_plot,
    center=3.70, outfile=OUT2
)
