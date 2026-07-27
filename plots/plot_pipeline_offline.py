"""
Pipeline offline: señal cruda → Butterworth LP → envolvente para onset detection.
Solo el camino offline (filtfilt, zero-phase). Sin RMS.
Uso: python plots/plot_pipeline_offline.py
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

CSV   = "data/raw/P01/P01_S03_20260525_194322.csv"
TASK  = "fast_flex"
SEG_S = 5.0
T_OFFSET_S = 2.0

df = pd.read_csv(CSV)
df["t"] = df["t_micros"] / 1e6
df["t"] -= df["t"].iloc[0]
fs = 1.0 / np.median(np.diff(df["t"].values))

block = df[df["task"] == TASK]
t0  = block["t"].iloc[0] + T_OFFSET_S
seg = df[(df["t"] >= t0) & (df["t"] < t0 + SEG_S)].copy()
t   = (seg["t"].values - seg["t"].values[0]) * 1000  # ms

x      = seg["extensor"].values.astype(float)
x_dc   = x - x.mean()

sos_bp = signal.butter(4, [70, 300], btype="bandpass", fs=fs, output="sos")
x_bp   = signal.sosfiltfilt(sos_bp, x_dc)

x_rect = np.abs(x_bp)

sos_lp   = signal.butter(2, 1.0, btype="lowpass", fs=fs, output="sos")
x_butter = signal.sosfiltfilt(sos_lp, x_rect)

stages = [
    ("Señal cruda (DC removido)",                     x_dc,      "steelblue"),
    ("Bandpass 70–300 Hz  (offline, zero-phase)",     x_bp,      "steelblue"),
    ("|Rectificada|",                                 x_rect,    "steelblue"),
    ("Envolvente Butterworth LP 1 Hz\n(offline, filtfilt → onset detection)", x_butter, "mediumpurple"),
]

fig, axes = plt.subplots(len(stages), 1, figsize=(12, 8), sharex=True)
fig.suptitle("Pipeline offline — extensor digitorum  (S03, fast_flex)", fontsize=12)

for ax, (name, data, color) in zip(axes, stages):
    ax.plot(t, data, color=color, lw=1.0, alpha=0.9)
    ax.set_ylabel(name, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

axes[-1].set_xlabel("Tiempo (ms)", fontsize=10)

plt.tight_layout()
plt.savefig("plots/pipeline_offline.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/pipeline_offline.png")
plt.show()
