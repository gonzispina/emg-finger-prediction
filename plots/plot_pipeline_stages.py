"""
Comparación Arduino (S03, 10-bit) vs ADS1256 (S04, 24-bit) y
pipeline de procesamiento EMG etapa por etapa para ambas señales.

Genera:
  1. arduino_vs_ads_raw.png   — señal cruda: resolución 10-bit vs 24-bit
  2. pipeline_stages.png      — cada etapa del pipeline para ambas señales
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

CSV_S03 = "data/raw/P01/P01_S03_20260525_194322.csv"   # Arduino 10-bit
CSV_S04 = "data/raw/P01/P01_S04_20260618_162615.csv"   # ADS1256 24-bit

TASK_S03 = "fast_ext"
TASK_S04 = "fast_ext_ind"
SEG_S = 4.0          # segundos de señal a mostrar
BP_LOW, BP_HIGH = 70, 300
RMS_FAST_MS = 40
EMA_TAU_S = 1.0


def load_task_seg(path, task, emg_col, t_seconds, t_offset_s=2.0):
    df = pd.read_csv(path)
    df["t"] = df["t_micros"] / 1e6
    df["t"] -= df["t"].iloc[0]
    fs = 1.0 / np.median(np.diff(df["t"].values))
    block = df[df["task"] == task]
    t0 = block["t"].iloc[0] + t_offset_s
    seg = df[(df["t"] >= t0) & (df["t"] < t0 + t_seconds)].copy()
    x = seg[emg_col].values.astype(float)
    t = (seg["t"].values - seg["t"].values[0]) * 1000  # ms
    return x, t, fs


def pipeline(x, fs):
    """Devuelve cada etapa del pipeline como dict."""
    x_dc = x - x.mean()

    sos_bp = signal.butter(4, [BP_LOW, BP_HIGH], btype="bandpass", fs=fs, output="sos")
    x_bp = signal.sosfiltfilt(sos_bp, x_dc)

    x_rect = np.abs(x_bp)

    def rms(x, ms):
        w = max(1, int(fs * ms / 1000))
        cs = np.cumsum(x ** 2)
        avg = np.empty_like(cs)
        avg[:w] = cs[:w] / np.arange(1, w + 1)
        avg[w:] = (cs[w:] - cs[:-w]) / w
        return np.sqrt(avg)

    x_rms_fast = rms(x_rect, RMS_FAST_MS)
    x_rms_slow = rms(x_rect, 150)

    return {
        "DC removido":              x_dc,
        f"Bandpass {BP_LOW}–{BP_HIGH} Hz": x_bp,
        "|Rectificada|":            x_rect,
        f"RMS rápido ({RMS_FAST_MS} ms)": x_rms_fast,
        "RMS lento (150 ms)":       x_rms_slow,
    }


# ── Cargar segmentos ──────────────────────────────────────────────────────────
x_ard, t_ard, fs_ard = load_task_seg(CSV_S03, TASK_S03, "extensor",  SEG_S)
x_ads, t_ads, fs_ads = load_task_seg(CSV_S04, TASK_S04, "emg0",      SEG_S)

print(f"Arduino: fs={fs_ard:.1f} Hz | {len(x_ard)} samples | range=[{x_ard.min():.0f}, {x_ard.max():.0f}]")
print(f"ADS1256: fs={fs_ads:.1f} Hz | {len(x_ads)} samples | range=[{x_ads.min():.0f}, {x_ads.max():.0f}]")

# ── FIGURA 1: señal cruda — resolución ───────────────────────────────────────
# Zoom en 200ms para que se vea la cuantización del Arduino
ZOOM_MS = 200
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
fig.suptitle("Señal EMG cruda: Arduino Uno (10-bit) vs ADS1256 (24-bit)", fontsize=13)

for row, (x, t, fs, label, color, bits, levels) in enumerate([
    (x_ard, t_ard, fs_ard, "Arduino Uno  10-bit  ~605 Hz", "steelblue",  10, 1_024),
    (x_ads, t_ads, fs_ads, "ADS1256       24-bit  ~638 Hz", "darkorange", 24, 16_777_216),
]):
    x_dc = x - x.mean()
    # Normalizar a z-score para comparación visual de amplitud
    x_z = x_dc / (x_dc.std() + 1e-9)

    # Panel izquierdo: señal completa normalizada
    axes[row, 0].plot(t, x_z, color=color, lw=0.8)
    axes[row, 0].set_title(f"{label} — {SEG_S:.0f}s  ({bits} bits = {levels:,} niveles)", fontsize=9)
    axes[row, 0].set_ylabel("Amplitud (z-score)")
    axes[row, 0].grid(True, alpha=0.3)
    if row == 1:
        axes[row, 0].set_xlabel("Tiempo (ms)")

    # Panel derecho: zoom en escala original para ver cuantización
    zoom_mask = t <= ZOOM_MS
    axes[row, 1].plot(t[zoom_mask], x_dc[zoom_mask], color=color, lw=1.2, marker=".", markersize=3)
    axes[row, 1].set_title(f"Zoom {ZOOM_MS}ms — escala ADC original", fontsize=9)
    axes[row, 1].set_ylabel("ADC (DC removido)")
    axes[row, 1].grid(True, alpha=0.3)
    if row == 1:
        axes[row, 1].set_xlabel("Tiempo (ms)")

plt.tight_layout()
plt.savefig("plots/arduino_vs_ads_raw.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/arduino_vs_ads_raw.png")
plt.close()

# ── FIGURA 2: pipeline por etapas ─────────────────────────────────────────────
stages_ard = pipeline(x_ard, fs_ard)
stages_ads = pipeline(x_ads, fs_ads)
stage_names = list(stages_ard.keys())

fig, axes = plt.subplots(len(stage_names), 2, figsize=(14, 2.5 * len(stage_names)))
fig.suptitle("Pipeline de procesamiento EMG — etapa por etapa\n"
             "Izquierda: Arduino 10-bit (S03)  |  Derecha: ADS1256 24-bit (S04)",
             fontsize=12)

for row, name in enumerate(stage_names):
    for col, (x_stage, t_s, color, label) in enumerate([
        (stages_ard[name], t_ard, "steelblue",  "Arduino"),
        (stages_ads[name], t_ads, "darkorange",  "ADS1256"),
    ]):
        ax = axes[row, col]
        ax.plot(t_s, x_stage, color=color, lw=0.8, alpha=0.9)
        ax.set_ylabel(name, fontsize=8)
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.set_title(label, fontsize=10)
        if row == len(stage_names) - 1:
            ax.set_xlabel("Tiempo (ms)")

plt.tight_layout()
plt.savefig("plots/pipeline_stages.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/pipeline_stages.png")
plt.close()
