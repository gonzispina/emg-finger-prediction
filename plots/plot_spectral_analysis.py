"""
Análisis espectral comparativo: Arduino (S03) vs ADS1256 (S04).
Objetivo: identificar componentes espectrales del ruido espurio del ADS1256.

Genera:
  1. spectral_comparison.png  — PSD Arduino vs ADS1256 en reposo y en movimiento
  2. spectrogram_ads.png      — espectrograma ADS1256 para ver si el ruido es estacionario
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

CSV_S03 = "data/raw/P01/P01_S03_20260525_194322.csv"
CSV_S04 = "data/raw/P01/P01_S04_20260618_162615.csv"


def load(path, emg_col, task=None, duration_s=30.0, offset_s=5.0):
    df = pd.read_csv(path)
    df["t"] = df["t_micros"] / 1e6
    df["t"] -= df["t"].iloc[0]
    fs = 1.0 / np.median(np.diff(df["t"].values))
    if task:
        block = df[df["task"] == task]
    else:
        block = df
    t0 = block["t"].iloc[0] + offset_s
    seg = df[(df["t"] >= t0) & (df["t"] < t0 + duration_s)]
    x = seg[emg_col].values.astype(float)
    x -= x.mean()
    return x, fs


# ── Cargar: reposo y movimiento para ambos dispositivos ───────────────────────
x_ard_rest, fs_ard = load(CSV_S03, "extensor",  task="rest_baseline",  duration_s=30)
x_ads_rest, fs_ads = load(CSV_S04, "emg0",       task="rest_baseline",  duration_s=30)
x_ard_move, _      = load(CSV_S03, "extensor",  task="fast_ext",       duration_s=20)
x_ads_move, _      = load(CSV_S04, "emg0",       task="fast_ext_ind",   duration_s=20)

print(f"Arduino: fs={fs_ard:.1f} Hz")
print(f"ADS1256: fs={fs_ads:.1f} Hz")


# ── FIGURA 1: PSD comparativa ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle("Análisis espectral: Arduino (10-bit) vs ADS1256 (24-bit)", fontsize=13)

pairs = [
    (axes[0, 0], axes[0, 1], x_ard_rest, x_ads_rest, "Reposo (rest_baseline)"),
    (axes[1, 0], axes[1, 1], x_ard_move, x_ads_move, "Movimiento (fast_ext)"),
]

for ax_ard, ax_ads, x_a, x_d, title in pairs:
    for ax, x, fs, label, color in [
        (ax_ard, x_a, fs_ard, "Arduino 10-bit", "steelblue"),
        (ax_ads, x_d, fs_ads, "ADS1256 24-bit", "darkorange"),
    ]:
        f, psd = signal.welch(x, fs=fs, nperseg=int(fs * 4), scaling="density")
        psd_db = 10 * np.log10(psd + 1e-20)
        ax.plot(f, psd_db, color=color, lw=1.2, label=label)

        # Marcar armónicos de 50 Hz
        for h in np.arange(50, fs/2, 50):
            ax.axvline(h, color="red", lw=0.6, ls="--", alpha=0.4,
                       label="50 Hz armónicos" if h == 50 else "")

        # Banda EMG
        ax.axvspan(70, 300, alpha=0.07, color="green",
                   label="Banda EMG (70–300 Hz)" if ax == ax_ard else "")

        ax.set_xlim(0, fs/2)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
        ax.set_ylabel("PSD (dB/Hz)", fontsize=9)
        ax.set_title(f"{label} — {title}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("plots/spectral_comparison.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/spectral_comparison.png")
plt.close()


# ── FIGURA 2: espectrograma ADS1256 en reposo ──────────────────────────────────
# Para ver si el ruido es estacionario o aparece en bursts
fig, axes = plt.subplots(3, 1, figsize=(13, 9))
fig.suptitle("Espectrograma ADS1256 — reposo (rest_baseline)\n"
             "¿El ruido espurio es estacionario o aparece en bursts?", fontsize=12)

# Señal temporal
t_axis = np.arange(len(x_ads_rest)) / fs_ads
axes[0].plot(t_axis, x_ads_rest, color="darkorange", lw=0.6)
axes[0].set_ylabel("ADC (DC removido)")
axes[0].set_title("Señal temporal")
axes[0].grid(True, alpha=0.3)

# Espectrograma
f_sg, t_sg, Sxx = signal.spectrogram(x_ads_rest, fs=fs_ads,
                                      nperseg=int(fs_ads * 0.5),
                                      noverlap=int(fs_ads * 0.4),
                                      scaling="density")
Sxx_db = 10 * np.log10(Sxx + 1e-20)
im = axes[1].pcolormesh(t_sg, f_sg, Sxx_db, shading="gouraud",
                         cmap="inferno", vmin=np.percentile(Sxx_db, 20))
axes[1].set_ylabel("Frecuencia (Hz)")
axes[1].set_title("Espectrograma")
axes[1].set_ylim(0, fs_ads/2)
plt.colorbar(im, ax=axes[1], label="dB/Hz")
for h in np.arange(50, fs_ads/2, 50):
    axes[1].axhline(h, color="cyan", lw=0.5, ls="--", alpha=0.5)

# PSD promedio
f_psd, psd = signal.welch(x_ads_rest, fs=fs_ads, nperseg=int(fs_ads * 4))
axes[2].semilogy(f_psd, psd, color="darkorange", lw=1.5)
axes[2].axvspan(70, 300, alpha=0.1, color="green", label="Banda EMG")
for h in np.arange(50, fs_ads/2, 50):
    axes[2].axvline(h, color="red", lw=0.7, ls="--", alpha=0.5,
                    label="50 Hz armónicos" if h == 50 else "")
axes[2].set_xlabel("Frecuencia (Hz)")
axes[2].set_ylabel("PSD (V²/Hz)")
axes[2].set_title("PSD promedio")
axes[2].set_xlim(0, fs_ads/2)
axes[2].legend(fontsize=9)
axes[2].grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("plots/spectrogram_ads.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/spectrogram_ads.png")
plt.close()
