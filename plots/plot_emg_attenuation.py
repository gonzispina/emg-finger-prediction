"""
Visualizacion de la atenuacion de señal EMG.
Muestra RMS flexor original vs comprimido (alpha=0.4) sobre un segmento de fast_flex_ind.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

CSV = "data/raw/P01/P01_S02_20260525_184004.csv"
ALPHA = 0.4   # mismo que champion
PLOT_S = 8.0  # segundos a mostrar

# --- cargar datos ---
import pandas as pd
df  = pd.read_csv(CSV)
fs  = float(len(df) / (df["t_micros"].iloc[-1] - df["t_micros"].iloc[0]) * 1e6)
emg = df["emg1"].to_numpy(dtype=float)   # emg1 = flexor
tasks = df["task"].to_numpy()

# --- pipeline RMS (igual que training) ---
def causal_bandpass(x, fs, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def causal_rms(x, fs, win_ms=40.0):
    n = max(1, int(round(win_ms * 1e-3 * fs)))
    x2 = x ** 2
    kernel = np.ones(n) / n
    return np.sqrt(np.convolve(x2, kernel, mode="same"))

bp  = causal_bandpass(emg, fs, 70.0, 300.0)
rms = causal_rms(bp, fs, 40.0)

# baseline: mediana de los ultimos 10s de rest_baseline (como en training)
rest_idx = np.where(tasks == "rest_baseline")[0]
last10 = rest_idx[rest_idx >= rest_idx[-1] - int(10.0 * fs)]
baseline_val = float(np.median(rms[last10]))

# atenuacion: comprimir picos sobre el baseline
rms_attn = np.where(
    rms > baseline_val,
    baseline_val + (rms - baseline_val) * ALPHA,
    rms
)

# --- segmento: fast_flex_ind ---
task_name = "fast_flex_ind" if "fast_flex_ind" in tasks else "fast_flex"
idx = np.where(tasks == task_name)[0]
i0 = idx[0]
i1 = min(i0 + int(PLOT_S * fs), idx[-1] + 1)
sl = slice(i0, i1)
tr = (df["t_micros"].to_numpy()[sl] - df["t_micros"].iloc[i0]) / 1e6

# --- plot ---
fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(tr, rms[sl],      color="steelblue", lw=1.8, label="RMS flexor (original)",   alpha=0.9)
ax.plot(tr, rms_attn[sl], color="tomato",    lw=1.8, label=f"RMS flexor comprimido  (α={ALPHA})", ls="--")
ax.axhline(baseline_val,  color="gray",      lw=1.2, ls=":", label="baseline reposo")

ax.fill_between(tr, rms_attn[sl], rms[sl],
                where=(rms[sl] > baseline_val),
                color="orange", alpha=0.2, label="zona comprimida")

ax.set_xlabel("Tiempo (s)", fontsize=11)
ax.set_ylabel("RMS EMG (ADC)", fontsize=11)
ax.set_title("Atenuación de señal EMG flexor — fast_flex  (S02)", fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/emg_attenuation.png", dpi=150, bbox_inches="tight")
print(f"Guardado: plots/emg_attenuation.png  (baseline={baseline_val:.1f} ADC, α={ALPHA})")
plt.show()
