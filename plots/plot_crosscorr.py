"""
Cross-correlación entre envolvente EMG y joystick.
Muestra el lag óptimo (EMG precede al joystick) y la señal alineada.

Genera:
  1. crosscorr_curve.png  — función de cross-correlación con lag marcado
  2. crosscorr_aligned.png — EMG vs joystick con y sin alineación temporal
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

CSVS = [
    "data/raw/P01/P01_S02_20260525_184004.csv",
    "data/raw/P01/P01_S03_20260525_194322.csv",
]
BP_LOW, BP_HIGH = 70, 300
ENV_LP_HZ = 1.0
SIGMA_JOY_MS = 50
TASKS_USE = ["fast_flex", "fast_ext", "slow_flex", "slow_ext", "free_controlled"]


def load_sessions(paths):
    parts, t_off = [], 0.0
    fs = None
    for p in paths:
        df = pd.read_csv(p)
        df["t"] = df["t_micros"] / 1e6
        df["t"] -= df["t"].iloc[0]
        if "emg0" in df.columns:
            df = df.rename(columns={"emg0": "extensor", "emg1": "flexor"})
            df["y_top"]    = df["joy_ext_ind"]
            df["y_bottom"] = df["joy_flex_ind"]
        _fs = 1.0 / np.median(np.diff(df["t"].values))
        if fs is None:
            fs = _fs
        df["t"] += t_off
        t_off = df["t"].iloc[-1] + 1.0 / fs
        parts.append(df)
    return pd.concat(parts, ignore_index=True), fs


def make_envelope(x, fs, lp_hz=ENV_LP_HZ):
    sos_bp = signal.butter(4, [BP_LOW, BP_HIGH], btype="bandpass", fs=fs, output="sos")
    bp = signal.sosfiltfilt(sos_bp, x)
    sos_lp = signal.butter(2, lp_hz, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos_lp, np.abs(bp))


def cross_corr_lag(a, b, fs, max_lag_s=1.0):
    """Devuelve lags (s), correlaciones normalizadas, y el lag óptimo."""
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    max_lag = int(max_lag_s * fs)
    corrs, lags_s = [], []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            r = np.corrcoef(a[lag:], b[:-lag])[0, 1]
        elif lag < 0:
            r = np.corrcoef(a[:lag], b[-lag:])[0, 1]
        else:
            r = np.corrcoef(a, b)[0, 1]
        corrs.append(r)
        lags_s.append(lag / fs)
    corrs = np.array(corrs)
    lags_s = np.array(lags_s)
    best_lag_s = lags_s[np.argmax(corrs)]
    return lags_s, corrs, best_lag_s


def main():
    df, fs = load_sessions(CSVS)
    print(f"fs={fs:.1f} Hz | N={len(df)}")

    # Filtrar solo tareas de movimiento
    df = df[df["task"].isin(TASKS_USE)].reset_index(drop=True)

    df["extensor"] -= df["extensor"].mean()
    df["flexor"]   -= df["flexor"].mean()

    # Envolvente EMG (actividad neta: ext - flex)
    env_ext = make_envelope(df["extensor"].values, fs)
    env_flx = make_envelope(df["flexor"].values,   fs)
    # Normalizar y combinar: ext - flex ≈ actividad neta del movimiento
    def norm(x): return (x - x.mean()) / (x.std() + 1e-9)
    emg_activity = norm(env_ext) - norm(env_flx)

    # Target joystick combinado
    joy = gaussian_filter1d(
        (-df["y_top"].values  + df["y_top"].mean()) +
        (-df["y_bottom"].values + df["y_bottom"].mean()),
        sigma=SIGMA_JOY_MS / 1000 * fs, mode="nearest"
    )

    # ── FIGURA 1: curva de cross-correlación ──────────────────────────────────
    lags_s, corrs, best_lag = cross_corr_lag(emg_activity, joy, fs, max_lag_s=0.8)
    print(f"Lag óptimo: {best_lag*1000:.1f} ms  (corr={corrs.max():.3f})")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(lags_s * 1000, corrs, color="steelblue", lw=1.8)
    ax.axvline(best_lag * 1000, color="red", lw=2, ls="--",
               label=f"EMG adelanta al joystick en {abs(best_lag)*1000:.0f} ms\n(pico de correlación = {corrs.max():.3f})")
    ax.axvline(0, color="gray", lw=1, ls=":")
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_xlabel("Lag EMG → joystick (ms)\nValores positivos = EMG adelantado", fontsize=11)
    ax.set_ylabel("Correlación de Pearson", fontsize=11)
    ax.set_title("Cross-correlación: envolvente EMG vs posición joystick\n"
                 f"(S02+S03, tareas de movimiento, BP 70-300Hz, LP {ENV_LP_HZ}Hz)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/crosscorr_curve.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/crosscorr_curve.png")
    plt.close()

    # ── FIGURA 2: señal sin alinear vs alineada ───────────────────────────────
    # Segmento representativo de fast_ext
    seg = df[df["task"] == "fast_ext"].iloc[:int(fs * 20)]
    idx = seg.index
    env_seg  = emg_activity[idx[0]:idx[-1]+1]
    joy_seg  = joy[idx[0]:idx[-1]+1]
    t_seg    = (np.arange(len(env_seg)) / fs * 1000)

    lag_samples = int(round(best_lag * fs))
    if lag_samples > 0:
        env_al = env_seg[lag_samples:]
        joy_al = joy_seg[:-lag_samples]
    elif lag_samples < 0:
        env_al = env_seg[:lag_samples]
        joy_al = joy_seg[-lag_samples:]
    else:
        env_al, joy_al = env_seg, joy_seg
    t_al = np.arange(len(env_al)) / fs * 1000

    def n01(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=False)
    fig.suptitle("Envolvente EMG vs joystick — sin y con compensación de lag", fontsize=12)

    ax1.plot(t_seg, n01(env_seg), color="steelblue",  lw=1.5, label="Actividad EMG")
    ax1.plot(t_seg, n01(joy_seg), color="darkorange",  lw=1.5, label="Joystick")
    ax1.set_title(f"Sin compensar  (EMG ≈ {best_lag*1000:.0f}ms adelantado al joystick)")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3); ax1.set_ylabel("Norm. [0-1]")

    ax2.plot(t_al, n01(env_al), color="steelblue",  lw=1.5, label="Actividad EMG (adelantada)")
    ax2.plot(t_al, n01(joy_al), color="darkorange",  lw=1.5, label="Joystick")
    ax2.set_title(f"Con compensación de {best_lag*1000:.0f}ms — señales alineadas")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Norm. [0-1]"); ax2.set_xlabel("Tiempo (ms)")

    plt.tight_layout()
    plt.savefig("plots/crosscorr_aligned.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/crosscorr_aligned.png")
    plt.close()


if __name__ == "__main__":
    main()
