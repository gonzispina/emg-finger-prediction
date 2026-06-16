"""
Visualizacion offline de un CSV capturado con capture_raw.py.

Genera un panel con:
  - Senal raw de cada EMG (top: 4 traces).
  - Senal filtrada (bandpass causal).
  - FFT (magnitud, log scale) — para identificar peaks de ruido (50/60 Hz,
    harmonics, switching).
  - Joysticks crudos.

El proposito es diagnostico: ver donde esta el ruido, si la senal muscular
real esta ahi (oculta por DC offset o picos), y a que frecuencia.

Uso:
    python plot_raw.py data/debug/test_001.csv
    python plot_raw.py data/debug/test_001.csv --bp 20 250
    python plot_raw.py data/debug/test_001.csv --emg-only emg0
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--bp", type=float, nargs=2, default=[70.0, 300.0],
                   metavar=("LOW", "HIGH"),
                   help="cutoffs del bandpass (Hz). Default 70-300 (estandar EMG).")
    p.add_argument("--notch-base", type=float, default=50.0,
                   help="frecuencia base del notch (50 en EU/AR, 60 en US). "
                        "Se aplican notches en base, 2*base, 3*base... hasta Nyquist. "
                        "Usar 0 para desactivar.")
    p.add_argument("--notch-q", type=float, default=30.0,
                   help="Q del notch (mayor = mas estrecho). Default 30.")
    p.add_argument("--fft-xmax", type=float, default=1000.0,
                   help="frecuencia maxima a mostrar en los paneles FFT (Hz). "
                        "Lo real esta acotado por Nyquist=fs/2, mas alla queda en blanco.")
    p.add_argument("--bp-order", type=int, default=4)
    p.add_argument("--emg-only", default=None,
                   help="ploetar solo un EMG (emg0 / emg1 / emg2 / emg3)")
    p.add_argument("--out", type=Path, default=None,
                   help="path para guardar PNG en vez de mostrar")
    return p.parse_args()


def detect_fs(t_micros: np.ndarray) -> float:
    """fs efectiva a partir del t_micros del MCU."""
    dt = np.diff(t_micros) * 1e-6
    dt = dt[(dt > 0) & (dt < 0.5)]  # filtrar outliers (resets, etc)
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int):
    safe_high = min(high, 0.45 * fs)
    if safe_high <= low + 1:
        return x
    sos = butter(order, [low, safe_high], btype="band", fs=fs, output="sos")
    sig_dc = x.mean()
    sig_ac = x - sig_dc
    zi = sosfilt_zi(sos) * sig_ac[0]
    y, _ = sosfilt(sos, sig_ac, zi=zi)
    return y


def apply_notches(x: np.ndarray, fs: float, base: float, q: float):
    """Aplica notches en base, 2*base, 3*base, ... hasta Nyquist."""
    if base <= 0:
        return x
    nyq = fs / 2
    y = x.astype(np.float64).copy()
    f = base
    while f < nyq * 0.95:
        b, a = iirnotch(f, q, fs=fs)
        sos = tf2sos(b, a)
        zi = sosfilt_zi(sos) * y[0]
        y, _ = sosfilt(sos, y, zi=zi)
        f += base
    return y


def fft_magnitude(x: np.ndarray, fs: float):
    n = len(x)
    if n == 0:
        return np.array([]), np.array([])
    # ventana Hann para reducir leakage
    win = np.hanning(n)
    X = np.fft.rfft((x - x.mean()) * win)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(X) / n * 2
    return freqs, mag


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    fs = detect_fs(df["t_micros"].to_numpy())
    print(f"CSV: {args.csv}  samples={len(df)}  fs_efectiva={fs:.1f} Hz")

    emg_cols = ["emg0", "emg1", "emg2", "emg3"]
    if args.emg_only:
        emg_cols = [args.emg_only]

    n_emg = len(emg_cols)
    # Estructura: por cada EMG: raw + filtered. Despues 2 paneles de FFT
    # (raw y filtered, agregando todos los EMG) y por ultimo los joysticks.
    n_rows = n_emg * 2 + 2 + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(13, 2.0 * n_rows))
    axes = list(axes)

    # Tiempo en segundos desde el primer t_micros
    t0 = float(df["t_micros"].iloc[0])
    t_s = (df["t_micros"].to_numpy() - t0) * 1e-6

    ax_idx = 0
    raw_signals = {}
    filtered_signals = {}

    for col in emg_cols:
        raw = df[col].to_numpy(dtype=np.float64)
        # Pipeline: raw -> notch (50/100/150...) -> bandpass
        notched = apply_notches(raw, fs, args.notch_base, args.notch_q)
        filtered = bandpass(notched, fs, args.bp[0], args.bp[1], args.bp_order)
        raw_signals[col] = raw
        filtered_signals[col] = filtered

        # raw
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(t_s, raw, lw=0.5, color="tab:gray")
        ax.set_title(f"{col} raw  (mean={raw.mean():.0f}, std={raw.std():.0f})",
                     fontsize=9)
        ax.set_ylabel("ADC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

        # post pipeline (notch + bandpass)
        notch_label = (f" + notch {args.notch_base:.0f}+harm Q{args.notch_q:.0f}"
                       if args.notch_base > 0 else "")
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(t_s, filtered, lw=0.5, color="tab:red")
        ax.set_title(f"{col} bandpass {args.bp[0]:.0f}-{args.bp[1]:.0f} Hz{notch_label}  "
                     f"(std={filtered.std():.0f}, p99|·|={np.percentile(np.abs(filtered), 99):.0f})",
                     fontsize=9)
        ax.set_ylabel("ADC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    # FFT pre-bandpass (raw, post-DC removal para que no domine el bin 0).
    # Usado para identificar ruido de linea, harmonics, etc.
    ax = axes[ax_idx]; ax_idx += 1
    for col in emg_cols:
        sig = raw_signals[col]
        if sig.std() < 1e-9:
            continue
        freqs, mag = fft_magnitude(sig, fs)
        ax.semilogy(freqs, mag + 1e-3, lw=0.6, label=col)
    ax.axvspan(args.bp[0], min(args.bp[1], fs / 2), color="tab:red",
               alpha=0.08, label=f"banda {args.bp[0]:.0f}-{args.bp[1]:.0f} Hz")
    # Marcar lineas a 50 y 60 Hz para diagnostico de ruido de red
    for f_line in (50, 60):
        if f_line < fs / 2:
            ax.axvline(f_line, color="orange", lw=0.5, alpha=0.4,
                       linestyle="--")
    nyq = fs / 2
    fft_xmax = max(args.fft_xmax, nyq + 1)
    # Sombra gris despues de Nyquist: ahi no hay datos reales
    if nyq < fft_xmax:
        ax.axvspan(nyq, fft_xmax, color="gray", alpha=0.15)
        ax.text(nyq + (fft_xmax - nyq) * 0.5, ax.get_ylim()[1] * 0.5,
                f"NYQUIST = {nyq:.0f} Hz\n(no hay datos)",
                ha="center", va="center", fontsize=8, color="dimgray")
    ax.set_title(f"FFT pre-bandpass (raw - DC). Vertical: lineas 50/60 Hz. "
                 f"fs={fs:.0f} Hz, Nyquist={nyq:.0f} Hz",
                 fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("|X(f)|", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, fft_xmax)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=len(emg_cols) + 1)

    # FFT post-bandpass
    ax = axes[ax_idx]; ax_idx += 1
    for col in emg_cols:
        if filtered_signals[col].std() < 1e-9:
            continue
        freqs, mag = fft_magnitude(filtered_signals[col], fs)
        ax.semilogy(freqs, mag + 1e-3, lw=0.6, label=col)
    if nyq < fft_xmax:
        ax.axvspan(nyq, fft_xmax, color="gray", alpha=0.15)
    ax.set_title(f"FFT post-bandpass {args.bp[0]:.0f}-{args.bp[1]:.0f} Hz",
                 fontsize=9)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=8)
    ax.set_ylabel("|X(f)|", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, fft_xmax)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=len(emg_cols))

    # Joysticks (los 2 en un mismo axis)
    ax = axes[ax_idx]; ax_idx += 1
    ax.plot(t_s, df["joy_a"].to_numpy(), lw=0.7, color="tab:green", label="joy_a")
    ax.plot(t_s, df["joy_b"].to_numpy(), lw=0.7, color="tab:purple", label="joy_b")
    ax.set_title(f"Joysticks (raw, ADC 12-bit)", fontsize=9)
    ax.set_xlabel("tiempo (s)", fontsize=8)
    ax.set_ylabel("ADC", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"Guardado en {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
