"""
PSD de la señal EMG cruda mostrando el ruido de red eléctrica.
Usa un CSV del ESP32 (formato binary_v2).
Simula dos condiciones comparando un segmento con armonicos visibles
y uno más limpio (o con la señal filtrada para mostrar el efecto).

Uso:
    python plots/plot_fft_noise.py --csv data/raw/P01/P01_S06_20260620_113004.csv
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--channel", default="emg0", help="columna EMG a analizar")
    p.add_argument("--segment-s", type=float, default=10.0, help="segundos a usar")
    p.add_argument("--offset-s",  type=float, default=5.0,  help="offset inicial en segundos")
    return p.parse_args()

def load_csv(path):
    df = pd.read_csv(path)
    # Detectar formato (binary_v2 tiene t_micros, formato viejo tiene t_ms)
    if "t_micros" in df.columns:
        df["t"] = df["t_micros"] / 1e6
    elif "t_ms" in df.columns:
        df["t"] = df["t_ms"] / 1e3
    return df

def estimate_fs(df):
    dt = np.diff(df["t"].values)
    return 1.0 / np.median(dt)

def main():
    args = parse_args()
    df = load_csv(args.csv)
    fs = estimate_fs(df)
    print(f"fs estimada: {fs:.1f} Hz")

    # Segmento de análisis
    t0 = args.offset_s
    t1 = t0 + args.segment_s
    seg = df[(df["t"] >= t0) & (df["t"] < t1)]
    x = seg[args.channel].values.astype(np.float64)
    x -= x.mean()

    # Señal filtrada (Butterworth 70-300 Hz)
    b, a = signal.butter(4, [70, 300], btype="bandpass", fs=fs)
    x_filt = signal.filtfilt(b, a, x)

    # PSD con Welch
    nperseg = int(fs * 2)
    f_raw,  psd_raw  = signal.welch(x,      fs=fs, nperseg=nperseg)
    f_filt, psd_filt = signal.welch(x_filt, fs=fs, nperseg=nperseg)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"PSD de señal EMG — canal {args.channel}\n"
                 f"Segmento {t0:.1f}–{t1:.1f}s  |  fs={fs:.0f} Hz", fontsize=12)

    for ax, f, psd, label, color in [
        (axes[0], f_raw,  psd_raw,  "Señal cruda",   "steelblue"),
        (axes[1], f_filt, psd_filt, "Señal filtrada (70–300 Hz)", "darkorange"),
    ]:
        ax.semilogy(f, psd, color=color, lw=1.2, label=label)
        # Marcar armónicos de red
        for harm in [50, 100, 150, 200, 250]:
            ax.axvline(harm, color="red", lw=0.8, ls="--", alpha=0.6,
                       label="50 Hz + armónicos" if harm == 50 else "")
        ax.axvspan(70, 300, alpha=0.08, color="green", label="Banda EMG (70–300 Hz)")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("PSD (V²/Hz)")
        ax.set_xlim(0, fs / 2)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("plots/fft_noise.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/fft_noise.png")
    plt.show()

if __name__ == "__main__":
    main()
