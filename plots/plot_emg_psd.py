"""
PSD de señal EMG limpia mostrando la distribución de energía espectral.
Muestra también la banda de paso del filtro elegido (70-300 Hz).

Uso:
    python plots/plot_emg_psd.py
    python plots/plot_emg_psd.py --csv data/raw/P01/P01_S02_20260525_184004.csv --channel emg0
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

CSV_DEFAULT = "data/raw/P01/P01_S04_20260618_162615.csv"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",     default=CSV_DEFAULT)
    p.add_argument("--channel", default=None, help="columna EMG; autodetecta si se omite")
    p.add_argument("--segment-s", type=float, default=20.0)
    p.add_argument("--offset-s",  type=float, default=10.0)
    return p.parse_args()

def load_seg(path, col, t0, t1):
    df = pd.read_csv(path)
    if "t_micros" in df.columns:
        df["t"] = df["t_micros"] / 1e6
    elif "t_ms" in df.columns:
        df["t"] = df["t_ms"] / 1e3
    # Rebase to 0 so offset_s works regardless of absolute timestamp
    df["t"] -= df["t"].iloc[0]
    fs = 1.0 / np.median(np.diff(df["t"].values))
    seg = df[(df["t"] >= t0) & (df["t"] < t1)][col].values.astype(np.float64)
    seg -= seg.mean()
    return seg, fs

def main():
    args = parse_args()
    # Autodetect channel name (old format: flexor/extensor; new: emg0..emg3)
    if args.channel is None:
        df_head = pd.read_csv(args.csv, nrows=1)
        if "emg0" in df_head.columns:
            channel = "emg0"
        elif "flexor" in df_head.columns:
            channel = "flexor"
        else:
            raise ValueError(f"No se encontró columna EMG en {args.csv}")
    else:
        channel = args.channel
    x, fs = load_seg(args.csv, channel, args.offset_s, args.offset_s + args.segment_s)
    print(f"canal={channel} | fs={fs:.1f} Hz | {len(x)} samples")

    # Bandpass 20-500 Hz (solo para mostrar distribución de energía, sin DC ni ruido extremo)
    b, a = signal.butter(2, [20, min(499, fs/2 - 1)], btype="bandpass", fs=fs)
    x_ac = signal.filtfilt(b, a, x)

    nperseg = int(fs * 2)
    f_raw, psd_raw = signal.welch(x,    fs=fs, nperseg=nperseg)
    f_ac,  psd_ac  = signal.welch(x_ac, fs=fs, nperseg=nperseg)

    psd_norm = psd_ac / psd_ac.max()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(f_ac, psd_norm, color="steelblue", lw=1.5, label="PSD señal (20–500 Hz AC)")

    ax.axvspan(20,  500, alpha=0.06, color="blue",  label="Rango típico EMG (20–500 Hz)")
    ax.axvspan(70,  300, alpha=0.12, color="green", label="Banda de paso filtro (70–300 Hz)")
    ax.axvline(50,  color="red",  lw=1, ls="--", alpha=0.7, label="50 Hz (red eléctrica)")
    ax.axvline(fs/2, color="gray", lw=1, ls=":", alpha=0.7, label=f"Nyquist ({fs/2:.0f} Hz)")

    ax.set_xlim(0, fs / 2)
    ax.set_ylim(1e-4, 2)
    ax.set_xlabel("Frecuencia (Hz)", fontsize=11)
    ax.set_ylabel("PSD normalizada", fontsize=11)
    ax.set_title(f"Densidad espectral de potencia — EMG canal {channel}\n"
                 f"(señal cruda sin filtro de paso de banda de presentación)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    plt.savefig("plots/emg_psd.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/emg_psd.png")
    plt.show()

if __name__ == "__main__":
    main()
