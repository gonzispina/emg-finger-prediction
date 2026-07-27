"""
Onset detection analysis — plots para la presentacion.

Genera 4 figuras:
  1. onset_overview.png   — señal EMG + joystick con marcadores de onset
  2. onset_histogram.png  — distribucion de delays EMG→joystick (ext vs flx)
  3. envelope_compare.png — RMS_fast vs RMS_slow vs Butterworth LP
  4. rms_vs_joy.png       — RMS extensor/flexor vs eje joystick correspondiente

Uso:
    python plots/plot_onset_analysis.py
    python plots/plot_onset_analysis.py --csv S02.csv S03.csv
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# ── configuracion ──────────────────────────────────────────────────────────────
CSVS_DEFAULT = [
    "data/raw/P01/P01_S02_20260525_184004.csv",
    "data/raw/P01/P01_S03_20260525_194322.csv",
]
BP_LOW, BP_HIGH = 70, 300   # Hz bandpass
RMS_FAST_MS = 40            # ms ventana RMS rapido
RMS_SLOW_MS = 150           # ms ventana RMS lento
SIGMA_JOY_MS = 50           # ms gaussian del target


# ── helpers ───────────────────────────────────────────────────────────────────
def load_one(path):
    df = pd.read_csv(path)
    df["t"] = df["t_micros"] / 1e6
    df["t"] -= df["t"].iloc[0]
    if "emg0" in df.columns:
        df = df.rename(columns={"emg0": "extensor", "emg1": "flexor"})
        df["y_top"]    = df["joy_ext_ind"]
        df["y_bottom"] = df["joy_flex_ind"]
    return df

def load(paths):
    """Carga uno o más CSVs y los concatena con timestamps continuos."""
    if isinstance(paths, str):
        paths = [paths]
    parts = []
    t_offset = 0.0
    fs = None
    for p in paths:
        d = load_one(p)
        _fs = 1.0 / np.median(np.diff(d["t"].values))
        if fs is None:
            fs = _fs
        d["t"] += t_offset
        t_offset = d["t"].iloc[-1] + 1.0 / fs  # 1 sample gap entre sesiones
        parts.append(d)
    return pd.concat(parts, ignore_index=True), fs

def bandpass(x, fs, lo=BP_LOW, hi=BP_HIGH, order=4):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfilt(sos, x)

def bandpass_zero(x, fs, lo=BP_LOW, hi=BP_HIGH, order=4):
    sos = signal.butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x)

def rolling_rms(x, fs, window_ms):
    w = max(1, int(fs * window_ms / 1000))
    x2 = x ** 2
    cs = np.cumsum(x2)
    avg = np.empty_like(cs)
    avg[:w] = cs[:w] / np.arange(1, w + 1)
    avg[w:] = (cs[w:] - cs[:-w]) / w
    return np.sqrt(avg)

def butter_envelope(x, fs, lp_hz=1.0):
    rect = np.abs(x)
    sos = signal.butter(2, lp_hz, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, rect)

def gaussian_smooth(x, fs, sigma_ms):
    return gaussian_filter1d(x.astype(float), sigma=sigma_ms / 1000 * fs, mode="nearest")

def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-9)

def first_crossing(t, mask, sustain_ms, fs):
    n = max(1, int(fs * sustain_ms / 1000))
    mask = np.asarray(mask)
    for i in range(len(mask) - n + 1):
        if mask[i:i+n].all():
            return t[i]
    return np.nan

def sustained_crossings(t, mask, sustain_ms, fs):
    n = max(1, int(fs * sustain_ms / 1000))
    mask = np.asarray(mask)
    out, i = [], 0
    while i <= len(mask) - n:
        if mask[i:i+n].all():
            out.append(t[i])
            while i < len(mask) and mask[i]:
                i += 1
        else:
            i += 1
    return out

def choose_emg_onset(emg_times, t_joy):
    if len(emg_times) == 0 or np.isnan(t_joy):
        return np.nan
    before = [t for t in emg_times if t <= t_joy]
    if before:
        return max(before)
    return min(emg_times, key=lambda t: abs(t - t_joy))


# ── onset detection ────────────────────────────────────────────────────────────
def detect_onsets(df, fs, emg_env_col, joy_col, tasks_move, k_emg=2.0, k_joy=5,
                  min_joy_delta=20, sustain_ms=35, refractory_s=0.2, window_s=0.5):
    """
    Detecta pares de onsets (EMG, joystick) dentro de bloques de movimiento.
    Usa rest_baseline global como referencia para los umbrales.
    Cada cruce sostenido del joystick = un evento; busca el onset EMG previo.
    """
    baseline = df[df["task"] == "rest_baseline"]
    if baseline.empty:
        raise ValueError("No hay rest_baseline en el CSV")

    emg_thr  = baseline[emg_env_col].mean() + k_emg * baseline[emg_env_col].std()
    joy_base = baseline[joy_col].median()
    joy_thr  = max(min_joy_delta, k_joy * baseline[joy_col].std())

    results = []
    for task in tasks_move:
        block = df[df["task"] == task]
        if block.empty:
            continue

        t    = block["t"].values
        emg  = block[emg_env_col].values
        joy  = block[joy_col].values

        joy_mask = np.abs(joy - joy_base) > joy_thr
        joy_onsets = sustained_crossings(t, joy_mask, sustain_ms, fs)

        last_t = -np.inf
        for t_joy in joy_onsets:
            if t_joy - last_t < refractory_s:
                continue
            # Umbral LOCAL: median de 300ms justo antes de la ventana de búsqueda
            local_mask = (t >= t_joy - window_s - 0.3) & (t < t_joy - window_s)
            local_baseline = emg[local_mask]
            if len(local_baseline) < 10:
                local_thr = emg_thr  # fallback al global
            else:
                local_thr = np.median(local_baseline) + k_emg * local_baseline.std()

            win_mask = (t >= t_joy - window_s) & (t <= t_joy + 0.1)
            t_win    = t[win_mask]
            emg_win  = emg[win_mask] > local_thr
            emg_times = sustained_crossings(t_win, emg_win, sustain_ms, fs)
            t_emg = choose_emg_onset(emg_times, t_joy)

            delta = t_joy - t_emg if not np.isnan(t_emg) else np.nan
            results.append({
                "task": task, "t_emg": t_emg, "t_joy": t_joy, "delta": delta,
                "emg_thr": emg_thr, "joy_base": joy_base,
            })
            last_t = t_joy

    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["task", "t_emg", "t_joy", "delta", "emg_thr", "joy_base"])


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", default=CSVS_DEFAULT)
    args = ap.parse_args()

    df, fs = load(args.csv)
    print(f"Sesiones cargadas: {len(args.csv)} | N total={len(df)}")
    print(f"fs={fs:.1f} Hz | N={len(df)}")

    # Centrado DC
    df["extensor"] -= df["extensor"].mean()
    df["flexor"]   -= df["flexor"].mean()

    # Target joystick
    joy_top    = gaussian_smooth(-df["y_top"].values    + df["y_top"].mean(),    fs, SIGMA_JOY_MS)
    joy_bottom = gaussian_smooth(-df["y_bottom"].values + df["y_bottom"].mean(), fs, SIGMA_JOY_MS)
    df["joy"]      = joy_top + joy_bottom
    df["joy_top_s"]    = joy_top
    df["joy_bottom_s"] = joy_bottom

    # Features EMG
    ext_bp = bandpass_zero(df["extensor"].values, fs)
    flx_bp = bandpass_zero(df["flexor"].values,   fs)
    df["ext_rms_fast"] = rolling_rms(ext_bp, fs, RMS_FAST_MS)
    df["ext_rms_slow"] = rolling_rms(ext_bp, fs, RMS_SLOW_MS)
    df["flx_rms_fast"] = rolling_rms(flx_bp, fs, RMS_FAST_MS)
    df["flx_rms_slow"] = rolling_rms(flx_bp, fs, RMS_SLOW_MS)
    df["ext_env_lp"]    = butter_envelope(ext_bp, fs, lp_hz=1.0)  # 1Hz: visualización
    df["flx_env_lp"]    = butter_envelope(flx_bp, fs, lp_hz=1.0)
    df["ext_env_onset"] = butter_envelope(ext_bp, fs, lp_hz=2.0)  # 2Hz: más suave, detección
    df["flx_env_onset"] = butter_envelope(flx_bp, fs, lp_hz=2.0)
    df["ext_bp"]       = ext_bp
    df["flx_bp"]       = flx_bp

    # Onset detection: fast_ext (extensor) + fast_flex (flexor)
    # Usamos la envolvente Butterworth bidireccional (filtfilt): más suave que
    # el RMS, sin delay artificial → umbrales más limpios. Solo válido offline.
    onsets_ext = detect_onsets(df, fs, "ext_env_onset", "y_top",
                               tasks_move=["fast_ext"], k_emg=2.0, window_s=0.6)
    onsets_flx = detect_onsets(df, fs, "flx_env_onset", "y_bottom",
                               tasks_move=["fast_flex"], k_emg=2.0, window_s=0.5)
    onsets = pd.concat([onsets_ext, onsets_flx], ignore_index=True)
    onsets_clean = onsets.dropna(subset=["delta"])
    onsets_clean = onsets_clean[onsets_clean["delta"].between(
        onsets_clean["delta"].quantile(0.05), onsets_clean["delta"].quantile(0.95))]

    print(f"Onsets detectados: {len(onsets_clean)} válidos / {len(onsets)} total")
    if len(onsets_clean):
        print(f"  delay mean={onsets_clean['delta'].mean()*1000:.0f}ms  "
              f"median={onsets_clean['delta'].median()*1000:.0f}ms  "
              f"max={onsets_clean['delta'].max()*1000:.0f}ms")

    # ── FIGURA 1: onset overview — fast_ext ───────────────────────────────────
    t_start = df[df["task"] == "fast_ext"]["t"].iloc[0] - 1
    t_end   = t_start + 20
    seg     = df[(df["t"] >= t_start) & (df["t"] <= t_end)]

    env = seg["ext_env_onset"].values.astype(float)
    joy = seg["joy_top_s"].values.astype(float)
    env_n = (env - env.min()) / (env.max() - env.min() + 1e-9)
    joy_n = (joy - joy.min()) / (joy.max() - joy.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(seg["t"], env_n, color="steelblue",  lw=1.8, alpha=0.9,
            label="Envolvente EMG extensor (Butterworth LP 1 Hz, bidireccional)")
    ax.plot(seg["t"], joy_n, color="darkorange", lw=1.8, alpha=0.9,
            label="Posición joystick extensión (normalizada)")

    first_onset = True
    for _, row in onsets_ext.iterrows():
        if not (np.isnan(row["t_emg"]) or np.isnan(row["t_joy"])):
            if t_start <= row["t_emg"] <= t_end:
                lbl_emg = "Onset EMG"      if first_onset else ""
                lbl_joy = "Onset joystick" if first_onset else ""
                ax.axvline(row["t_emg"], color="steelblue",  lw=1.5, ls="--", alpha=0.85, label=lbl_emg)
                ax.axvline(row["t_joy"], color="darkorange", lw=1.5, ls=":",  alpha=0.85, label=lbl_joy)
                if first_onset:
                    ax.annotate("", xy=(row["t_joy"], 0.88), xytext=(row["t_emg"], 0.88),
                                arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))
                    ax.text((row["t_emg"] + row["t_joy"]) / 2, 0.92,
                            f"Δt≈{row['delta']*1000:.0f}ms", ha="center", fontsize=9, color="gray")
                first_onset = False

    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Amplitud normalizada [0–1]", fontsize=11)
    ax.set_title("Onset detection: la señal EMG precede al movimiento del joystick  (fast_ext)\n"
                 "Envolvente Butterworth bidireccional — válida solo offline", fontsize=11)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/onset_overview.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/onset_overview.png")
    plt.close()

    # ── FIGURA 2: histograma de delays — extensión vs flexión ─────────────────
    ext_clean = onsets_ext.dropna(subset=["delta"])
    ext_clean = ext_clean[ext_clean["delta"].between(
        ext_clean["delta"].quantile(0.05), ext_clean["delta"].quantile(0.95))]
    flx_clean = onsets_flx.dropna(subset=["delta"])
    flx_clean = flx_clean[flx_clean["delta"].between(
        flx_clean["delta"].quantile(0.05), flx_clean["delta"].quantile(0.95))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    fig.suptitle("Distribución de delays EMG → joystick\n"
                 "Valores positivos = EMG anticipa al joystick ✓", fontsize=12)

    for ax, data, label, color in [
        (ax1, ext_clean["delta"].values * 1000, "Extensión  (fast_ext)", "steelblue"),
        (ax2, flx_clean["delta"].values * 1000, "Flexión  (fast_flex)",  "tomato"),
    ]:
        bins = np.linspace(0, data.max() * 1.05, 16)
        ax.hist(data, bins=bins, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(data),   color="black",  lw=1.8, ls="--",
                   label=f"Media = {np.mean(data):.0f} ms")
        ax.axvline(np.median(data), color="orange", lw=1.8, ls=":",
                   label=f"Mediana = {np.median(data):.0f} ms")
        p95 = np.percentile(data, 95)
        ax.axvline(p95, color="green", lw=1.5, ls="-.",
                   label=f"P95 = {p95:.0f} ms")
        ax.set_xlim(0, data.max() * 1.08)
        ax.set_xlabel("t_joystick − t_EMG  (ms)", fontsize=11)
        ax.set_ylabel("Cantidad de onsets", fontsize=11)
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/onset_histogram.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/onset_histogram.png")
    plt.close()

    # ── FIGURA 3: comparacion de envolventes ───────────────────────────────────
    # Usar fast_ext para comparar envolventes — movimientos limpios y repetidos
    seg2 = df[df["task"] == "fast_ext"].iloc[:int(fs * 30)]
    # Mostrar |señal| para que todo sea positivo
    def norm01(x):
        x = np.asarray(x, dtype=float)
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(seg2["t"], norm01(np.abs(seg2["ext_bp"])),    color="lightgray",  lw=0.6,
            alpha=0.8, label="|Señal filtrada| (70-300 Hz)")
    ax.plot(seg2["t"], norm01(seg2["ext_rms_fast"]),      color="steelblue",  lw=1.8,
            label=f"RMS rápido ({RMS_FAST_MS} ms)  — causal")
    ax.plot(seg2["t"], norm01(seg2["ext_rms_slow"]),      color="darkorange", lw=1.8,
            label=f"RMS lento ({RMS_SLOW_MS} ms)  — causal")
    ax.plot(seg2["t"], norm01(seg2["ext_env_lp"]),        color="green",      lw=1.8,
            ls="--", label="Butterworth LP 1 Hz  — bidireccional (offline)")
    ax.set_xlabel("Tiempo (s)", fontsize=11)
    ax.set_ylabel("Amplitud normalizada [0–1]", fontsize=11)
    ax.set_title("Comparación de envolventes — canal extensor  (fast_ext)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/envelope_compare.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/envelope_compare.png")
    plt.close()

    # ── FIGURA 4: RMS vs joystick — free_controlled ────────────────────────────
    seg_free = df[df["task"] == "free_controlled"].copy()

    # Normalizar EMG y joystick juntos al mismo rango global [0,1]
    # para que sean comparables en el mismo eje
    ext_rms = seg_free["ext_rms_fast"].values.astype(float)
    flx_rms = seg_free["flx_rms_fast"].values.astype(float)
    joy_top = seg_free["joy_top_s"].values.astype(float)
    joy_bot = seg_free["joy_bottom_s"].values.astype(float)

    # Normalizar cada par (RMS, joystick) al rango combinado para que la
    # comparación de amplitudes sea honesta
    def norm_pair(a, b):
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        return (a - lo) / (hi - lo + 1e-9), (b - lo) / (hi - lo + 1e-9)

    # Normalizar cada señal independientemente para ver su forma
    ext_n     = norm01(ext_rms)
    joy_top_n = norm01(joy_top)
    flx_n     = norm01(flx_rms)
    joy_bot_n = norm01(joy_bot)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle("Correlación EMG → joystick  (free_controlled)", fontsize=12)

    ax1.plot(seg_free["t"], ext_n,     color="steelblue",  lw=1.2, alpha=0.9,
             label="RMS extensor (norm)")
    ax1.plot(seg_free["t"], joy_top_n, color="darkorange",  lw=1.5, alpha=0.9,
             label="Joystick extensión (norm)")
    ax1.set_ylabel("Amplitud [0–1]")
    ax1.set_title("Extensor  →  joystick extensión")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(seg_free["t"], flx_n,     color="tomato",  lw=1.2, alpha=0.9,
             label="RMS flexor (norm)")
    ax2.plot(seg_free["t"], joy_bot_n, color="purple",  lw=1.5, alpha=0.9,
             label="Joystick flexión (norm)")
    ax2.set_ylabel("Amplitud [0–1]")
    ax2.set_xlabel("Tiempo (s)")
    ax2.set_title("Flexor  →  joystick flexión")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plots/rms_vs_joy.png", dpi=150, bbox_inches="tight")
    print("Guardado: plots/rms_vs_joy.png")
    plt.close()


if __name__ == "__main__":
    main()
