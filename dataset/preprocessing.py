"""
Pipeline de preprocesamiento offline para CSVs de captura.

Target:
    1. y_top, y_bottom -> invertidos y centrados (-y + offset), smoothed
       con gaussiana (sigma 150 ms por default).
    2. joy = y_top_smoothed + y_bottom_smoothed.
    3. velocity = central diff (np.gradient) de joy.
    4. (En el Dataset) normalizacion del target.

Features EMG (set configurable por canal, default ("rms_norm", "d_rms_norm")):
    Pipeline base por canal:
      raw -> bandpass causal 70-300 Hz, orden 4 (sosfilt, una pasada).

    Feature types disponibles:
      - "rms_norm":     RMS(40ms) / EMA(tau=1s)
      - "d_rms_norm":   backward diff de rms_norm
      - "wl_norm":      WL(40ms) / EMA(tau=1s)   (mean |diff| ventana)
      - "d_wl_norm":    backward diff de wl_norm

    Para cada feature type emite 2 canales (flx_*, ext_*).
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter, sosfilt


ALL_FEATURE_TYPES = (
    "rms_norm",         # RMS(40ms) / EMA(1s)
    "d_rms_norm",       # backward diff de rms_norm
    "rms_slow_norm",    # RMS(150ms) / EMA(1s) — smoothed amplitude
    "d_rms_slow_norm",  # backward diff de rms_slow_norm
    "rms",              # RMS(40ms) crudo (sin EMA) — para ablation
    "d_rms",            # backward diff de rms crudo
    "rms_slow",         # RMS(150ms) crudo (sin EMA) — para ablation
    "d_rms_slow",       # backward diff de rms_slow crudo
    "wl_norm",          # WL(40ms) / EMA(1s)
    "d_wl_norm",        # backward diff de wl_norm
    "ssc_norm",         # slope sign changes(40ms) / EMA(1s) — info espectral
)
DEFAULT_FEATURE_SET: Tuple[str, ...] = ("rms_norm", "d_rms_norm")


# ---------- utilities ----------

def estimate_fs(t_seconds: np.ndarray) -> float:
    """fs media a partir de timestamps en segundos. Asume t monotono creciente."""
    dt = np.diff(t_seconds)
    if np.any(dt <= 0):
        raise ValueError("t no es monotono creciente — revisar t_micros / overflow")
    return float(1.0 / np.mean(dt))


def central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Diferencia central (np.gradient). Solo offline (no causal)."""
    return np.gradient(x.astype(np.float64), dt)


def backward_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Diferencia causal (x[t] - x[t-1]) / dt. Primera muestra = 0."""
    out = np.empty_like(x, dtype=np.float64)
    out[0] = 0.0
    out[1:] = (x[1:] - x[:-1]) / dt
    return out


# ---------- target pipeline ----------

def gaussian_smooth_target(x: np.ndarray, fs: float, sigma_ms: float = 150.0) -> np.ndarray:
    """Filtro gaussiano (no causal) sobre la senal de posicion del joystick."""
    sigma_samples = sigma_ms / 1000.0 * fs
    return gaussian_filter1d(x.astype(np.float64), sigma=sigma_samples, mode="nearest")


def compute_target_velocity(
    y_top: np.ndarray,
    y_bottom: np.ndarray,
    fs: float,
    sigma_ms: float = 150.0,
    top_offset: float = 500.0,
    bottom_offset: float = 550.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (joy_position, velocity_unnormalized).

    joy = smooth(-y_top + top_offset) + smooth(-y_bottom + bottom_offset)
    velocity = d(joy)/dt usando diferencia central (no causal).

    El signo queda: extension positiva (sube y_top -> baja a 0 -> -0+500=500),
    flexion negativa (sube y_bottom hasta 1023 -> -1023+550=-473).
    """
    yt = gaussian_smooth_target(-y_top.astype(np.float64) + top_offset, fs, sigma_ms)
    yb = gaussian_smooth_target(-y_bottom.astype(np.float64) + bottom_offset, fs, sigma_ms)
    joy = yt + yb
    velocity = central_diff(joy, dt=1.0 / fs)
    return joy, velocity


# ---------- EMG feature pipeline ----------

def causal_bandpass(
    x: np.ndarray,
    fs: float,
    low: float = 70.0,
    high: float = 300.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass causal (sosfilt una pasada)."""
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"Banda invalida: low={low}, high={high}, nyq={nyq}")
    sos = butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfilt(sos, x.astype(np.float64))


def causal_rms(x: np.ndarray, fs: float, window_ms: float = 40.0) -> np.ndarray:
    """
    RMS causal con ventana rectangular movil de window_ms.
    En el warm-up (primeras win muestras) divide por el count efectivo.
    """
    win = max(1, int(round(window_ms / 1000.0 * fs)))
    sq = x.astype(np.float64) ** 2
    csum = np.cumsum(sq)
    avg = np.empty_like(sq)
    avg[:win] = csum[:win] / np.arange(1, win + 1)
    avg[win:] = (csum[win:] - csum[:-win]) / win
    return np.sqrt(avg)


def causal_ssc(
    x: np.ndarray,
    fs: float,
    window_ms: float = 40.0,
    threshold_rel: float = 0.1,
    threshold_ema_tau_s: float = 1.0,
) -> np.ndarray:
    """
    Slope Sign Changes (SSC) causal con threshold relativo.

    Cuenta extremos locales (picos + valles) en la ventana: en cada sample i,
    el slope de i-1 a i tiene signo opuesto al slope de i-2 a i-1, lo cual
    indica un maximo o minimo local en i-1.

    Threshold adaptativo (sobre la magnitud del slope) para evitar contar
    extremos generados por ruido:

        threshold[t] = threshold_rel * EMA(|x|, tau=threshold_ema_tau_s)[t]

    Devuelve extremos locales por segundo (Hz).
    """
    x = x.astype(np.float64)

    # Slopes de un sample al siguiente
    d1 = np.empty(len(x), dtype=np.float64)
    d1[0] = 0.0
    d1[1:] = x[1:] - x[:-1]

    # Cambio de signo del slope (extremo local en i-1).
    sign_d1 = np.sign(d1)
    sign_change = np.empty(len(x), dtype=bool)
    sign_change[0] = False
    sign_change[1:] = sign_d1[1:] != sign_d1[:-1]

    # Threshold adaptativo, exige que ambos slopes (actual y previo) sean
    # de magnitud > threshold para descartar extremos por ruido.
    baseline = causal_ema(np.abs(x), fs, threshold_ema_tau_s)
    thr = threshold_rel * baseline

    abs_d1 = np.abs(d1)
    big_now = abs_d1 > thr
    big_prev = np.empty(len(x), dtype=bool)
    big_prev[0] = False
    big_prev[1:] = big_now[:-1]

    valid_ssc = sign_change & big_now & big_prev

    win = max(1, int(round(window_ms / 1000.0 * fs)))
    csum = np.cumsum(valid_ssc.astype(np.float64))
    out = np.empty_like(csum)
    out[:win] = csum[:win] / np.arange(1, win + 1)
    out[win:] = (csum[win:] - csum[:-win]) / win

    return out * fs  # extremos por segundo (Hz)


def causal_wl(x: np.ndarray, fs: float, window_ms: float = 40.0) -> np.ndarray:
    """
    Waveform Length (variante "mean absolute first-difference") causal.

    WL[t] = (1/W) * sum_{i=t-W+1..t} |x[i] - x[i-1]|

    Captura mezcla de amplitud y contenido frecuencial: oscilaciones rapidas
    de baja amplitud dan WL similar a movimientos lentos de alta amplitud.
    Hudgins 1993 lo define como suma (no media); usamos media para mantener
    la misma escala que RMS y poder normalizar por EMA del mismo modo.
    """
    win = max(1, int(round(window_ms / 1000.0 * fs)))
    abs_diff = np.abs(np.diff(x.astype(np.float64), prepend=x[0]))
    csum = np.cumsum(abs_diff)
    out = np.empty_like(abs_diff)
    out[:win] = csum[:win] / np.arange(1, win + 1)  # warm-up: average over count efectivo
    out[win:] = (csum[win:] - csum[:-win]) / win
    return out


def causal_ema(x: np.ndarray, fs: float, tau_s: float = 1.0) -> np.ndarray:
    """EMA causal con constante de tiempo tau_s, via lfilter (vectorizado)."""
    dt = 1.0 / fs
    alpha = 1.0 - float(np.exp(-dt / tau_s))
    # y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
    return lfilter([alpha], [1.0, -(1.0 - alpha)], x.astype(np.float64))


def ema_normalize(
    rms: np.ndarray,
    fs: float,
    tau_s: float = 1.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normaliza RMS dividiendo por su EMA (~baseline adaptativa)."""
    baseline = causal_ema(rms, fs, tau_s=tau_s)
    return rms / (baseline + eps)


def _channel_features(
    x_raw: np.ndarray,
    fs: float,
    feature_set: Sequence[str],
    bp_low: float, bp_high: float, bp_order: int,
    rms_window_ms: float, rms_slow_window_ms: float,
    wl_window_ms: float,
    ssc_window_ms: float, ssc_threshold_rel: float,
    ema_tau_s: float,
) -> Dict[str, np.ndarray]:
    """Computa los features pedidos para un solo canal EMG."""
    dt = 1.0 / fs
    nyq = fs / 2.0
    high = min(bp_high, nyq * 0.95)

    x_bp = causal_bandpass(x_raw, fs, bp_low, high, bp_order)

    out: Dict[str, np.ndarray] = {}

    needs_rms_norm = "rms_norm" in feature_set or "d_rms_norm" in feature_set
    needs_rms_raw  = "rms"      in feature_set or "d_rms"      in feature_set
    if needs_rms_norm or needs_rms_raw:
        rms = causal_rms(x_bp, fs, rms_window_ms)
        if needs_rms_norm:
            rms_norm = ema_normalize(rms, fs, ema_tau_s)
            if "rms_norm" in feature_set:
                out["rms_norm"] = rms_norm
            if "d_rms_norm" in feature_set:
                out["d_rms_norm"] = backward_diff(rms_norm, dt)
        if needs_rms_raw:
            if "rms" in feature_set:
                out["rms"] = rms
            if "d_rms" in feature_set:
                out["d_rms"] = backward_diff(rms, dt)

    needs_rms_slow     = "rms_slow_norm" in feature_set or "d_rms_slow_norm" in feature_set
    needs_rms_slow_raw = "rms_slow"      in feature_set or "d_rms_slow"      in feature_set
    if needs_rms_slow or needs_rms_slow_raw:
        rms_slow = causal_rms(x_bp, fs, rms_slow_window_ms)
        if needs_rms_slow:
            rms_slow_norm = ema_normalize(rms_slow, fs, ema_tau_s)
            if "rms_slow_norm" in feature_set:
                out["rms_slow_norm"] = rms_slow_norm
            if "d_rms_slow_norm" in feature_set:
                out["d_rms_slow_norm"] = backward_diff(rms_slow_norm, dt)
        if needs_rms_slow_raw:
            if "rms_slow" in feature_set:
                out["rms_slow"] = rms_slow
            if "d_rms_slow" in feature_set:
                out["d_rms_slow"] = backward_diff(rms_slow, dt)

    needs_wl_norm = "wl_norm" in feature_set or "d_wl_norm" in feature_set
    if needs_wl_norm:
        wl = causal_wl(x_bp, fs, wl_window_ms)
        wl_norm = ema_normalize(wl, fs, ema_tau_s)
        if "wl_norm" in feature_set:
            out["wl_norm"] = wl_norm
        if "d_wl_norm" in feature_set:
            out["d_wl_norm"] = backward_diff(wl_norm, dt)

    if "ssc_norm" in feature_set:
        ssc = causal_ssc(
            x_bp, fs, window_ms=ssc_window_ms,
            threshold_rel=ssc_threshold_rel,
            threshold_ema_tau_s=ema_tau_s,
        )
        out["ssc_norm"] = ema_normalize(ssc, fs, ema_tau_s)

    return out


def compute_emg_features(
    flexor: np.ndarray,
    extensor: np.ndarray,
    fs: float,
    feature_set: Sequence[str] = DEFAULT_FEATURE_SET,
    bp_low: float = 70.0,
    bp_high: float = 300.0,
    bp_order: int = 4,
    rms_window_ms: float = 40.0,
    rms_slow_window_ms: float = 150.0,
    wl_window_ms: float = 40.0,
    ssc_window_ms: float = 40.0,
    ssc_threshold_rel: float = 0.1,
    ema_tau_s: float = 1.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Devuelve (features [C, T], feature_labels) con C = len(feature_set) * 2 canales.

    Para cada feature type del set se emiten dos filas en orden:
        flx_<type>, ext_<type>

    Default feature_set = ("rms_norm", "d_rms_norm") -> 4 canales (compat hacia atras).
    Agregar "wl_norm" -> 6 canales (orden: flx_rms_norm, ext_rms_norm,
                                    flx_d_rms_norm, ext_d_rms_norm,
                                    flx_wl_norm, ext_wl_norm).
    """
    for name in feature_set:
        if name not in ALL_FEATURE_TYPES:
            raise ValueError(
                f"Feature '{name}' no disponible. Opciones: {ALL_FEATURE_TYPES}"
            )

    kwargs = dict(
        fs=fs, feature_set=feature_set,
        bp_low=bp_low, bp_high=bp_high, bp_order=bp_order,
        rms_window_ms=rms_window_ms, rms_slow_window_ms=rms_slow_window_ms,
        wl_window_ms=wl_window_ms,
        ssc_window_ms=ssc_window_ms, ssc_threshold_rel=ssc_threshold_rel,
        ema_tau_s=ema_tau_s,
    )
    flx_feats = _channel_features(flexor, **kwargs)
    ext_feats = _channel_features(extensor, **kwargs)

    labels: List[str] = []
    stack: List[np.ndarray] = []
    for name in feature_set:
        labels.append(f"flx_{name}")
        stack.append(flx_feats[name])
        labels.append(f"ext_{name}")
        stack.append(ext_feats[name])

    return np.stack(stack, axis=0), labels
