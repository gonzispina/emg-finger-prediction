"""
Replay grafico de una sesion con prediccion del modelo.

Soporta:
  - Modelos de 1 dedo (EMGSessionDataset, schema legacy): 1 cajita.
  - Modelos de 2 dedos (EMGSession2FingersDataset, schema binary_v2):
    2 cajitas side-by-side + 2 paneles de velocidad.

El modo se detecta del checkpoint:
  output_dim==1 -> 1 dedo; output_dim==2 -> 2 dedos.

Flag --hide-target oculta la senal real (circulo verde + linea verde) para
demostraciones donde solo querer mostrar la prediccion.

Uso:
    # 1 dedo (legacy)
    python visualize.py --checkpoint checkpoints/tcn_s01_50ep/best.pt \\
                        --csv data/raw/P01/P01_S01_20260524_181522.csv

    # 2 dedos
    python visualize.py --checkpoint checkpoints/tcn_2f_s04s05_baseline/best.pt \\
                        --csv data/raw/P01/P01_S05_20260618_172229.csv

    # MODO LIVE (ESP32 en vivo):
    python visualize.py --checkpoint checkpoints/tcn_s02s03_RF567_sobolev01_200ep/best.pt \\
                        --port /dev/cu.usbmodem21101

    # MODO LIVE (mock sin hardware):
    python visualize.py --checkpoint checkpoints/tcn_s02s03_RF567_sobolev01_200ep/best.pt \\
                        --mock

    # Sin target visible
    python visualize.py ... --hide-target
"""
from __future__ import annotations

import argparse
import collections
import struct
import time
from pathlib import Path
from typing import Optional

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import CheckButtons
from scipy.signal import butter, lfilter, sosfilt_zi

from dataset import DEFAULT_FEATURE_SET, EMGSessionDataset, PreprocessConfig
from dataset.emg_dataset_2fingers import EMGSession2FingersDataset, PreprocessConfig2F
from models import EMGTCN


# ── Frame protocol (ESP32) ────────────────────────────────────────────────────
_FRAME_SIZE  = 31
_MAGIC       = b"\xAA\x55"
_PAYLOAD_FMT = "<I4iHHHH"

def _crc8_table_init() -> bytes:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc << 1) ^ 0x07 if crc & 0x80 else crc << 1
            crc &= 0xFF
        table.append(crc)
    return bytes(table)

_CRC8_TABLE = _crc8_table_init()

def _crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _CRC8_TABLE[crc ^ b]
    return crc


# ── Streams ───────────────────────────────────────────────────────────────────

class _SerialStream:
    def __init__(self, port: str, baud: int = 2000000):
        import serial
        self.ser = serial.Serial(port, baud, timeout=0.01)
        self.buf = bytearray()

    def read(self) -> Optional[np.ndarray]:
        nwait = self.ser.in_waiting
        if nwait:
            self.buf.extend(self.ser.read(nwait))
        while True:
            idx = self.buf.find(_MAGIC)
            if idx < 0:
                if len(self.buf) > 1024:
                    del self.buf[:-1]
                return None
            if idx > 0:
                del self.buf[:idx]
            if len(self.buf) < _FRAME_SIZE:
                return None
            frame = bytes(self.buf[:_FRAME_SIZE])
            if frame[30] == _crc8(frame[2:30]):
                del self.buf[:_FRAME_SIZE]
                _, e0, e1, e2, e3, j_ei, j_fi, j_em, j_fm = struct.unpack(
                    _PAYLOAD_FMT, frame[2:30])
                return np.array([e0, e1, e2, e3, j_ei, j_fi, j_em, j_fm],
                                dtype=np.float32)
            del self.buf[0]

    def close(self):
        try: self.ser.close()
        except Exception: pass

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", default=None,
                   help="CSV para replay (omitir si se usa --port o --mock)")
    p.add_argument("--port", default=None,
                   help="Puerto serial para inferencia live")
    p.add_argument("--baud", type=int, default=2000000,
                   help="Baud rate serial (default 2000000 para ESP32; usar 1000000 con Arduino)")
    p.add_argument("--mock", action="store_true",
                   help="Stream mock (sin hardware, para probar)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--speed", type=float, default=2.0,
                   help="playback speed (1.0 = realtime)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="display update rate")
    p.add_argument("--start", type=float, default=0.0,
                   help="start time in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="duration in seconds; default = whole session")
    p.add_argument("--leak-tau", type=float, default=0.0,
                   help="leaky integrator time constant (s); smaller = mas anclaje a 0. "
                        "0 (default) = sin leak, integracion cinematica pura + clipping fisico")
    p.add_argument("--vel-smooth-tau", type=float, default=0.01,
                   help="EMA tau (s) para suavizar la velocidad predicha antes de integrar; "
                        "0 = sin smoothing")
    p.add_argument("--dead-zone", type=float, default=0.0,
                   help="threshold (en unidades de velocidad raw) abajo del cual v_pred = 0. "
                        "0 (default) = sin dead zone.")
    p.add_argument("--save", default=None,
                   help="path para guardar como mp4/gif (requiere ffmpeg/imagemagick)")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    output_dim = ckpt.get("output_dim", 1)
    kernel_size = cfg.get("kernel_size", 3)
    dilations = cfg.get("dilations", [1, 3, 9, 27, 81])
    model = EMGTCN(
        in_channels=ckpt["in_channels"],
        hidden_channels=cfg["hidden_channels"],
        output_dim=output_dim,
        kernel_size=kernel_size,
        dropout=cfg["dropout"],
        dilations=dilations,
        use_weight_norm=not cfg.get("no_weight_norm", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_full_sequence(model: EMGTCN, features: np.ndarray,
                          device: torch.device) -> np.ndarray:
    """Devuelve predicciones [output_dim, T] sobre toda la secuencia."""
    x = torch.from_numpy(features).unsqueeze(0).to(device)   # [1, C, T]
    h = x
    for block in model.blocks:
        h = block(h)
    if model.has_final_conv:
        h = model.final_drop(model.final_act(model.final_conv(h)))
    h = model.head(h)                                        # [1, output_dim, T]
    return h[0].cpu().numpy()                                # [output_dim, T]


def smooth_velocity(v: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """EMA causal sobre la velocidad predicha. tau=0 -> sin smoothing."""
    if tau_s <= 0:
        return v.astype(np.float64)
    dt = 1.0 / fs
    alpha = 1.0 - float(np.exp(-dt / tau_s))
    return lfilter([alpha], [1.0, -(1.0 - alpha)], v.astype(np.float64))


def apply_dead_zone(v: np.ndarray, threshold: float) -> np.ndarray:
    """Manda a 0 las velocidades con |v| < threshold."""
    if threshold <= 0:
        return v
    return np.where(np.abs(v) < threshold, 0.0, v)


def leaky_integrate(velocity: np.ndarray, dt: float, leak_tau: float,
                    initial: float = 0.0,
                    pos_min: float | None = None,
                    pos_max: float | None = None) -> np.ndarray:
    """pos[t] = clip( pos[t-1] * (1 - leak) + velocity[t] * dt,  pos_min, pos_max )

    leak = dt / leak_tau (tiende a 0 con time constant leak_tau).

    Si leak_tau <= 0 o no finito, se usa leak=0 (integracion pura con
    clipping, sin damping hacia el reposo).

    Si pos_min/pos_max no son None, clippea cada paso para que el
    integrador no se salga del rango fisico (= topes mecanicos del dedo).
    """
    if leak_tau is None or leak_tau <= 0 or not np.isfinite(leak_tau):
        leak = 0.0
    else:
        leak = dt / leak_tau
    pos = np.empty_like(velocity, dtype=np.float64)
    pos[0] = initial
    if pos_min is None and pos_max is None:
        for i in range(1, len(velocity)):
            pos[i] = pos[i - 1] * (1.0 - leak) + velocity[i] * dt
    else:
        lo = -np.inf if pos_min is None else pos_min
        hi = +np.inf if pos_max is None else pos_max
        for i in range(1, len(velocity)):
            nxt = pos[i - 1] * (1.0 - leak) + velocity[i] * dt
            if nxt < lo:
                nxt = lo
            elif nxt > hi:
                nxt = hi
            pos[i] = nxt
    return pos


def detect_mode(ckpt) -> int:
    """Devuelve 1 (legacy) o 2 (dos dedos) segun el checkpoint."""
    cfg = ckpt.get("config", {})
    return int(ckpt.get("output_dim", cfg.get("output_dim", 1)))


def process_finger_velocity(pred_raw, fs, dt, args, tasks, true_position,
                            target_scale):
    """Aplica warmup-mask + smoothing + dead-zone + leaky integrate.
    Devuelve (pred_velocity, pred_position, dead_zone_thr)."""
    pred_velocity_raw = pred_raw * target_scale       # unidades joy/s
    warmup_samples = int(round(fs * 1.0))
    pred_velocity_raw[:warmup_samples] = 0.0

    pred_velocity_smooth = smooth_velocity(pred_velocity_raw, fs, args.vel_smooth_tau)

    if args.dead_zone is None:
        rest_mask = tasks == "rest_baseline"
        if rest_mask.any():
            rest_vel = pred_velocity_smooth[rest_mask]
            dead_zone_thr = float(np.percentile(np.abs(rest_vel), 95))
        else:
            dead_zone_thr = float(np.percentile(np.abs(pred_velocity_smooth), 25))
    else:
        dead_zone_thr = float(args.dead_zone)

    pred_velocity = apply_dead_zone(pred_velocity_smooth, dead_zone_thr)
    pred_position = leaky_integrate(
        pred_velocity, dt, args.leak_tau,
        initial=float(true_position[0]),
        pos_min=float(true_position.min()),
        pos_max=float(true_position.max()),
    )
    return pred_velocity, pred_position, dead_zone_thr


def process_finger_position(pred_raw, fs, dt, position_scale, joy_center, smooth_tau=0.0):
    """Para modelos predict_pos: la salida ya es posicion normalizada.
    Devuelve (pred_velocity, pred_position) — sin integracion."""
    from dataset.preprocessing import central_diff as _central_diff, causal_ema
    warmup_samples = int(round(fs * 1.0))
    pred_position = pred_raw * position_scale - joy_center
    pred_position[:warmup_samples] = 0.0
    if smooth_tau > 0:
        pred_position = causal_ema(pred_position, fs, tau_s=smooth_tau).astype(np.float32)
    pred_velocity = _central_diff(pred_position, dt=dt)
    return pred_velocity, pred_position


def build_box_panel(ax, title, p_low, p_high, margin, start_position_true,
                    start_position_pred):
    """Arma una cajita vertical con 2 circulos (real verde / pred rojo).
    Devuelve dict con artistas para update."""
    ax.set_xlim(-1, 1)
    y_lim = (p_low - margin, p_high + margin)
    ax.set_ylim(*y_lim)
    ax.set_xticks([])
    ax.set_ylabel("position (joy)")
    ax.set_title(title)

    box = patches.Rectangle(
        (-0.75, p_low - margin * 0.3),
        1.5, (p_high - p_low) + margin * 0.6,
        linewidth=2, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(box)
    ax.axhline(0, color="gray", lw=0.7, linestyle="--", zorder=1)
    ax.axhline(p_high, color="lightgray", lw=0.5, linestyle=":", zorder=1)
    ax.axhline(p_low, color="lightgray", lw=0.5, linestyle=":", zorder=1)

    radius_units = max(15, (p_high - p_low) * 0.04)
    true_circle = patches.Circle(
        (-0.35, start_position_true),
        radius=radius_units, color="seagreen", alpha=0.9, zorder=3,
    )
    pred_circle = patches.Circle(
        (+0.35, start_position_pred),
        radius=radius_units, color="crimson", alpha=0.9, zorder=3,
    )
    ax.add_patch(true_circle); ax.add_patch(pred_circle)
    ax.text(-0.35, y_lim[0] + margin * 0.4, "real",
            ha="center", fontsize=10, fontweight="bold", color="seagreen")
    ax.text(+0.35, y_lim[0] + margin * 0.4, "predicho",
            ha="center", fontsize=10, fontweight="bold", color="crimson")
    return {"true_circle": true_circle, "pred_circle": pred_circle,
            "y_lim": y_lim, "box_patch": box}


def build_velocity_panel(ax, title, fs, v_lim, ylabel="vel (joy/s)"):
    """Señal scrolling con 2 lineas. Devuelve dict con buffers + lineas."""
    vel_buf_len = int(5.0 * fs)
    real_buf = np.zeros(vel_buf_len)
    pred_buf = np.zeros(vel_buf_len)
    time_axis = np.linspace(-5.0, 0.0, vel_buf_len)

    ax.set_xlim(-5, 0); ax.set_ylim(*v_lim)
    ax.set_xlabel("t (s, relativo a ahora)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, linestyle="--", zorder=1)
    ax.grid(True, alpha=0.3, zorder=0)
    real_line, = ax.plot(time_axis, real_buf, color="seagreen", lw=1.2,
                         label="real", zorder=2)
    pred_line, = ax.plot(time_axis, pred_buf, color="crimson", lw=1.2,
                         label="predicho", alpha=0.85, zorder=3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    return {"real_buf": real_buf, "pred_buf": pred_buf,
            "real_line": real_line, "pred_line": pred_line,
            "vel_buf_len": vel_buf_len}


def run_live(args):
    """
    Modo live: inferencia online desde serial (o mock).

    Estrategia: rolling buffer de 10s de raw samples. Cada UPDATE_EVERY muestras,
    corre compute_emg_features() y compute_target_velocity() sobre el buffer completo
    (las mismas funciones que usa el replay) y toma el último valor.
    Con 10s de historia la EMA está bien inicializada.
    """
    from dataset.preprocessing import compute_emg_features, compute_target_velocity

    device = pick_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    cfg      = ckpt["config"]
    mode     = detect_mode(ckpt)
    fs       = float(ckpt.get("fs") or cfg.get("fs") or 606.0)
    dt       = 1.0 / fs
    features = cfg.get("features") or list(DEFAULT_FEATURE_SET)
    sigma_ms = float(cfg.get("sigma_ms", 50.0))
    window_len = cfg["window_length"]
    # resolved_feature_clip es el valor real usado en entrenamiento (None = sin clip).
    # cfg["feature_clip"] es el default del argparse (20.0) y NO es confiable cuando
    # se uso --no-feature-clip; usar siempre resolved_feature_clip primero.
    _raw_clip = cfg.get("resolved_feature_clip", ckpt.get("feature_clip"))
    feat_clip = float(_raw_clip) if _raw_clip is not None else None

    predict_accel = bool(ckpt.get("predict_accel", False))
    _tgt_label = "accel" if predict_accel else "vel"
    print(f"Modo live — {mode} dedo(s)  |  features={features}  |  fs={fs:.0f} Hz  |  sigma={sigma_ms}ms  "
          f"|  feat_clip={feat_clip}  |  target={_tgt_label}")

    # Buffer crudo: 10s para que la EMA (tau=1s) esté inicializada al leer el extremo
    HISTORY_N = max(window_len, int(10.0 * fs))

    # Joy init con 512 (centro de hardware) → pos = (-512+512)×2 = 0 = centro de cajita
    # EMG init con 0 (sin actividad)
    raw = {}
    for ch in ("emg0", "emg1", "emg2", "emg3"):
        raw[ch] = collections.deque([0.0] * HISTORY_N, maxlen=HISTORY_N)
    for ch in ("joy_ext_ind", "joy_flex_ind", "joy_ext_med", "joy_flex_med"):
        raw[ch] = collections.deque([512.0] * HISTORY_N, maxlen=HISTORY_N)

    if mode == 1:
        target_scales = [float(ckpt["target_scale"])]
    else:
        target_scales = [float(ckpt["target_scale_ind"]), float(ckpt["target_scale_med"])]

    # Estado del leaky integrator de posición predicha
    pos_state = [0.0] * mode
    # Para predict_accel: buffer de velocidad integrada desde aceleración
    accel_vel_state = [0.0] * mode

    # Stream
    if args.port:
        stream = _SerialStream(args.port, baud=args.baud)
        print(f"Serial: {args.port} @ {args.baud} bps")
    else:
        exit(0)

    # ── Paneles: Arduino 10-bit, joy 0-1023, centro ~512
    # Posicion = -(joy_ext - 512) + -(joy_flex - 512), rango real ≈ ±512
    p_low, p_high, margin = -300.0, 300.0, 40.0
    v_amp = max(target_scales)   # target_scale = max vel de entrenamiento

    # Dead zone auto-calibrada desde los primeros 3s de predicciones (asume reposo inicial)
    dz_state   = {"value": 0.0, "computed": False}
    pred_vel_warmup: list[list[float]] = [[] for _ in range(mode)]
    DZ_WARMUP_N = int(3.0 * fs)

    n = mode
    if n == 1:
        fig, (ax_box, ax_vel) = plt.subplots(
            2, 1, figsize=(6, 10), gridspec_kw={"height_ratios": [3, 1]})
        finger_axes = [{"box": ax_box, "vel": ax_vel}]
    else:
        fig, axs = plt.subplots(
            2, n, figsize=(6*n, 10), gridspec_kw={"height_ratios": [3, 1]})
        finger_axes = [{"box": axs[0, i], "vel": axs[1, i]} for i in range(n)]

    labels = ["INDICE"] if n == 1 else ["INDICE", "MEDIO"]
    box_artists = []
    vel_artists = []
    for i, fa in enumerate(finger_axes):
        ba = build_box_panel(fa["box"], labels[i] + "  (live)",
                             p_low, p_high, margin, 0.0, 0.0)
        va = build_velocity_panel(fa["vel"], None, fs, (-v_amp, v_amp))
        box_artists.append(ba); vel_artists.append(va)

    ax_check = fig.add_axes([0.02, 0.92, 0.13, 0.05])
    check = CheckButtons(ax_check, ["show target"], [True])

    def toggle(label):
        v = check.get_status()[0]
        for ba in box_artists: ba["true_circle"].set_visible(v)
        for va in vel_artists:  va["real_line"].set_visible(v)
        fig.canvas.draw_idle()
    check.on_clicked(toggle)

    status_text = finger_axes[0]["box"].text(
        0, p_high + margin * 0.6, "", ha="center", fontsize=10, family="monospace")

    sample_count   = [0]
    real_samples   = [0]  # muestras reales recibidas (para offset dinámico)

    # EMA persistente para suavizar pred_vel entre frames (evita reset cada llamada)
    pred_ema_state = [0.0] * mode
    pred_ema_alpha = 1.0 - np.exp(-1.0 / (fs * args.vel_smooth_tau)) if args.vel_smooth_tau > 0 else 1.0

    def _arr(key): return np.array(raw[key], dtype=np.float64)

    def _joy_offset(key_ext, key_flx):
        """Median del buffer completo (incluye init + datos reales).
        Con init=512 el offset arranca correcto para Arduino desde el primer sample.
        Para ESP32 converge a ~1900 en cuanto llegan datos reales."""
        arr_ext = np.array(raw[key_ext])
        arr_flx = np.array(raw[key_flx])
        return float(np.median(arr_ext)), float(np.median(arr_flx))

    @torch.no_grad()
    def _compute_frame(new_samples: int):
        """Devuelve (true_pos, true_vel_arr, pred_vel_arr) donde *_arr tiene longitud new_samples."""
        # ── Features (mismas funciones que el replay) ─────────────────────────
        if mode == 1:
            feats, _ = compute_emg_features(
                _arr("emg1"), _arr("emg0"), fs, feature_set=features)
        else:
            fi, _ = compute_emg_features(_arr("emg1"), _arr("emg0"), fs, feature_set=features)
            fm, _ = compute_emg_features(_arr("emg3"), _arr("emg2"), fs, feature_set=features)
            feats = np.concatenate([fi, fm], axis=0)  # [C, T]
        # Clip identico al de entrenamiento (None = sin clip).
        if feat_clip is not None and feat_clip > 0:
            np.clip(feats, -feat_clip, feat_clip, out=feats)

        # ── Pred: predict_full_sequence sobre los últimos window_len+new_samples ──
        n_ctx = window_len + new_samples
        preds_norm = predict_full_sequence(model, feats[:, -n_ctx:].astype(np.float32), device)
        pred_raw_arr = preds_norm[:, -new_samples:]  # [output_dim, new_samples]

        # ── True position+velocity (offset dinámico del joystick) ──────────────
        off_ei, off_fi = _joy_offset("joy_ext_ind", "joy_flex_ind")
        joy_ind, vel_ind = compute_target_velocity(
            _arr("joy_ext_ind"), _arr("joy_flex_ind"), fs=fs, sigma_ms=sigma_ms,
            top_offset=off_ei, bottom_offset=off_fi)

        true_pos = np.zeros(mode)
        true_vel = np.zeros((mode, new_samples))
        true_pos[0] = float(joy_ind[-1])
        true_vel[0]  = vel_ind[-new_samples:]

        if mode == 2:
            off_em, off_fm = _joy_offset("joy_ext_med", "joy_flex_med")
            # Medio: joy_ext_med sube en extensión (→4095) y joy_flex_med baja en flexión (→0).
            # Para mostrar extensión=positivo (igual que índice) NO negamos; pasamos ya negados
            # al compute_target_velocity que internamente hace -y+offset:
            #   -(-joy_ext_med) + (-off_em) = joy_ext_med - off_em  → positivo en extensión ✓
            # Equivalente a llamar con 4095-joy_ext_med como "top" usando offset 4095-off_em.
            # Más simple: computar directo sin compute_target_velocity.
            from dataset.preprocessing import gaussian_smooth_target, central_diff
            je_med = _arr("joy_ext_med"); jf_med = _arr("joy_flex_med")
            joy_med = (gaussian_smooth_target(je_med - off_em, fs, sigma_ms)
                       + gaussian_smooth_target(-(jf_med - off_fm), fs, sigma_ms))
            vel_med = central_diff(joy_med, dt=1.0/fs)
            true_pos[1] = float(joy_med[-1])
            true_vel[1]  = vel_med[-new_samples:]

        # ── Pred velocity/position: warmup mask + EMA persistente + dead zone ───
        pred_vel = np.zeros((mode, new_samples))
        warmup_done = real_samples[0] > int(fs * 1.0)
        use_auto_dz = args.dead_zone is None
        dz = dz_state["value"] if use_auto_dz else float(args.dead_zone)
        for i in range(mode):
            v_raw = pred_raw_arr[i] * target_scales[i]
            if i == 1:
                v_raw = -v_raw
            if not warmup_done:
                v_raw[:] = 0.0
            # EMA persistente muestra a muestra
            v_smooth = np.empty_like(v_raw)
            for t, vt in enumerate(v_raw):
                pred_ema_state[i] = pred_ema_alpha * float(vt) + (1 - pred_ema_alpha) * pred_ema_state[i]
                v_smooth[t] = pred_ema_state[i]
            if not predict_accel:
                if use_auto_dz and not dz_state["computed"]:
                    pred_vel_warmup[i].extend(v_smooth.tolist())
                v_smooth = apply_dead_zone(v_smooth, dz)
            pred_vel[i] = v_smooth

        # Auto dead zone: p95 de las pred vel en los primeros 3s (asume reposo)
        if use_auto_dz and not dz_state["computed"] and real_samples[0] >= DZ_WARMUP_N:
            all_vels = []
            for wup in pred_vel_warmup:
                all_vels.extend(wup)
            if all_vels:
                dz_state["value"] = float(np.percentile(np.abs(all_vels), 95))
                dz_state["computed"] = True
                print(f"[live] dead_zone auto: {dz_state['value']:.1f}")

        return true_pos, true_vel, pred_vel

    def update(_frame):
        new_samples = 0
        while True:
            sample = stream.read()
            if sample is None:
                break
            new_samples += 1
            sample_count[0] += 1
            for ch, idx in [("emg0", 0), ("emg1", 1), ("emg2", 2), ("emg3", 3),
                             ("joy_ext_ind", 4), ("joy_flex_ind", 5),
                             ("joy_ext_med", 6), ("joy_flex_med", 7)]:
                raw[ch].append(float(sample[idx]))

        if new_samples == 0:
            return ()

        real_samples[0] += new_samples
        true_pos, true_vel, pred_vel = _compute_frame(new_samples)

        # Posición predicha: integración cinemática directa (x(t) = x(t-1) + v(t)*dt)
        # Si leak_tau > 0 se aplica decay exponencial para amortiguar drift.
        leak = dt / args.leak_tau if args.leak_tau > 0 else 0.0
        for i in range(mode):
            if predict_accel:
                # Doble integración cinemática: accel → vel → pos
                for a in pred_vel[i]:
                    accel_vel_state[i] += float(a) * dt
                    if leak > 0:
                        accel_vel_state[i] *= (1 - leak)
                pos_state[i] = np.clip(
                    pos_state[i] + accel_vel_state[i] * dt,
                    p_low, p_high)
            else:
                # Integración cinemática: pos += vel * dt (pura, sin amortiguamiento)
                for v in pred_vel[i]:
                    pos_state[i] = np.clip(
                        pos_state[i] + float(v) * dt,
                        p_low, p_high)

            ba = box_artists[i]
            ba["true_circle"].center = (-0.35, float(true_pos[i]))
            ba["pred_circle"].center = (+0.35, pos_state[i])

            va = vel_artists[i]
            k = new_samples
            va["real_buf"][:-k] = va["real_buf"][k:]
            va["pred_buf"][:-k] = va["pred_buf"][k:]
            va["real_buf"][-k:] = true_vel[i]
            va["pred_buf"][-k:] = pred_vel[i]
            va["real_line"].set_ydata(va["real_buf"])
            va["pred_line"].set_ydata(va["pred_buf"])

        status_text.set_text(f"t={sample_count[0]/fs:.1f}s")
        return ()

    anim = animation.FuncAnimation(fig, update, interval=33, blit=False, cache_frame_data=False)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    try:
        plt.show()
    finally:
        stream.close()


def main():
    args = parse_args()

    if args.port or args.mock:
        run_live(args)
        return

    if not args.csv:
        raise SystemExit("--csv es obligatorio en modo replay (o usá --port / --mock para live)")

    device = pick_device(args.device)
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    cfg = ckpt["config"]
    mode = detect_mode(ckpt)
    predict_pos = bool(cfg.get("predict_pos", False))
    print(f"Modo: {mode} dedo(s)  |  predict_pos={predict_pos}")

    # ----- Cargar dataset segun modo -----
    ckpt_sigma = float(cfg.get("sigma_ms", 50.0 if mode == 2 else 150.0))
    if mode == 1:
        ckpt_features = cfg.get("features") or list(DEFAULT_FEATURE_SET)
        ckpt_joy_center = ckpt.get("joy_center_mode") or cfg.get("joy_center_mode") or "fixed"
        ds = EMGSessionDataset(
            args.csv,
            window_length=cfg["window_length"],
            stride=1,
            config=PreprocessConfig(feature_set=tuple(ckpt_features),
                                    sigma_ms=ckpt_sigma,
                                    feature_clip=float(ckpt.get("feature_clip") or 20.0),
                                    joy_center_mode=ckpt_joy_center),
            verbose=False,
        )
        target_scales = {"single": float(ckpt["target_scale"])}
        # Para predict_pos, target_scale guarda la escala de posicion (legacy swap).
        # Para modelos normales, position_scale guarda la escala de posicion.
        if predict_pos:
            position_scales = {"single": float(ckpt["target_scale"])}
        else:
            position_scales = {"single": float(ckpt.get("position_scale", ds.position_scale))}
        _joy_raw = ds.joy.astype(np.float64)
        _rest = ds.tasks == "rest_baseline"
        _joy_center = float(np.median(_joy_raw[_rest])) if _rest.any() else float(np.median(_joy_raw))
        fingers = [{
            "key":           "single",
            "label":         "Finger",
            "true_position": _joy_raw - _joy_center,
            "true_velocity": ds.target_raw.astype(np.float64),
            "pred_idx":      0,
        }]
    else:
        ckpt_features_2f = tuple(cfg.get("features", ("rms_norm", "d_rms_norm")))
        ds = EMGSession2FingersDataset(
            args.csv,
            window_length=cfg["window_length"],
            stride=1,
            config=PreprocessConfig2F(sigma_ms=ckpt_sigma, feature_set=ckpt_features_2f),
            verbose=False,
        )
        # Aplicar z-norm con stats del entrenamiento. Formato actual: feature_center
        # + feature_scale (cubre median+MAD y mean+std). Formatos viejos por compat.
        norm_applied = False
        if cfg.get("feature_center") is not None and cfg.get("feature_scale") is not None:
            ds.apply_feature_norm(np.array(cfg["feature_center"]),
                                  np.array(cfg["feature_scale"]))
            norm_applied = True
        elif "feature_median" in cfg and "feature_mad" in cfg:
            ds.apply_feature_norm(np.array(cfg["feature_median"]),
                                  np.array(cfg["feature_mad"]))
            norm_applied = True
        elif "feature_mean" in cfg and "feature_std" in cfg:
            ds.apply_feature_norm(np.array(cfg["feature_mean"]),
                                  np.array(cfg["feature_std"]))
            norm_applied = True
        # Mismo clip que en training (valor guardado en config, default 10)
        clip_val = float(cfg.get("clip_value", 10.0))
        if norm_applied and clip_val > 0:
            np.clip(ds.features, -clip_val, clip_val, out=ds.features)
        target_scales = {
            "ind": float(ckpt.get("target_scale_ind", cfg["target_scale_ind"])),
            "med": float(ckpt.get("target_scale_med", cfg["target_scale_med"])),
        }
        # Recomputo joy y velocity raw por dedo (sin normalizar)
        from dataset.preprocessing import compute_target_velocity
        joy_ind, vel_ind = compute_target_velocity(
            ds.features[0]*0,  # placeholder, no se usa
            ds.features[0]*0,
            fs=ds.fs, sigma_ms=ckpt_sigma,
        ) if False else (None, None)
        # Mejor releer del CSV
        import pandas as pd
        df = pd.read_csv(args.csv)
        je_ind = df["joy_ext_ind"].to_numpy(np.float64)
        jf_ind = df["joy_flex_ind"].to_numpy(np.float64)
        je_med = df["joy_ext_med"].to_numpy(np.float64)
        jf_med = df["joy_flex_med"].to_numpy(np.float64)
        joy_ind, vel_ind = compute_target_velocity(
            je_ind, jf_ind, fs=ds.fs, sigma_ms=ckpt_sigma,
            top_offset=float(np.median(je_ind)),
            bottom_offset=float(np.median(jf_ind)),
        )
        joy_med, vel_med = compute_target_velocity(
            je_med, jf_med, fs=ds.fs, sigma_ms=ckpt_sigma,
            top_offset=float(np.median(je_med)),
            bottom_offset=float(np.median(jf_med)),
        )
        fingers = [
            {"key":"ind", "label":"INDICE",
             "true_position": joy_ind.astype(np.float64),
             "true_velocity": vel_ind.astype(np.float64),
             "pred_idx": 0},
            {"key":"med", "label":"MEDIO",
             "true_position": joy_med.astype(np.float64),
             "true_velocity": vel_med.astype(np.float64),
             "pred_idx": 1},
        ]

    fs = ds.fs
    dt = 1.0 / fs
    n_total = ds.features.shape[1]
    print(f"  fs = {fs:.2f} Hz   n = {n_total:,} samples   sigma_ms = {ckpt_sigma}")

    # ----- Inferencia -----
    print("Inferencia causal sobre la sesion completa...")
    preds_norm = predict_full_sequence(model, ds.features, device)  # [output_dim, T]
    if preds_norm.ndim == 1:
        preds_norm = preds_norm[None, :]

    # ----- Pipeline post-inferencia por dedo -----
    for f in fingers:
        ts = target_scales[f["key"]]
        if predict_pos:
            ps = position_scales[f["key"]]
            pv, pp = process_finger_position(
                preds_norm[f["pred_idx"]], fs, dt,
                position_scale=ps, joy_center=_joy_center,
                smooth_tau=args.vel_smooth_tau,
            )
            f["pred_velocity"] = pv
            f["pred_position"] = pp
            f["target_scale"] = ts
            print(f"  {f['label']}: position_scale={ps:.2f}  (predict_pos mode)")
        else:
            pv, pp, dz = process_finger_velocity(
                preds_norm[f["pred_idx"]], fs, dt, args,
                tasks=ds.tasks, true_position=f["true_position"],
                target_scale=ts,
            )
            f["pred_velocity"] = pv
            f["pred_position"] = pp
            f["target_scale"] = ts
            print(f"  {f['label']}: target_scale={ts:.2f}, dead_zone={dz:.2f}")

    # ----- Ventana temporal -----
    start_sample = max(0, int(args.start * fs))
    if args.duration:
        end_sample = min(start_sample + int(args.duration * fs), n_total)
    else:
        end_sample = n_total
    samples_per_frame = max(1, int(round(fs * args.speed / args.fps)))
    n_frames = (end_sample - start_sample) // samples_per_frame
    print(f"Playback: {(end_sample - start_sample)/fs:.0f}s, "
          f"speed={args.speed}x, fps={args.fps}, frames={n_frames}")

    # ----- Layout dinamico -----
    n = len(fingers)
    if n == 1:
        fig, (ax_box, ax_vel) = plt.subplots(
            2, 1, figsize=(6, 10), gridspec_kw={"height_ratios": [3, 1]})
        finger_axes = [{"box": ax_box, "vel": ax_vel}]
    else:
        fig, axs = plt.subplots(
            2, n, figsize=(6 * n, 10),
            gridspec_kw={"height_ratios": [3, 1]})
        finger_axes = [{"box": axs[0, i], "vel": axs[1, i]} for i in range(n)]

    # ----- Construccion de paneles por dedo -----
    for f, fa in zip(fingers, finger_axes):
        tp = f["true_position"]
        p_abs = max(abs(float(np.percentile(tp, 1))), abs(float(np.percentile(tp, 99))))
        p_low, p_high = -p_abs, p_abs
        margin = p_abs * 0.15
        f["box_artists"] = build_box_panel(
            fa["box"], f["label"], p_low, p_high, margin,
            float(tp[start_sample]), float(f["pred_position"][start_sample]),
        )
        p_amp = float(np.percentile(np.abs(f["true_position"]), 99))
        f["vel_artists"] = build_velocity_panel(
            fa["vel"], None, fs, (-p_amp * 1.2, p_amp * 1.2), ylabel="position (joy)",
        )

    # Status text (compartido)
    status_ax = finger_axes[0]["box"]
    status = status_ax.text(0, status_ax.get_ylim()[1] -
                            (status_ax.get_ylim()[1] - status_ax.get_ylim()[0]) * 0.04,
                            "", ha="center", fontsize=11, family="monospace")

    # ----- Checkbox para mostrar/ocultar target -----
    ax_check = fig.add_axes([0.02, 0.92, 0.13, 0.05])
    check = CheckButtons(ax_check, ["show target"], [True])
    show_target_state = {"val": True}

    def toggle_target(label):
        visible = check.get_status()[0]
        show_target_state["val"] = visible
        for f in fingers:
            f["box_artists"]["true_circle"].set_visible(visible)
            f["vel_artists"]["real_line"].set_visible(visible)
        fig.canvas.draw_idle()

    check.on_clicked(toggle_target)

    # ----- Animation update -----
    def update(frame: int):
        sample = start_sample + frame * samples_per_frame
        if sample >= end_sample:
            sample = end_sample - 1
        spf = samples_per_frame

        for f in fingers:
            ba = f["box_artists"]; va = f["vel_artists"]
            y_lo, y_hi = ba["y_lim"]
            ba["true_circle"].center = (-0.35, float(np.clip(f["true_position"][sample], y_lo, y_hi)))
            ba["pred_circle"].center = (+0.35, float(np.clip(f["pred_position"][sample], y_lo, y_hi)))
            new_real = f["true_position"][max(0, sample - spf + 1): sample + 1]
            new_pred = f["pred_position"][max(0, sample - spf + 1): sample + 1]
            k = len(new_real)
            if k > 0:
                va["real_buf"][:-k] = va["real_buf"][k:]
                va["real_buf"][-k:] = new_real
                va["pred_buf"][:-k] = va["pred_buf"][k:]
                va["pred_buf"][-k:] = new_pred
            va["real_line"].set_ydata(va["real_buf"])
            va["pred_line"].set_ydata(va["pred_buf"])
        status.set_text(f"t = {sample/fs:6.1f}s   task = {ds.tasks[sample]}")
        return ()

    interval_ms = 1000.0 / args.fps
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms,
        blit=False, repeat=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])   # deja espacio arriba para el checkbox

    if args.save:
        print(f"Guardando a {args.save}...")
        anim.save(args.save, fps=args.fps)
        print("Listo.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
