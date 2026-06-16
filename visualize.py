"""
Replay grafico de una sesion con prediccion del modelo.

Muestra una cajita vertical con dos circulos:
- Verde:  posicion REAL del dedo (joy smoothed del joystick)
- Rojo:   posicion PREDICHA (integrando la velocidad estimada con leak)

Pensado para reusar en online: misma cajita, mismo render loop;
solo cambia la fuente de los samples (CSV vs Serial).

Uso:
    python visualize.py --checkpoint checkpoints/tcn_s01_50ep/best.pt \\
                        --csv data/raw/P01/P01_S01_20260524_181522.csv
    python visualize.py ... --speed 4 --start 300 --duration 90
    python visualize.py ... --save replay.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import lfilter

from dataset import DEFAULT_FEATURE_SET, EMGSessionDataset, PreprocessConfig
from models import EMGTCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--speed", type=float, default=2.0,
                   help="playback speed (1.0 = realtime)")
    p.add_argument("--fps", type=float, default=30.0,
                   help="display update rate")
    p.add_argument("--start", type=float, default=0.0,
                   help="start time in seconds")
    p.add_argument("--duration", type=float, default=None,
                   help="duration in seconds; default = whole session")
    p.add_argument("--leak-tau", type=float, default=5.0,
                   help="leaky integrator time constant (s); smaller = mas anclaje a 0. "
                        "0 (o negativo) = sin leak, integracion pura + clipping fisico")
    p.add_argument("--vel-smooth-tau", type=float, default=0.05,
                   help="EMA tau (s) para suavizar la velocidad predicha antes de integrar; "
                        "0 = sin smoothing")
    p.add_argument("--dead-zone", type=float, default=None,
                   help="threshold (en unidades de velocidad raw) abajo del cual v_pred = 0. "
                        "default: auto = p95 de |v| durante rest_baseline")
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
def predict_full_sequence(model: EMGTCN, features: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    h = model.block1(x)
    h = model.block2(h)
    h = model.final_drop(model.final_act(model.final_conv(h)))
    h = model.head(h)
    return h[0, 0, :].cpu().numpy()


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


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    cfg = ckpt["config"]
    target_scale = float(ckpt["target_scale"])

    # Reusar el feature set y sigma_ms con el que se entreno el modelo
    ckpt_features = cfg.get("features") or list(DEFAULT_FEATURE_SET)
    ckpt_sigma = float(cfg.get("sigma_ms", 150.0))
    print(f"Cargando sesion: {args.csv}")
    print(f"Feature set del checkpoint: {ckpt_features}")
    print(f"Target sigma_ms       : {ckpt_sigma}")
    ds = EMGSessionDataset(
        args.csv,
        window_length=cfg["window_length"],
        stride=1,
        config=PreprocessConfig(feature_set=tuple(ckpt_features), sigma_ms=ckpt_sigma),
        verbose=False,
    )
    fs = ds.fs
    dt = 1.0 / fs
    print(f"  fs = {fs:.2f} Hz   n = {ds.features.shape[1]:,} samples")

    print("Inferencia causal sobre la sesion completa...")
    preds_norm = predict_full_sequence(model, ds.features, device)
    pred_velocity_raw = preds_norm * target_scale        # unidades joy/s
    true_position = ds.joy.astype(np.float64)            # unidades joy

    # --- Mascarar warmup del pipeline causal (bandpass IIR + RMS rolling + EMA) ---
    # Durante el primer ~1s los features son transients de los filtros y
    # el modelo nunca los vio durante training (el dataset excluye warmup
    # del set valido). En inferencia las predicciones ahi son nonsense
    # (visto: picos de ~5000 joy/s). Las pongo a 0 para no contaminar la
    # integral del controlador.
    warmup_samples = int(round(fs * 1.0))
    pred_velocity_raw[:warmup_samples] = 0.0

    # --- Post-procesamiento de la velocidad antes de integrar ---
    # 1. Smoothing EMA (low-pass de alta frecuencia)
    pred_velocity_smooth = smooth_velocity(pred_velocity_raw, fs, args.vel_smooth_tau)

    # 2. Dead zone (frena drift durante holds)
    if args.dead_zone is None:
        rest_mask = ds.tasks == "rest_baseline"
        if rest_mask.any():
            rest_vel = pred_velocity_smooth[rest_mask]
            dead_zone_thr = float(np.percentile(np.abs(rest_vel), 95))
        else:
            dead_zone_thr = float(np.percentile(np.abs(pred_velocity_smooth), 25))
    else:
        dead_zone_thr = float(args.dead_zone)
    print(f"  vel smoothing EMA tau = {args.vel_smooth_tau:.3f} s")
    print(f"  dead zone threshold   = {dead_zone_thr:.2f} joy/s  "
          f"(equiv ~{dead_zone_thr/target_scale*100:.1f}% del target_scale)")

    pred_velocity = apply_dead_zone(pred_velocity_smooth, dead_zone_thr)

    # Limites fisicos: usamos min/max real del joy (topes mecanicos efectivos)
    pos_min_raw = float(true_position.min())
    pos_max_raw = float(true_position.max())

    pred_position = leaky_integrate(
        pred_velocity, dt, args.leak_tau,
        initial=float(true_position[0]),
        pos_min=pos_min_raw,
        pos_max=pos_max_raw,
    )

    # Rango vertical para la cajita: percentil para evitar que los topes
    # ocupen los extremos del marco
    p_low = float(np.percentile(true_position, 1))
    p_high = float(np.percentile(true_position, 99))
    margin = (p_high - p_low) * 0.15
    y_lim = (p_low - margin, p_high + margin)
    print(f"  pos clip: [{pos_min_raw:.0f}, {pos_max_raw:.0f}]   "
          f"display: [{p_low:.0f}, {p_high:.0f}]")

    # Ventana temporal
    n_total = len(true_position)
    start_sample = max(0, int(args.start * fs))
    if args.duration:
        end_sample = min(start_sample + int(args.duration * fs), n_total)
    else:
        end_sample = n_total

    # Velocidad real (label) y velocidad predicha en unidades joy/s
    true_velocity = ds.target_raw.astype(np.float64)

    samples_per_frame = max(1, int(round(fs * args.speed / args.fps)))
    n_frames = (end_sample - start_sample) // samples_per_frame

    # Buffer scrolling para el time-series de velocidad (5 s de sesion)
    vel_buf_len = int(5.0 * fs)
    real_vel_buf = np.zeros(vel_buf_len)
    pred_vel_buf = np.zeros(vel_buf_len)
    vel_time_axis = np.linspace(-5.0, 0.0, vel_buf_len)
    # Rango simetrico para velocidad
    v_amp = float(np.percentile(np.abs(true_velocity), 99))
    v_lim = (-v_amp * 1.2, v_amp * 1.2)

    print(f"Playback: {(end_sample - start_sample)/fs:.0f}s, "
          f"speed={args.speed}x, fps={args.fps}, "
          f"samples/frame={samples_per_frame}, total_frames={n_frames}")

    # ---- Figura: cajita arriba + velocidad scrolling abajo ----
    fig, (ax, ax_vel) = plt.subplots(
        2, 1, figsize=(6, 10),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # === Panel posicion ===
    ax.set_xlim(-1, 1)
    ax.set_ylim(*y_lim)
    ax.set_xticks([])
    ax.set_ylabel("finger position (joy units)")
    ax.set_title("EMG → finger replay")

    box = patches.Rectangle(
        (-0.75, p_low - margin * 0.3),
        1.5,
        (p_high - p_low) + margin * 0.6,
        linewidth=2, edgecolor="black", facecolor="whitesmoke", zorder=0,
    )
    ax.add_patch(box)

    ax.axhline(0, color="gray", lw=0.7, linestyle="--", zorder=1)
    ax.text(0.78, 0, "  rest", va="center", fontsize=9, color="gray")
    ax.axhline(p_high, color="lightgray", lw=0.5, linestyle=":", zorder=1)
    ax.axhline(p_low, color="lightgray", lw=0.5, linestyle=":", zorder=1)

    radius_units = max(15, (p_high - p_low) * 0.04)
    true_circle = patches.Circle(
        (-0.35, float(true_position[start_sample])),
        radius=radius_units, color="seagreen", alpha=0.9, zorder=3,
    )
    pred_circle = patches.Circle(
        (+0.35, float(pred_position[start_sample])),
        radius=radius_units, color="crimson", alpha=0.9, zorder=3,
    )
    ax.add_patch(true_circle)
    ax.add_patch(pred_circle)

    ax.text(-0.35, y_lim[0] + margin * 0.4, "real",
            ha="center", fontsize=11, fontweight="bold", color="seagreen")
    ax.text(+0.35, y_lim[0] + margin * 0.4, "predicho",
            ha="center", fontsize=11, fontweight="bold", color="crimson")

    status = ax.text(0, y_lim[1] - margin * 0.35, "",
                     ha="center", fontsize=11, family="monospace")

    # === Panel velocidad scrolling ===
    ax_vel.set_xlim(-5, 0)
    ax_vel.set_ylim(*v_lim)
    ax_vel.set_xlabel("t (s, relativo a ahora)")
    ax_vel.set_ylabel("velocity (joy/s)")
    ax_vel.axhline(0, color="gray", lw=0.5, linestyle="--", zorder=1)
    ax_vel.grid(True, alpha=0.3, zorder=0)
    real_vel_line, = ax_vel.plot(
        vel_time_axis, real_vel_buf, color="seagreen", lw=1.2, label="real", zorder=2,
    )
    pred_vel_line, = ax_vel.plot(
        vel_time_axis, pred_vel_buf, color="crimson", lw=1.2,
        label="predicho", alpha=0.85, zorder=3,
    )
    ax_vel.legend(loc="upper left", fontsize=9, framealpha=0.9)

    def update(frame: int):
        sample = start_sample + frame * samples_per_frame
        if sample >= end_sample:
            sample = end_sample - 1

        # Posicion (cajita)
        true_circle.center = (-0.35, float(true_position[sample]))
        pred_circle.center = (+0.35, float(pred_position[sample]))
        status.set_text(f"t = {sample/fs:6.1f}s   task = {ds.tasks[sample]}")

        # Velocidad (scrolling)
        spf = samples_per_frame
        new_real = true_velocity[max(0, sample - spf + 1) : sample + 1]
        new_pred = pred_velocity[max(0, sample - spf + 1) : sample + 1]
        n = len(new_real)
        if n > 0:
            real_vel_buf[:-n] = real_vel_buf[n:]
            real_vel_buf[-n:] = new_real
            pred_vel_buf[:-n] = pred_vel_buf[n:]
            pred_vel_buf[-n:] = new_pred
        real_vel_line.set_ydata(real_vel_buf)
        pred_vel_line.set_ydata(pred_vel_buf)

        return true_circle, pred_circle, status, real_vel_line, pred_vel_line

    interval_ms = 1000.0 / args.fps
    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=interval_ms, blit=False,
        repeat=False,
    )

    plt.tight_layout()

    if args.save:
        print(f"Guardando a {args.save}...")
        anim.save(args.save, fps=args.fps)
        print("Listo.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
