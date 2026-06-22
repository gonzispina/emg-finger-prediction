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

    # Sin target visible
    python visualize.py ... --hide-target
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.widgets import CheckButtons
from scipy.signal import lfilter

from dataset import DEFAULT_FEATURE_SET, EMGSessionDataset, PreprocessConfig
from dataset.emg_dataset_2fingers import EMGSession2FingersDataset, PreprocessConfig2F
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
    return {"true_circle": true_circle, "pred_circle": pred_circle, "y_lim": y_lim}


def build_velocity_panel(ax, title, fs, v_lim):
    """Velocidad scrolling con 2 lineas. Devuelve dict con buffers + lineas."""
    vel_buf_len = int(5.0 * fs)
    real_buf = np.zeros(vel_buf_len)
    pred_buf = np.zeros(vel_buf_len)
    time_axis = np.linspace(-5.0, 0.0, vel_buf_len)

    ax.set_xlim(-5, 0); ax.set_ylim(*v_lim)
    ax.set_xlabel("t (s, relativo a ahora)")
    ax.set_ylabel("vel (joy/s)")
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


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    cfg = ckpt["config"]
    mode = detect_mode(ckpt)
    print(f"Modo: {mode} dedo(s)")

    # ----- Cargar dataset segun modo -----
    ckpt_sigma = float(cfg.get("sigma_ms", 50.0 if mode == 2 else 150.0))
    if mode == 1:
        ckpt_features = cfg.get("features") or list(DEFAULT_FEATURE_SET)
        ds = EMGSessionDataset(
            args.csv,
            window_length=cfg["window_length"],
            stride=1,
            config=PreprocessConfig(feature_set=tuple(ckpt_features),
                                    sigma_ms=ckpt_sigma),
            verbose=False,
        )
        target_scales = {"single": float(ckpt["target_scale"])}
        fingers = [{
            "key":           "single",
            "label":         "Finger",
            "true_position": ds.joy.astype(np.float64),
            "true_velocity": ds.target_raw.astype(np.float64),
            "pred_idx":      0,
        }]
    else:
        ds = EMGSession2FingersDataset(
            args.csv,
            window_length=cfg["window_length"],
            stride=1,
            config=PreprocessConfig2F(sigma_ms=ckpt_sigma),
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
        p_low = float(np.percentile(tp, 1))
        p_high = float(np.percentile(tp, 99))
        margin = (p_high - p_low) * 0.15
        f["box_artists"] = build_box_panel(
            fa["box"], f["label"], p_low, p_high, margin,
            float(tp[start_sample]), float(f["pred_position"][start_sample]),
        )
        v_amp = float(np.percentile(np.abs(f["true_velocity"]), 99))
        f["vel_artists"] = build_velocity_panel(
            fa["vel"], None, fs, (-v_amp * 1.2, v_amp * 1.2),
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
            ba["true_circle"].center = (-0.35, float(f["true_position"][sample]))
            ba["pred_circle"].center = (+0.35, float(f["pred_position"][sample]))
            new_real = f["true_velocity"][max(0, sample - spf + 1): sample + 1]
            new_pred = f["pred_velocity"][max(0, sample - spf + 1): sample + 1]
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
