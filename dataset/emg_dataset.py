"""
PyTorch Dataset que carga un CSV de sesion y produce ventanas
(features, target_velocity_en_el_ultimo_timestep).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import (
    DEFAULT_FEATURE_SET,
    central_diff,
    compute_emg_features,
    compute_target_velocity,
    estimate_fs,
)


@dataclass
class PreprocessConfig:
    # Target
    sigma_ms: float = 150.0
    # Sigma adaptivo por task (None = usar sigma_ms para todo)
    # slow tasks (slow_flex_ind, slow_ext_ind, holds_intermediate_ind, slow_cycles_ind)
    adaptive_sigma_slow_ms: Optional[float] = None
    top_offset: float = 500.0
    bottom_offset: float = 550.0
    # Centering del joystick: "fixed" usa top/bottom_offset hardcoded;
    # "rest" usa mediana durante task rest_baseline (o mediana global si no hay);
    # "median" siempre usa mediana global.
    joy_center_mode: str = "fixed"
    # EMG features
    bp_low: float = 70.0
    bp_high: float = 300.0
    bp_order: int = 4
    rms_window_ms: float = 40.0
    rms_slow_window_ms: float = 150.0
    wl_window_ms: float = 40.0
    ssc_window_ms: float = 40.0
    ssc_threshold_rel: float = 0.1
    ema_tau_s: float = 1.0
    feature_set: Tuple[str, ...] = DEFAULT_FEATURE_SET
    # Atenuacion EMG: comprime picos de RMS sobre el baseline de rest_baseline.
    # 1.0 = sin atenuacion. Aplica a todas las tasks excepto rest_baseline y noise_*.
    # emg_attenuation_flx / _ext sobreescriben emg_attenuation para cada musculo.
    emg_attenuation: float = 1.0
    emg_attenuation_flx: Optional[float] = None
    emg_attenuation_ext: Optional[float] = None
    # Normalizacion
    target_scale: Optional[float] = None
    target_scale_percentile: float = 99.0
    feature_clip: Optional[float] = 20.0


@dataclass
class SessionStats:
    csv_path: str
    fs: float
    n_samples: int
    n_windows: int
    target_scale: float
    position_scale: float = 0.0
    feature_mean: List[float] = field(default_factory=list)
    feature_std: List[float] = field(default_factory=list)
    feature_min: List[float] = field(default_factory=list)
    feature_max: List[float] = field(default_factory=list)
    target_min: float = 0.0
    target_max: float = 0.0
    target_std: float = 0.0


class EMGSessionDataset(Dataset):
    """
    Dataset de una sola sesion.

    Carga el CSV, computa fs desde t_micros, deriva features EMG (4 canales)
    y target (velocidad del joystick en el ultimo timestep de cada ventana).

    __getitem__(i) -> (features [4, window_length], target scalar).

    Windowing:
        - cada ventana termina en un indice 'last' del CSV.
        - last >= window_length - 1.
        - last se descarta si tasks[last] esta en exclude_tasks.
        - stride controla el paso entre ventanas consecutivas.
    """

    # Mantenido por compatibilidad con codigo viejo. El set real de cada
    # instancia esta en self.feature_names.
    FEATURE_NAMES = ("flx_rms_norm", "ext_rms_norm", "flx_d_rms_norm", "ext_d_rms_norm")

    def __init__(
        self,
        csv_path: str | Path,
        window_length: int = 243,
        stride: int = 1,
        exclude_tasks: Sequence[str] = ("transition",),
        config: Optional[PreprocessConfig] = None,
        horizon: int = 1,
        verbose: bool = True,
    ):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {self.horizon}")
        # Input se extiende hacia atras para que los K outputs finales vean RF completo.
        self.input_length = self.window_length + self.horizon - 1
        self.exclude_tasks = tuple(exclude_tasks)
        self.config = config or PreprocessConfig()

        df = pd.read_csv(self.csv_path)
        required = {"t_micros", "joy_ext_ind", "joy_flex_ind", "emg0", "emg1", "task"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {self.csv_path}: {missing}")

        t_sec = df["t_micros"].to_numpy(dtype=np.float64) / 1e6
        self.fs = estimate_fs(t_sec)

        y_top = df["joy_ext_ind"].to_numpy()
        y_bottom = df["joy_flex_ind"].to_numpy()
        flexor = df["emg1"].to_numpy()
        extensor = df["emg0"].to_numpy()

        tasks_col = df["task"].to_numpy()

        # Offsets del joystick: fijos o dinámicos por sesión
        jcm = self.config.joy_center_mode
        if jcm == "fixed":
            top_offset    = self.config.top_offset
            bottom_offset = self.config.bottom_offset
        else:
            rest_mask = tasks_col == "rest_baseline"
            if jcm == "rest" and rest_mask.any():
                top_offset    = float(np.median(y_top[rest_mask]))
                bottom_offset = float(np.median(y_bottom[rest_mask]))
            else:
                top_offset    = float(np.median(y_top))
                bottom_offset = float(np.median(y_bottom))

        joy, velocity = compute_target_velocity(
            y_top, y_bottom,
            fs=self.fs,
            sigma_ms=self.config.sigma_ms,
            top_offset=top_offset,
            bottom_offset=bottom_offset,
        )

        # Sigma adaptivo por task: slow tasks reciben un sigma mas grande para
        # suavizar el stick-slip del joystick sin perder amplitud.
        _SLOW_TASKS = frozenset({
            "slow_flex_ind", "slow_ext_ind", "holds_intermediate_ind", "slow_cycles_ind",
        })
        if self.config.adaptive_sigma_slow_ms is not None:
            sigma_slow = self.config.adaptive_sigma_slow_ms
            if sigma_slow != self.config.sigma_ms:
                _, vel_slow = compute_target_velocity(
                    y_top, y_bottom,
                    fs=self.fs,
                    sigma_ms=sigma_slow,
                    top_offset=top_offset,
                    bottom_offset=bottom_offset,
                )
                slow_mask = np.isin(tasks_col, list(_SLOW_TASKS))
                velocity[slow_mask] = vel_slow[slow_mask]

        feats, feat_labels = compute_emg_features(
            flexor, extensor,
            fs=self.fs,
            feature_set=tuple(self.config.feature_set),
            bp_low=self.config.bp_low,
            bp_high=self.config.bp_high,
            bp_order=self.config.bp_order,
            rms_window_ms=self.config.rms_window_ms,
            rms_slow_window_ms=self.config.rms_slow_window_ms,
            wl_window_ms=self.config.wl_window_ms,
            ssc_window_ms=self.config.ssc_window_ms,
            ssc_threshold_rel=self.config.ssc_threshold_rel,
            ema_tau_s=self.config.ema_tau_s,
        )
        self.feature_names = tuple(feat_labels)

        # Atenuacion EMG: comprime picos de RMS sobre el baseline por sesion.
        # baseline = mediana del RMS durante los ultimos 10 s de rest_baseline.
        # Si emg_attenuation_flx/_ext estan definidos, se usan por canal;
        # si no, se usa emg_attenuation global.
        _SKIP_ATTN = frozenset({"rest_baseline", "noise_other_fingers", "noise_wrist_grip"})
        alpha_flx = self.config.emg_attenuation_flx
        alpha_ext = self.config.emg_attenuation_ext
        alpha_global = self.config.emg_attenuation
        _any_attn = (
            (alpha_flx is not None and alpha_flx < 1.0)
            or (alpha_ext is not None and alpha_ext < 1.0)
            or (alpha_flx is None and alpha_ext is None and alpha_global < 1.0)
        )
        if _any_attn:
            rest_idx = np.where(tasks_col == "rest_baseline")[0]
            if len(rest_idx) > 0:
                last10_mask = rest_idx >= rest_idx[-1] - int(10.0 * self.fs)
                last10_idx = rest_idx[last10_mask]
                attn_mask = ~np.isin(tasks_col, list(_SKIP_ATTN))
                for c, name in enumerate(feat_labels):
                    if "flx" in name:
                        alpha = alpha_flx if alpha_flx is not None else alpha_global
                    else:
                        alpha = alpha_ext if alpha_ext is not None else alpha_global
                    if alpha >= 1.0:
                        continue
                    baseline_c = float(np.median(feats[c, last10_idx]))
                    feats[c, attn_mask] = np.where(
                        feats[c, attn_mask] > baseline_c,
                        baseline_c + (feats[c, attn_mask] - baseline_c) * alpha,
                        feats[c, attn_mask],
                    )

        # Normalizacion de target — escala robusta por percentil 99 si no se da
        if self.config.target_scale is None:
            scale = float(np.percentile(np.abs(velocity), self.config.target_scale_percentile))
            if scale <= 0:
                scale = 1.0
        else:
            scale = float(self.config.target_scale)
        self.target_scale = scale
        target_norm = (velocity / scale).astype(np.float32)

        # Position target normalization (mismo enfoque que velocity)
        pos_scale = float(np.percentile(np.abs(joy), self.config.target_scale_percentile))
        if pos_scale <= 0:
            pos_scale = 1.0
        self.position_scale = pos_scale
        position_norm = (joy / pos_scale).astype(np.float32)

        # Acceleration target: d(velocity)/dt (central diff para simetria con velocity)
        accel_raw = central_diff(velocity, dt=1.0 / self.fs)
        accel_scale = float(np.percentile(np.abs(accel_raw), self.config.target_scale_percentile))
        if accel_scale <= 0:
            accel_scale = 1.0
        self.accel_scale = accel_scale
        self.accel_raw = accel_raw.astype(np.float32)
        self.accel = (accel_raw / accel_scale).astype(np.float32)

        # Clipping opcional de features para acotar spikes del /EMA
        if self.config.feature_clip is not None:
            np.clip(feats, -self.config.feature_clip, self.config.feature_clip, out=feats)

        # Mascara de samples validos para SUPERVISION (last sample de la ventana).
        valid = np.ones(len(df), dtype=bool)
        for t in self.exclude_tasks:
            valid &= tasks_col != t
        # Tambien evitar muestras durante el warm-up del bandpass/RMS/EMA (~1s)
        warmup = int(round(self.fs * 1.0))
        valid[:warmup] = False

        # last_idx = ultimo timestep del horizonte de output. Necesitamos:
        #   1) suficiente historia para que la conv vea input_length samples:
        #      last_idx >= input_length - 1
        #   2) todos los timesteps del horizonte [last-K+1 .. last] sean validos.
        last_indices = np.arange(self.input_length - 1, len(df), self.stride)
        if self.horizon == 1:
            last_indices = last_indices[valid[last_indices]]
        else:
            keep = np.ones(len(last_indices), dtype=bool)
            for offset in range(self.horizon):
                keep &= valid[last_indices - offset]
            last_indices = last_indices[keep]

        # Guardamos arrays
        self.features = feats.astype(np.float32)        # [C, T]
        self.target = target_norm                       # [T] velocity normalizada
        self.target_raw = velocity.astype(np.float32)   # [T] velocity pre-normalizacion
        self.position = position_norm                   # [T] position normalizada (joy/pos_scale)
        self.joy = joy.astype(np.float32)               # [T] position smoothed (sin normalizar)
        self.tasks = tasks_col
        self.indices = last_indices.astype(np.int64)

        n_ch = self.features.shape[0]
        self.stats = SessionStats(
            csv_path=str(self.csv_path),
            fs=self.fs,
            n_samples=len(df),
            n_windows=len(self.indices),
            target_scale=self.target_scale,
            position_scale=self.position_scale,
            feature_mean=[float(self.features[c].mean()) for c in range(n_ch)],
            feature_std=[float(self.features[c].std()) for c in range(n_ch)],
            feature_min=[float(self.features[c].min()) for c in range(n_ch)],
            feature_max=[float(self.features[c].max()) for c in range(n_ch)],
            target_min=float(self.target.min()),
            target_max=float(self.target.max()),
            target_std=float(self.target.std()),
        )

        if verbose:
            self.print_summary()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Devuelve (features, target).

        Si horizon == 1:
            features: [C, window_length]
            target:   [2]                 (vel_norm, pos_norm) en el ultimo timestep
        Si horizon  > 1:
            features: [C, window_length + horizon - 1]
            target:   [horizon, 2]        (vel_norm, pos_norm) en los ultimos K timesteps

        El loop de training decide si usa solo vel (single task) o vel+pos (multi).
        Para entrenamiento con regularizacion Sobolev (--accel-weight), usar
        horizon >= 2 y computar MSE(Δpred, Δtarget) sobre el horizonte.
        """
        last = int(self.indices[idx])
        start = last - self.input_length + 1
        x = self.features[:, start : last + 1]
        if self.horizon == 1:
            y = np.array([self.target[last], self.position[last]], dtype=np.float32)
        else:
            first = last - self.horizon + 1
            vel_seq = self.target[first : last + 1]
            pos_seq = self.position[first : last + 1]
            y = np.stack([vel_seq, pos_seq], axis=1).astype(np.float32)  # [K, 2]
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(y),
        )

    def apply_feature_norm(self, center: np.ndarray, scale: np.ndarray) -> None:
        """Z-normaliza features in-place por canal: (x - center) / scale."""
        self.features = ((self.features - center[:, None]) / scale[:, None]).astype(np.float32)

    def task_at(self, idx: int) -> str:
        return str(self.tasks[int(self.indices[idx])])

    def print_summary(self) -> None:
        s = self.stats
        print(f"\n== EMGSessionDataset ==")
        print(f"  csv          : {Path(s.csv_path).name}")
        print(f"  fs estimada  : {s.fs:.2f} Hz  (dt = {1000/s.fs:.3f} ms)")
        print(f"  n_samples    : {s.n_samples:,}")
        print(f"  window_length: {self.window_length}  (~{self.window_length / s.fs * 1000:.0f} ms, RF objetivo)")
        if self.horizon > 1:
            print(f"  horizon      : {self.horizon}  -> input_length {self.input_length} "
                  f"(~{self.input_length / s.fs * 1000:.0f} ms)")
        print(f"  stride       : {self.stride}")
        print(f"  n_windows    : {s.n_windows:,}  (post-exclude {self.exclude_tasks})")
        print(f"  target_scale : vel={s.target_scale:.4f}  pos={s.position_scale:.4f}  "
              f"(p{self.config.target_scale_percentile:.0f})")
        print(f"  target  norm : min={s.target_min:.3f}  max={s.target_max:.3f}  std={s.target_std:.3f}")
        print(f"  features (post-clip {self.config.feature_clip}, n={len(self.feature_names)}):")
        for c, name in enumerate(self.feature_names):
            print(f"    [{c}] {name:<20} mean={s.feature_mean[c]:+.3f}  std={s.feature_std[c]:.3f}  "
                  f"min={s.feature_min[c]:+.3f}  max={s.feature_max[c]:+.3f}")
        print()
