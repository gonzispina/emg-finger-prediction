"""
Dataset 2 dedos: 16 features (8 por dedo) -> 2 velocidades.

Schema CSV esperado (binary_v2 31B, ver collection/runner.py):
    t_micros, emg0, emg1, emg2, emg3,
    joy_ext_ind, joy_flex_ind, joy_ext_med, joy_flex_med,
    trial_id, task, session_id, subject_id, placement_id, optional_notes

Mapeo EMG:
    emg0 = ext_ind   emg1 = flx_ind
    emg2 = ext_med   emg3 = flx_med

Features (16 total = 8 por dedo):
    Por dedo, replicamos el feature_set default extendido del run viejo
    ("rms_norm", "d_rms_norm", "rms_slow_norm", "d_rms_slow_norm").
    Cada feature emite 2 canales (flx, ext) -> 8 por dedo.

Target:
    [vel_ind, vel_med] en el ultimo timestep de cada ventana (horizon=1)
    o [horizon, 2] con horizon>1.
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
    compute_emg_features,
    compute_target_velocity,
    estimate_fs,
)


DEFAULT_FEATURE_SET_2F: Tuple[str, ...] = (
    "rms_norm", "d_rms_norm", "rms_slow_norm", "d_rms_slow_norm",
)


@dataclass
class PreprocessConfig2F:
    # Target
    sigma_ms: float = 50.0       # champion segun memoria 'target sigma hallazgo'
    # EMG features
    bp_low: float = 70.0
    bp_high: float = 300.0
    bp_order: int = 4
    rms_window_ms: float = 40.0
    rms_slow_window_ms: float = 150.0
    ema_tau_s: float = 1.0
    feature_set: Tuple[str, ...] = DEFAULT_FEATURE_SET_2F
    # Normalizacion
    target_scale_ind: Optional[float] = None
    target_scale_med: Optional[float] = None
    target_scale_percentile: float = 99.0
    feature_clip: Optional[float] = 20.0


@dataclass
class SessionStats2F:
    csv_path: str
    fs: float
    n_samples: int
    n_windows: int
    target_scale_ind: float
    target_scale_med: float
    feature_names: List[str] = field(default_factory=list)
    feature_mean: List[float] = field(default_factory=list)
    feature_std: List[float] = field(default_factory=list)


class EMGSession2FingersDataset(Dataset):
    """
    Dataset de UNA sesion del nuevo schema (4 EMG + 4 joys, 2 dedos).

    __getitem__(i) -> (features [16, window_length], target [2] o [horizon, 2])
    """

    def __init__(
        self,
        csv_path: str | Path,
        window_length: int = 729,            # RF default de la TCN nueva
        stride: int = 1,
        exclude_tasks: Sequence[str] = ("transition",),
        config: Optional[PreprocessConfig2F] = None,
        horizon: int = 1,
        verbose: bool = True,
        target_finger: str = "both",         # "both" | "ind" | "med"
        feature_fingers: str = "both",       # qué dedos incluir como FEATURES
    ):
        super().__init__()
        if target_finger not in ("both", "ind", "med"):
            raise ValueError(f"target_finger debe ser 'both', 'ind' o 'med', "
                             f"recibido {target_finger}")
        if feature_fingers not in ("both", "ind", "med"):
            raise ValueError(f"feature_fingers debe ser 'both', 'ind' o 'med', "
                             f"recibido {feature_fingers}")
        self.target_finger = target_finger
        self.feature_fingers = feature_fingers
        self.csv_path = Path(csv_path)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError(f"horizon debe ser >= 1, recibido {self.horizon}")
        self.input_length = self.window_length + self.horizon - 1
        self.exclude_tasks = tuple(exclude_tasks)
        self.config = config or PreprocessConfig2F()

        df = pd.read_csv(self.csv_path)
        required = {
            "t_micros",
            "emg0", "emg1", "emg2", "emg3",
            "joy_ext_ind", "joy_flex_ind", "joy_ext_med", "joy_flex_med",
            "task",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas en {self.csv_path}: {missing}")

        # ESP32 emite t_micros como uint32, overflowea cada ~71.5 min.
        # Detectamos saltos negativos y sumamos 2^32 desde ahi.
        t_us = df["t_micros"].to_numpy(dtype=np.float64)
        UINT32_RANGE = 2.0 ** 32
        dt = np.diff(t_us)
        wraps = np.where(dt < -1e9)[0]   # solo cuenta un wrap real (no jitter)
        if len(wraps) > 0:
            print(f"  t_micros overflow detectado en {len(wraps)} punto(s); aplicando unwrap.")
            for w in wraps:
                t_us[w + 1:] += UINT32_RANGE
        t_sec = t_us / 1e6
        self.fs = estimate_fs(t_sec)

        # ----- TARGET por dedo -----
        # compute_target_velocity hace: smooth(-y_top + offset) + smooth(-y_bot + offset)
        # Pasamos offset = mediana de cada eje (no el 500/550 hardcoded del Arduino).
        je_ind = df["joy_ext_ind"].to_numpy(dtype=np.float64)
        jf_ind = df["joy_flex_ind"].to_numpy(dtype=np.float64)
        je_med = df["joy_ext_med"].to_numpy(dtype=np.float64)
        jf_med = df["joy_flex_med"].to_numpy(dtype=np.float64)

        joy_ind, vel_ind = compute_target_velocity(
            je_ind, jf_ind,
            fs=self.fs,
            sigma_ms=self.config.sigma_ms,
            top_offset=float(np.median(je_ind)),
            bottom_offset=float(np.median(jf_ind)),
        )
        joy_med, vel_med = compute_target_velocity(
            je_med, jf_med,
            fs=self.fs,
            sigma_ms=self.config.sigma_ms,
            top_offset=float(np.median(je_med)),
            bottom_offset=float(np.median(jf_med)),
        )

        # ----- FEATURES por dedo -----
        # compute_emg_features espera (flexor, extensor) -> [n_feat*2, T]
        # Indice
        feats_ind, labels_ind = compute_emg_features(
            flexor=df["emg1"].to_numpy(dtype=np.float64),   # flx_ind
            extensor=df["emg0"].to_numpy(dtype=np.float64), # ext_ind
            fs=self.fs,
            feature_set=tuple(self.config.feature_set),
            bp_low=self.config.bp_low,
            bp_high=self.config.bp_high,
            bp_order=self.config.bp_order,
            rms_window_ms=self.config.rms_window_ms,
            rms_slow_window_ms=self.config.rms_slow_window_ms,
            ema_tau_s=self.config.ema_tau_s,
        )
        labels_ind = [f"{lab}_ind" for lab in labels_ind]

        # Medio
        feats_med, labels_med = compute_emg_features(
            flexor=df["emg3"].to_numpy(dtype=np.float64),   # flx_med
            extensor=df["emg2"].to_numpy(dtype=np.float64), # ext_med
            fs=self.fs,
            feature_set=tuple(self.config.feature_set),
            bp_low=self.config.bp_low,
            bp_high=self.config.bp_high,
            bp_order=self.config.bp_order,
            rms_window_ms=self.config.rms_window_ms,
            rms_slow_window_ms=self.config.rms_slow_window_ms,
            ema_tau_s=self.config.ema_tau_s,
        )
        labels_med = [f"{lab}_med" for lab in labels_med]

        # Filtrar canales segun feature_fingers
        if self.feature_fingers == "ind":
            features = feats_ind
            feature_names = labels_ind
        elif self.feature_fingers == "med":
            features = feats_med
            feature_names = labels_med
        else:  # both
            features = np.concatenate([feats_ind, feats_med], axis=0)
            feature_names = labels_ind + labels_med
        self.feature_names = tuple(feature_names)

        # Clip seguridad
        if self.config.feature_clip is not None:
            np.clip(features, -self.config.feature_clip, self.config.feature_clip,
                    out=features)

        # ----- Normalizacion del target por dedo -----
        def _scale(v, override):
            if override is not None:
                return float(override)
            s = float(np.percentile(np.abs(v), self.config.target_scale_percentile))
            return s if s > 0 else 1.0

        self.target_scale_ind = _scale(vel_ind, self.config.target_scale_ind)
        self.target_scale_med = _scale(vel_med, self.config.target_scale_med)
        vel_ind_n = (vel_ind / self.target_scale_ind).astype(np.float32)
        vel_med_n = (vel_med / self.target_scale_med).astype(np.float32)
        self.target = np.stack([vel_ind_n, vel_med_n], axis=0)   # [2, T]

        # ----- Mascara de validez (igual que el viejo) -----
        tasks = df["task"].to_numpy()
        valid = np.ones(len(df), dtype=bool)
        for t in self.exclude_tasks:
            valid &= tasks != t
        warmup = int(round(self.fs * 1.0))   # bandpass/RMS/EMA warmup ~1s
        valid[:warmup] = False

        last_indices = np.arange(self.input_length - 1, len(df), self.stride)
        if self.horizon == 1:
            last_indices = last_indices[valid[last_indices]]
        else:
            keep = np.ones(len(last_indices), dtype=bool)
            for off in range(self.horizon):
                keep &= valid[last_indices - off]
            last_indices = last_indices[keep]

        self.features = features.astype(np.float32)   # [16, T]
        self.tasks = tasks
        self.indices = last_indices.astype(np.int64)

        n_ch = self.features.shape[0]
        self.stats = SessionStats2F(
            csv_path=str(self.csv_path),
            fs=self.fs,
            n_samples=len(df),
            n_windows=len(self.indices),
            target_scale_ind=self.target_scale_ind,
            target_scale_med=self.target_scale_med,
            feature_names=list(self.feature_names),
            feature_mean=[float(self.features[c].mean()) for c in range(n_ch)],
            feature_std=[float(self.features[c].std()) for c in range(n_ch)],
        )

        if verbose:
            self.print_summary()

    def apply_feature_norm(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Aplica z-normalizacion in-place a las features con mean/std externos.
        mean, std: shape [n_channels] = [16].
        Para que train e inferencia usen las MISMAS stats globales calculadas
        sobre el dataset de training."""
        mean = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(-1, 1)
        std = np.where(std < 1e-8, 1.0, std).astype(np.float32)
        self.features = ((self.features - mean) / std).astype(np.float32)
        # Actualizar stats descriptivos para el print
        n_ch = self.features.shape[0]
        self.stats.feature_mean = [float(self.features[c].mean()) for c in range(n_ch)]
        self.stats.feature_std = [float(self.features[c].std()) for c in range(n_ch)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        horizon==1: (features [16, window_length], target [2])
        horizon >1: (features [16, window_length + horizon - 1], target [horizon, 2])
        """
        last = int(self.indices[idx])
        start = last - self.input_length + 1
        x = self.features[:, start: last + 1]
        if self.horizon == 1:
            y_full = self.target[:, last].astype(np.float32)          # [2]
            if self.target_finger == "ind":
                y = y_full[0:1]
            elif self.target_finger == "med":
                y = y_full[1:2]
            else:
                y = y_full
        else:
            first = last - self.horizon + 1
            y_full = self.target[:, first: last + 1].T.astype(np.float32)  # [horizon, 2]
            if self.target_finger == "ind":
                y = y_full[:, 0:1]
            elif self.target_finger == "med":
                y = y_full[:, 1:2]
            else:
                y = y_full
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )

    def task_at(self, idx: int) -> str:
        return str(self.tasks[int(self.indices[idx])])

    def print_summary(self) -> None:
        s = self.stats
        print(f"\n== EMGSession2FingersDataset ==")
        print(f"  csv          : {Path(s.csv_path).name}")
        print(f"  fs estimada  : {s.fs:.2f} Hz")
        print(f"  n_samples    : {s.n_samples:,}")
        print(f"  window_length: {self.window_length}  "
              f"(~{self.window_length / s.fs * 1000:.0f} ms)")
        print(f"  stride       : {self.stride}   horizon: {self.horizon}")
        print(f"  n_windows    : {s.n_windows:,}  (post-exclude {self.exclude_tasks})")
        print(f"  target_scale : ind={s.target_scale_ind:.4f}  med={s.target_scale_med:.4f}  "
              f"(p{self.config.target_scale_percentile:.0f})")
        print(f"  features ({len(self.feature_names)}):")
        for c, name in enumerate(self.feature_names):
            print(f"    [{c:>2}] {name:<28} "
                  f"mean={s.feature_mean[c]:+.3f}  std={s.feature_std[c]:.3f}")
        print()


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/raw/P01/P01_S05_20260618_172229.csv"
    ds = EMGSession2FingersDataset(csv, window_length=729)
    x, y = ds[0]
    print(f"sample[0]: x.shape={tuple(x.shape)}  y.shape={tuple(y.shape)}")
    print(f"           y values: vel_ind={y[0]:+.3f}  vel_med={y[1]:+.3f}")
