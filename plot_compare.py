"""
Plot comparativo de predicciones de varios checkpoints sobre un CSV de sesion.

Para cada task seleccionada encuentra una region contigua en el CSV y
plotea el target (velocidad ground-truth) con las predicciones de cada
checkpoint superpuestas. Permite comparar visualmente diferencias de
suavidad / lag / amplitud entre modelos.

Uso:
    python plot_compare.py \\
        --csv data/raw/P01/P01_S03_20260525_194322.csv \\
        --checkpoints \\
            checkpoints/tcn_s02s03_slowrms/best.pt:baseline \\
            checkpoints/tcn_s02s03_K10_l0p0/best.pt:K10/lam0 \\
            checkpoints/tcn_s02s03_K10_l0p5/best.pt:K10/lam0.5 \\
        --tasks slow_cycles fast_flex holds_intermediate \\
                microcorrections free_controlled rest_baseline \\
        --out plots/sweep_compare.png

Notas:
- Cada checkpoint puede tener features distintas; se recomputan acordes.
- El target se grafica una sola vez (lo toma del primer checkpoint).
- predict_full_sequence devuelve preds causales por timestep, asi que
  las primeras window_length-1 muestras tienen contexto incompleto.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from analyze import load_model, predict_full_sequence, pick_device
from dataset import PreprocessConfig, EMGSessionDataset


DEFAULT_TASKS = [
    ("slow_cycles", 4000),
    ("fast_flex", 800),
    ("holds_intermediate", 4000),
    ("microcorrections", 2000),
    ("free_controlled", 3000),
    ("rest_baseline", 1500),
]

DEFAULT_COLORS = ["#666666", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument("--csv", required=True, help="CSV de sesion sobre el que plotear")
    p.add_argument("--checkpoints", required=True, nargs="+",
                   help="<path>:<label> entries. Label se usa en la leyenda.")
    p.add_argument("--tasks", nargs="+", default=None,
                   help="Tasks a plotear. Default: slow_cycles, fast_flex, holds, micro, free, rest.")
    p.add_argument("--win-lens", nargs="+", type=int, default=None,
                   help="Tamaño de ventana por task (mismo orden que --tasks).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--out", default="plots/compare.png", help="output PNG path")
    return p.parse_args()


def find_segments(tasks_arr: np.ndarray, task_name: str, min_len: int) -> List[Tuple[int, int]]:
    mask = tasks_arr == task_name
    if mask.sum() == 0:
        return []
    transitions = np.diff(mask.astype(int))
    starts = np.where(transitions == 1)[0] + 1
    ends = np.where(transitions == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    return [(s, e) for s, e in zip(starts, ends) if e - s >= min_len]


def pick_val_window(segs: List[Tuple[int, int]], win_len: int,
                    val_fraction: float = 0.2) -> Tuple[int, int]:
    """
    Devuelve (a, b) — ventana centrada en el VAL de cada bloque.
    El split estratificado usa el ultimo val_fraction de cada bloque
    como val, asi que tomamos una ventana centrada en esa region.
    Se queda con el segmento mas grande disponible y le aplica el offset.
    """
    if not segs:
        return None
    s, e = max(segs, key=lambda x: x[1] - x[0])
    # Val arranca en el ~80% del segmento. Centramos la ventana ahi.
    val_start = s + int((e - s) * (1 - val_fraction))
    val_end = e
    val_mid = (val_start + val_end) // 2
    half = win_len // 2
    a = max(s, val_mid - half)
    b = min(e, a + win_len)
    return a, b


def run_checkpoint(ckpt_path: str, csv_path: str, device: torch.device):
    """
    Carga checkpoint, recomputa features del CSV con su config, corre
    forward pass causal y devuelve (preds [T], target_norm [T], tasks [T]).
    """
    model, ckpt = load_model(ckpt_path, device)
    cfg = ckpt["config"]
    feature_set = tuple(cfg.get("features", ["rms_norm", "d_rms_norm"]))
    pc = PreprocessConfig(feature_set=feature_set)
    ds = EMGSessionDataset(
        csv_path=csv_path,
        window_length=cfg.get("window_length", 243),
        stride=1,
        config=pc,
        horizon=1,                                 # solo necesito features y target
        verbose=False,
    )
    preds = predict_full_sequence(model, ds.features, device)   # [T] en escala del CKPT
    # Re-escalar pred a la misma escala del target normalizado del CSV:
    # checkpoint guardó target_scale del training; el dataset reescalo el CSV
    # con su propio percentil. Para comparar normalizado a normalizado,
    # los pred del modelo ya estan en la escala de training. Re-normalizamos
    # a la escala del val actual.
    ckpt_scale = ckpt.get("target_scale", 1.0)
    preds_raw = preds * ckpt_scale                 # velocity en unidades crudas
    preds_norm = preds_raw / ds.target_scale       # re-normalizado para comparar
    target_norm = ds.target                        # target ya normalizado por p99 local
    return preds_norm, target_norm, ds.tasks


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")

    # Parsear checkpoints
    entries = []
    for spec in args.checkpoints:
        if ":" in spec:
            path, label = spec.split(":", 1)
        else:
            path, label = spec, Path(spec).parent.name
        entries.append((path, label))

    # Tasks
    if args.tasks is None:
        tasks_to_plot = DEFAULT_TASKS
    else:
        win_lens = args.win_lens or [800] * len(args.tasks)
        if len(win_lens) != len(args.tasks):
            raise SystemExit("--win-lens debe coincidir en largo con --tasks")
        tasks_to_plot = list(zip(args.tasks, win_lens))

    print(f"\nCargando {len(entries)} checkpoints sobre {Path(args.csv).name}:")
    target = None
    tasks_arr = None
    preds_by_model = {}
    for path, label in entries:
        print(f"  - {label:<20} {path}")
        preds, tgt, ts = run_checkpoint(path, args.csv, device)
        preds_by_model[label] = preds
        if target is None:
            target = tgt
            tasks_arr = ts

    # Plot
    n = len(tasks_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (tname, win_len) in zip(axes, tasks_to_plot):
        segs = find_segments(tasks_arr, tname, min_len=win_len)
        if not segs:
            ax.text(0.5, 0.5, f"no contiguous segment ≥ {win_len} for {tname}",
                    ha="center", transform=ax.transAxes)
            ax.set_title(tname)
            continue
        a, b = pick_val_window(segs, win_len=win_len, val_fraction=0.2)
        x = np.arange(b - a)

        ax.plot(x, target[a:b], color="k", lw=2.0, label="target", alpha=0.85)
        for i, (_, label) in enumerate(entries):
            color = DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
            ax.plot(x, preds_by_model[label][a:b], color=color, lw=1.0,
                    label=label, alpha=0.78)

        ax.set_title(f"{tname}  (samples {a}..{b}, len {b-a})")
        ax.set_ylabel("vel norm")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=len(entries) + 1)

    axes[-1].set_xlabel("sample index del CSV")
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
