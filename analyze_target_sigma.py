"""
Analiza como afecta el parametro `sigma_ms` del gaussian smoothing del
target a las estadisticas y forma de la velocidad. Util para diagnosticar
si el smoothing actual esta aplastando picos predecibles y limitando el
techo de R² alcanzable.

Genera:
  1. Tabla de stats globales: std, p99, p99.5, max de |velocity| por sigma.
  2. Tabla de std(velocity) por task y por sigma — clave para ver si el
     smoothing comprime las amplitudes hasta el punto de que tasks rapidas
     y lentas se vuelven indistinguibles.
  3. Plot 3x1: fast_flex, slow_cycles, holds_intermediate, mostrando la
     misma ventana temporal con la velocity reconstruida para varios sigma.

Uso:
    python analyze_target_sigma.py --csv data/raw/P01/P01_S03_*.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset.preprocessing import compute_target_velocity, estimate_fs


DEFAULT_SIGMAS = [10, 30, 50, 100, 150, 250]   # ms


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="CSV de sesion a analizar")
    p.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS,
                   help="lista de sigma_ms a comparar")
    p.add_argument("--out", default="plots/target_sigma_sweep.png",
                   help="path del PNG de salida")
    p.add_argument("--tasks", nargs="+",
                   default=["fast_flex", "slow_cycles", "holds_intermediate"],
                   help="tasks a plotear (una por fila del PNG)")
    p.add_argument("--spans", type=int, nargs="+",
                   default=[1500, 4000, 2500],
                   help="span en samples por task (mismo orden que --tasks)")
    return p.parse_args()


def find_largest_segment(tasks_arr: np.ndarray, task_name: str):
    mask = tasks_arr == task_name
    if mask.sum() == 0:
        return None
    tr = np.diff(mask.astype(int))
    starts = np.where(tr == 1)[0] + 1
    ends = np.where(tr == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    if len(starts) == 0:
        return None
    return max(zip(starts, ends), key=lambda x: x[1] - x[0])


def main():
    args = parse_args()
    csv_path = Path(args.csv)

    df = pd.read_csv(csv_path)
    t_sec = df["t_micros"].to_numpy(dtype=np.float64) / 1e6
    fs = estimate_fs(t_sec)
    y_top = df["y_top"].to_numpy()
    y_bot = df["y_bottom"].to_numpy()
    tasks = df["task"].to_numpy()

    print(f"\nCSV: {csv_path.name}  (n={len(df):,}, fs={fs:.2f} Hz)")

    # Reconstruir velocity para cada sigma
    velocities = {}
    for sigma in args.sigmas:
        _, v = compute_target_velocity(y_top, y_bot, fs=fs, sigma_ms=sigma)
        velocities[sigma] = v

    # Tabla 1: stats globales de |v|
    print(f"\n=== Stats de velocidad |v| por sigma_ms ===")
    print(f"{'sigma_ms':>8}  {'std':>10}  {'p99':>10}  {'p99.5':>10}  {'max':>10}")
    print("-" * 56)
    for sigma in args.sigmas:
        v = velocities[sigma]
        print(f"{sigma:>8.0f}  {v.std():>10.2f}  {np.percentile(np.abs(v), 99):>10.2f}  "
              f"{np.percentile(np.abs(v), 99.5):>10.2f}  {np.abs(v).max():>10.2f}")

    # Tabla 2: std(v) por task y sigma
    print(f"\n=== std(v) por task y sigma_ms ===")
    sig_labels = "  ".join(f"σ={int(s):>3}" for s in args.sigmas)
    header = f"{'task':<22} {sig_labels}"
    print(header)
    print("-" * len(header))
    for tname in sorted(set(tasks.tolist())):
        mask = tasks == tname
        if mask.sum() < 100:
            continue
        row = [f"{velocities[s][mask].std():>7.2f}" for s in args.sigmas]
        print(f"{tname:<22} " + "  ".join(row))

    # Plot: una fila por task
    if len(args.spans) != len(args.tasks):
        raise SystemExit("--spans debe coincidir en cantidad con --tasks")
    n = len(args.tasks)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 0.85, len(args.sigmas)))

    for ax, tname, span in zip(axes, args.tasks, args.spans):
        seg = find_largest_segment(tasks, tname)
        if seg is None:
            ax.text(0.5, 0.5, f"no segment for {tname}",
                    ha="center", transform=ax.transAxes)
            ax.set_title(tname)
            continue
        s, e = seg
        mid = (s + e) // 2
        a, b = max(s, mid - span // 2), min(e, mid + span // 2)
        x = np.arange(b - a)
        for sigma, color in zip(args.sigmas, colors):
            v = velocities[sigma][a:b]
            is_default = abs(sigma - 150) < 1e-6
            ax.plot(x, v, color=color, lw=2.0 if is_default else 1.0,
                    label=f"σ={int(sigma)}ms" + (" (default)" if is_default else ""),
                    alpha=0.8)
        ax.set_title(f"{tname}  (samples {a}..{b})")
        ax.set_ylabel("velocity (raw)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, ncol=len(args.sigmas))

    axes[-1].set_xlabel("sample del CSV")
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
