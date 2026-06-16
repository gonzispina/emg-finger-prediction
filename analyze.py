"""
Analisis offline de un checkpoint entrenado.

- Carga el modelo y aplica una pasada causal sobre TODA la sesion.
- Por cada task calcula: n samples, MAE, RMSE, Pearson R, R^2.
- Imprime tabla, guarda CSV y un bar chart con R^2 y MAE por task.

Uso:
    python analyze.py --checkpoint checkpoints/tcn_s01_50ep/best.pt \\
                      --csv data/raw/P01/P01_S01_20260524_181522.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import DEFAULT_FEATURE_SET, EMGSessionDataset, PreprocessConfig
from models import EMGTCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default=None,
                   help="default: <checkpoint_dir>/analysis")
    p.add_argument("--exclude", nargs="*", default=["transition"],
                   help="tasks a excluir del analisis")
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
    output_dim = ckpt.get("output_dim", 1)  # multi-task checkpoints saven 2
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
    state = ckpt["model_state"]
    # Compat: checkpoints antiguos guardaban block1./block2. cuando EMGTCN
    # tenia exactamente 2 bloques residuales. Ahora son blocks.0./blocks.1.
    state = {
        (k.replace("block1.", "blocks.0.").replace("block2.", "blocks.1.") if k.startswith(("block1.", "block2.")) else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_full_sequence(model: EMGTCN, features: np.ndarray, device: torch.device) -> np.ndarray:
    """
    features: [C, T] np.float32
    Returns: [T] np.float32 — predicciones causales para cada timestep.

    Posiciones < window_length-1 tienen contexto incompleto (padded);
    aun asi se devuelven y se filtran luego.
    """
    x = torch.from_numpy(features).unsqueeze(0).to(device)  # [1, C, T]
    h = x
    for block in model.blocks:
        h = block(h)
    if model.has_final_conv:
        h = model.final_drop(model.final_act(model.final_conv(h)))
    h = model.head(h)  # [1, output_dim, T]
    return h[0, 0, :].cpu().numpy()


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    p = pred.astype(np.float64)
    t = target.astype(np.float64)
    err = p - t
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    if p.std() < 1e-12 or t.std() < 1e-12:
        r = 0.0
    else:
        r = float(np.mean((p - p.mean()) * (t - t.mean())) / (p.std() * t.std()))
    ss_res = float(np.sum((t - p) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"n": len(p), "mae": mae, "rmse": rmse, "pearson": r, "r2": r2}


def plot_per_task_bars(per_task: Dict[str, Dict], out_path: Path) -> None:
    tasks = sorted(per_task.keys(), key=lambda k: per_task[k]["r2"], reverse=True)
    r2s = [per_task[t]["r2"] for t in tasks]
    maes = [per_task[t]["mae"] for t in tasks]
    ns = [per_task[t]["n"] for t in tasks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.4 * len(tasks) + 2))
    colors_r2 = ["seagreen" if v > 0 else "tomato" for v in r2s]
    ax1.barh(tasks, r2s, color=colors_r2)
    ax1.axvline(0, color="black", lw=0.6)
    ax1.set_xlabel("R²")
    ax1.set_title("R² por task")
    for i, (v, n) in enumerate(zip(r2s, ns)):
        ax1.text(v, i, f"  {v:+.3f}  (n={n:,})", va="center",
                 fontsize=9, color="dimgray")
    ax1.invert_yaxis()

    ax2.barh(tasks, maes, color="steelblue")
    ax2.set_xlabel("MAE (target normalizado)")
    ax2.set_title("MAE por task")
    for i, v in enumerate(maes):
        ax2.text(v, i, f"  {v:.3f}", va="center", fontsize=9, color="dimgray")
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"Device: {device}")

    ckpt_path = Path(args.checkpoint)
    model, ckpt = load_model(str(ckpt_path), device)
    cfg = ckpt["config"]
    window_length = cfg["window_length"]

    ckpt_features = cfg.get("features") or list(DEFAULT_FEATURE_SET)
    print(f"\nCargando sesion: {args.csv}")
    print(f"Feature set del checkpoint: {ckpt_features}")
    ds = EMGSessionDataset(
        csv_path=args.csv,
        window_length=window_length,
        stride=1,
        config=PreprocessConfig(feature_set=tuple(ckpt_features)),
        verbose=False,
    )
    print(f"  fs           = {ds.fs:.2f} Hz")
    print(f"  n_samples    = {ds.features.shape[1]:,}")
    print(f"  in_channels  = {ds.features.shape[0]}")
    print(f"  window       = {window_length}  (RF del modelo)")

    # ---- Inferencia causal sobre la sesion completa ----
    print(f"\nForward pass causal sobre {ds.features.shape[1]:,} samples...")
    preds_norm = predict_full_sequence(model, ds.features, device)

    # Las primeras (window-1) muestras tienen contexto incompleto -> excluir
    valid_start = window_length - 1
    preds_norm = preds_norm[valid_start:]
    targets_norm = ds.target[valid_start:]
    tasks = ds.tasks[valid_start:]

    # ---- Filtrar tasks excluidos ----
    keep = np.ones(len(tasks), dtype=bool)
    for t in args.exclude:
        keep &= tasks != t
    preds_norm = preds_norm[keep]
    targets_norm = targets_norm[keep]
    tasks = tasks[keep]

    # ---- Metrica agregada ----
    all_m = compute_metrics(preds_norm, targets_norm)
    print(f"\nALL ({all_m['n']:,} samples, excl {args.exclude}):")
    print(f"  MAE     = {all_m['mae']:.4f}")
    print(f"  RMSE    = {all_m['rmse']:.4f}")
    print(f"  Pearson = {all_m['pearson']:+.3f}")
    print(f"  R²      = {all_m['r2']:+.3f}")

    # ---- Por task ----
    per_task: Dict[str, Dict[str, float]] = {}
    for task in sorted(set(tasks.tolist())):
        mask = tasks == task
        if mask.sum() < 2:
            continue
        per_task[task] = compute_metrics(preds_norm[mask], targets_norm[mask])

    print(f"\nPor task (ordenado por R² desc):")
    header = f"{'task':<24} {'n':>8} {'MAE':>8} {'RMSE':>8} {'R':>7} {'R²':>7}"
    print(header)
    print("-" * len(header))
    sorted_tasks = sorted(per_task.items(), key=lambda kv: -kv[1]["r2"])
    for task, m in sorted_tasks:
        print(f"{task:<24} {m['n']:>8,} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{m['pearson']:>+7.3f} {m['r2']:>+7.3f}")

    # ---- Output ----
    out_dir = Path(args.output_dir) if args.output_dir else (ckpt_path.parent / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "per_task_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "n", "mae", "rmse", "pearson", "r2"])
        w.writerow(["ALL", all_m["n"], all_m["mae"], all_m["rmse"],
                    all_m["pearson"], all_m["r2"]])
        for task, m in sorted(per_task.items()):
            w.writerow([task, m["n"], m["mae"], m["rmse"],
                        m["pearson"], m["r2"]])
    print(f"\nCSV guardado en: {csv_path}")

    bar_path = out_dir / "per_task_metrics.png"
    plot_per_task_bars(per_task, bar_path)
    print(f"Bar chart guardado en: {bar_path}")


if __name__ == "__main__":
    main()
