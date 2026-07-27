"""
Compara 1-2 checkpoints sobre uno o mas CSVs y muestra R² por task.

Uso:
    # Un modelo, una sesion
    python eval_compare.py --csv data/raw/P01/P01_S12_*.csv \
        --models checkpoints/champ/best.pt

    # Dos modelos con etiquetas, multiples sesiones
    python eval_compare.py \
        --csv data/raw/P01/P01_S12_*.csv data/raw/TEST/TEST_*.csv \
        --models champ:checkpoints/champ/best.pt noise:checkpoints/noise/best.pt

    # Excluir tasks adicionales del overall
    python eval_compare.py --csv ... --models ... \
        --exclude-overall rest_baseline noise_other_fingers noise_wrist_grip

Formato de --models: "label:path" o solo "path" (el label se infiere del nombre del directorio).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from dataset.emg_dataset import EMGSessionDataset, PreprocessConfig
from models import EMGTCN


# Tasks que por definicion tienen varianza de velocidad ~0 y producen R2 catastrofico.
_DEFAULT_EXCLUDE_OVERALL = frozenset({
    "transition", "rest_baseline", "noise_other_fingers", "noise_wrist_grip",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_model_arg(s: str) -> Tuple[str, str]:
    """Acepta "label:path" o solo "path" (label = nombre de carpeta padre)."""
    if ":" in s:
        label, path = s.split(":", 1)
        return label.strip(), path.strip()
    p = Path(s)
    label = p.parent.name if p.name == "best.pt" else p.stem
    return label, str(p)


def _load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
    # Campos que en checkpoints nuevos estan en cfg y en viejos al top level del ckpt
    cfg["best_epoch"]  = ckpt.get("epoch", None)
    cfg["best_val_r2"] = (ckpt.get("val_metrics") or {}).get("r2", float("nan"))
    cfg.setdefault("in_channels",          ckpt.get("in_channels", 4))
    cfg.setdefault("output_dim",           ckpt.get("output_dim", 1))
    cfg.setdefault("feature_center",       ckpt.get("feature_center"))
    cfg.setdefault("feature_scale",        ckpt.get("feature_scale"))
    cfg.setdefault("shared_target_scale",  ckpt.get("target_scale"))
    # Checkpoints viejos no tienen preprocess_config: sintetizarlo desde cfg
    if "preprocess_config" not in cfg:
        cfg["preprocess_config"] = {
            "feature_set":            tuple(cfg.get("features", ["rms", "d_rms"])),
            "sigma_ms":               float(cfg.get("sigma_ms", 50.0)),
            "adaptive_sigma_slow_ms": None,
            "bp_low":                 float(cfg.get("bp_low", 70.0)),
            "feature_clip":           cfg.get("feature_clip", 20.0),
            "joy_center_mode":        cfg.get("joy_center_mode", "fixed"),
            "emg_attenuation":        1.0,
            "emg_attenuation_flx":    None,
            "emg_attenuation_ext":    None,
        }
    model = EMGTCN(
        in_channels     = cfg["in_channels"],
        hidden_channels = cfg["hidden_channels"],
        output_dim      = cfg["output_dim"],
        kernel_size     = cfg["kernel_size"],
        dropout         = cfg["dropout"],
        dilations       = cfg["dilations"],
        use_weight_norm = not cfg.get("no_weight_norm", False),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])
    return model, cfg


def _build_config(cfg: dict) -> PreprocessConfig:
    pcfg = cfg["preprocess_config"]
    return PreprocessConfig(
        feature_set            = tuple(pcfg["feature_set"]),
        sigma_ms               = pcfg["sigma_ms"],
        adaptive_sigma_slow_ms = pcfg.get("adaptive_sigma_slow_ms"),
        bp_low                 = pcfg["bp_low"],
        feature_clip           = pcfg.get("feature_clip"),
        joy_center_mode        = pcfg.get("joy_center_mode", "fixed"),
    )


@torch.no_grad()
def _infer_full(model: EMGTCN, features: np.ndarray, device: torch.device) -> np.ndarray:
    """Inferencia causal sobre toda la secuencia. Devuelve [output_dim, T]."""
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    h = x
    for block in model.blocks:
        h = block(h)
    if model.has_final_conv:
        h = model.final_drop(model.final_act(model.final_conv(h)))
    h = model.head(h)
    return h[0].cpu().numpy()


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - np.sum((target - pred) ** 2) / ss_tot)


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    sp, st = pred.std(), target.std()
    if sp < 1e-12 or st < 1e-12:
        return float("nan")
    return float(np.mean((pred - pred.mean()) * (target - target.mean())) / (sp * st))


# ---------------------------------------------------------------------------
# Eval de un modelo sobre lista de CSVs
# ---------------------------------------------------------------------------

def eval_model(
    model: EMGTCN,
    cfg: dict,
    csv_paths: List[str],
    exclude_overall: frozenset,
    warmup_s: float,
    device: torch.device,
) -> Dict[str, dict]:
    """
    Devuelve dict task -> {"r2": float, "pearson": float, "n": int}.
    Clave especial "__overall__" agrega todas las tasks no excluidas.
    """
    preprocess_cfg = _build_config(cfg)
    feat_center = np.array(cfg["feature_center"]) if cfg.get("feature_center") else None
    feat_scale  = np.array(cfg["feature_scale"])  if cfg.get("feature_scale")  else None
    target_scale = float(cfg["shared_target_scale"])

    buckets: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

    for csv_path in csv_paths:
        ds = EMGSessionDataset(
            csv_path,
            window_length = cfg["window_length"],
            stride        = 1,
            exclude_tasks = ("transition",),
            config        = preprocess_cfg,
            verbose       = False,
        )
        if feat_center is not None and feat_scale is not None:
            ds.apply_feature_norm(feat_center, feat_scale)
        ds.target = (ds.target_raw / target_scale).astype(np.float32)

        pred_all = _infer_full(model, ds.features, device)
        pred_vel = pred_all[0]

        warmup = int(round(ds.fs * warmup_s))
        tasks  = ds.tasks
        target = ds.target

        for task in np.unique(tasks):
            if task == "transition":
                continue
            mask = (tasks == task).copy()
            mask[:warmup] = False
            if mask.sum() < 10:
                continue
            if task not in buckets:
                buckets[task] = ([], [])
            buckets[task][0].append(pred_vel[mask])
            buckets[task][1].append(target[mask])

    results: Dict[str, dict] = {}
    all_preds, all_targets = [], []

    for task, (preds, targets) in sorted(buckets.items()):
        p = np.concatenate(preds)
        t = np.concatenate(targets)
        results[task] = {"r2": _r2(p, t), "pearson": _pearson(p, t), "n": len(p)}
        if task not in exclude_overall:
            all_preds.append(p)
            all_targets.append(t)

    if all_preds:
        p_all = np.concatenate(all_preds)
        t_all = np.concatenate(all_targets)
        results["__overall__"] = {
            "r2": _r2(p_all, t_all),
            "pearson": _pearson(p_all, t_all),
            "n": len(p_all),
        }

    return results


# ---------------------------------------------------------------------------
# Impresion de tabla
# ---------------------------------------------------------------------------

def _fmt(v: float, width: int = 7) -> str:
    if v != v:  # nan
        return f"{'n/a':>{width}}"
    if abs(v) > 100:
        return f"{'---':>{width}}"
    return f"{v:>+{width}.3f}"


def print_table(
    model_results: List[Tuple[str, Dict[str, dict]]],
    exclude_overall: frozenset,
    metric: str = "r2",
) -> None:
    names  = [name for name, _ in model_results]
    results = [res for _, res in model_results]

    # Union de tasks (excluyendo claves internas)
    tasks_all = sorted(
        set(t for res in results for t in res if not t.startswith("__"))
    )

    col_w = max(14, *(len(n) for n in names))
    task_w = max(30, *(len(t) for t in tasks_all), len("OVERALL (movement)"))

    header = f"  {'task':<{task_w}}" + "".join(f"  {n:>{col_w}}" for n in names)
    if len(names) == 2:
        header += f"  {'delta':>{col_w}}"
    print(header)
    print("-" * len(header))

    def row_line(task: str, display: str, excl_tag: str) -> None:
        vals = [res.get(task, {}).get(metric, float("nan")) for res in results]
        cells = "".join(f"  {_fmt(v, col_w)}" for v in vals)
        delta_cell = ""
        if len(vals) == 2:
            d = vals[1] - vals[0] if not (vals[0] != vals[0] or vals[1] != vals[1]) else float("nan")
            delta_cell = f"  {_fmt(d, col_w)}"
        print(f"  {display:<{task_w}}{cells}{delta_cell}{excl_tag}")

    for task in tasks_all:
        excl_tag = "  (excl.)" if task in exclude_overall else ""
        row_line(task, task, excl_tag)

    print("-" * len(header))
    if "__overall__" in results[0]:
        row_line("__overall__", "OVERALL (movement)", "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compara 1-2 checkpoints sobre CSVs de prueba.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv", nargs="+", required=True,
                   help="CSVs de evaluacion (glob expandido por shell).")
    p.add_argument("--models", nargs="+", required=True,
                   help="Checkpoints: 'label:path' o 'path' (max 2).")
    p.add_argument("--metric", choices=["r2", "pearson"], default="r2",
                   help="Metrica a mostrar (default: r2).")
    p.add_argument("--exclude-overall", nargs="+",
                   default=list(_DEFAULT_EXCLUDE_OVERALL),
                   help="Tasks a excluir del OVERALL (default: rest + noise).")
    p.add_argument("--warmup-s", type=float, default=1.0,
                   help="Segundos de warm-up a ignorar al inicio de cada sesion.")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.models) > 2:
        sys.exit("ERROR: maximo 2 modelos para comparar.")

    device = torch.device(args.device)
    exclude_overall = frozenset(args.exclude_overall)
    csv_paths = [str(p) for p in args.csv]

    print(f"\nCSVs      : {', '.join(Path(c).name for c in csv_paths)}")
    print(f"Metrica   : {args.metric}")
    print(f"Excl. overall: {', '.join(sorted(exclude_overall))}\n")

    model_results = []
    for model_arg in args.models:
        label, ckpt_path = _parse_model_arg(model_arg)
        print(f"Cargando [{label}] desde {ckpt_path} ...", end=" ", flush=True)
        model, cfg = _load_model(ckpt_path, device)
        print(f"RF={model.receptive_field}  ep={cfg.get('best_epoch','?')}  "
              f"val_r2={cfg.get('best_val_r2', float('nan')):.4f}")
        res = eval_model(model, cfg, csv_paths, exclude_overall, args.warmup_s, device)
        model_results.append((label, res))

    print()
    print_table(model_results, exclude_overall, metric=args.metric)
    print()


if __name__ == "__main__":
    main()
