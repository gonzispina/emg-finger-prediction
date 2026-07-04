"""
Training loop para EMGTCN sobre una sesion (por ahora P01_S01).

Split temporal con gap para evitar leakage del receptive field entre train/val:
    [    train    ][gap][   val   ]

Loss: Huber. Optimizer: AdamW. Sin scheduler — agregalo si entrenas mas largo.

Uso:
    python train.py --csv data/raw/P01/P01_S01_20260524_181522.csv
    python train.py --csv ... --epochs 30 --run-name tcn_s01_v2
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from dataset import ALL_FEATURE_TYPES, DEFAULT_FEATURE_SET, EMGSessionDataset, PreprocessConfig
from models import EMGTCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, nargs="+",
                   help="path al CSV de sesion (uno o varios)")
    p.add_argument("--window-length", type=int, default=243)
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--gap-windows", type=int, default=243,
                   help="ventanas descartadas entre train y val para evitar leakage")
    p.add_argument("--split-mode", default="stratified", choices=["stratified", "temporal"],
                   help="stratified: ultimo 20%% de cada bloque al val; temporal: ultimo 20%% del session")
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kernel-size", type=int, default=3,
                   help="tamaño del kernel temporal en todas las convs")
    p.add_argument("--dilations", type=int, nargs="+", default=[1, 3, 9, 27, 81],
                   help="lista de dilataciones (5 valores: 2 bloques residuales + capa final)")
    p.add_argument("--sigma-ms", type=float, default=150.0,
                   help="sigma del gaussiano que suaviza la posicion del joystick antes "
                        "de derivar para sacar la velocidad target. Default 150 ms.")
    p.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURE_SET),
                   choices=list(ALL_FEATURE_TYPES),
                   help="feature types por canal EMG (default: rms_norm d_rms_norm). "
                        "Se emiten 2 channels por feature (flx + ext).")
    p.add_argument("--no-weight-norm", action="store_true",
                   help="desactivar WeightNorm (util en MPS donde puede causar NaN)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--alpha-pos", type=float, default=0.0,
                   help="peso del head de posicion en la loss multi-task. "
                        "0 = solo velocidad (single task). >0 = multi-task con peso alpha.")
    p.add_argument("--smooth-weight", type=float, default=0.0,
                   help="[LEGACY] peso del termino TVD |pred[t+1]-pred[t]| cross-batch. "
                        "0 = sin smoothness. Preferir --horizon + --accel-weight.")
    p.add_argument("--horizon", type=int, default=1,
                   help="cuantos timesteps finales predecir por ventana. "
                        "1 = comportamiento clasico. >=2 habilita --accel-weight.")
    p.add_argument("--accel-weight", type=float, default=0.0,
                   help="peso del termino Sobolev/H1 = MSE(Δpred, Δtarget). "
                        "Requiere --horizon >= 2. Regulariza la derivada de la velocidad.")
    p.add_argument("--var-weight", type=float, default=0.0,
                   help="Variance matching: lambda * (std(pred_vel) - std(y_vel))^2. "
                        "Fuerza al modelo a capturar la energia del target.")
    p.add_argument("--upweight-tasks", nargs="+", default=[],
                   help="Tasks que se upweightan via WeightedRandomSampler. "
                        "Equivalente a multiplicar la loss por --upweight-factor en esas tasks.")
    p.add_argument("--exclude-tasks", nargs="+", default=["transition"],
                   help="Tasks excluidas tanto del train como del val. "
                        "Default: transition.")
    p.add_argument("--upweight-factor", type=float, default=1.0,
                   help="Factor de oversampling para tasks en --upweight-tasks.")
    p.add_argument("--init-from", default=None,
                   help="path a un checkpoint para inicializar pesos (fine-tuning).")
    p.add_argument("--feature-norm", default="none",
                   choices=["none", "median_mad", "mean_std"],
                   help="Z-normalizacion por canal sobre train. "
                        "median_mad: (x-median)/(MAD*1.4826), robusto a outliers. "
                        "mean_std: (x-mean)/std clasico. "
                        "Stats se guardan en el checkpoint para inferencia.")
    p.add_argument("--lr-scheduler", default="none", choices=["none", "plateau"],
                   help="Scheduler de lr. plateau: reduce lr cuando val_loss no mejora.")
    p.add_argument("--lr-factor", type=float, default=0.5,
                   help="Factor de reduccion del lr en plateau (default 0.5 = mitad).")
    p.add_argument("--lr-patience", type=int, default=10,
                   help="Epochs sin mejora antes de reducir lr (default 10).")
    p.add_argument("--lr-min", type=float, default=1e-5,
                   help="lr minimo para el scheduler (default 1e-5).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--run-name", default="tcn_s01")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def temporal_split(n: int, val_fraction: float, gap: int) -> tuple[list[int], list[int]]:
    """
    Indices 0..n-1 (dataset indices). Last val_fraction es val; los `gap`
    ultimos del train se descartan para evitar leakage del receptive field.
    """
    val_start = int(n * (1 - val_fraction))
    train_end = max(0, val_start - gap)
    return list(range(0, train_end)), list(range(val_start, n))


def stratified_split_multi(datasets, val_fraction: float, gap: int) -> tuple[list[int], list[int]]:
    """
    Aplica stratified_split por sesion y ofsetea los indices para usar contra
    un ConcatDataset(datasets). Garantiza que cada bloque de cada sesion
    aporte su 20 % final a val.
    """
    train_idx: list[int] = []
    val_idx: list[int] = []
    offset = 0
    for ds in datasets:
        tr, vl = stratified_split(ds, val_fraction, gap)
        train_idx.extend(i + offset for i in tr)
        val_idx.extend(i + offset for i in vl)
        offset += len(ds)
    return train_idx, val_idx


def lookup_concat(global_idx: int, datasets) -> tuple[int, int]:
    """Devuelve (dataset_idx, local_idx) para un indice global del ConcatDataset."""
    offset = 0
    for d, ds in enumerate(datasets):
        if global_idx < offset + len(ds):
            return d, global_idx - offset
        offset += len(ds)
    raise IndexError(global_idx)


def harmonize_target_scale(datasets, percentile: float = 99.0) -> float:
    """
    Calcula un target_scale compartido entre todas las sesiones (p99 de |v|
    sobre la velocidad cruda concatenada) y re-normaliza todos los datasets.
    """
    all_raw = np.concatenate([np.abs(ds.target_raw) for ds in datasets])
    scale = float(np.percentile(all_raw, percentile))
    if scale <= 0:
        scale = 1.0
    for ds in datasets:
        ds.target = (ds.target_raw / scale).astype(np.float32)
        ds.target_scale = scale
    return scale


def harmonize_position_scale(datasets, percentile: float = 99.0) -> float:
    """Igual que harmonize_target_scale pero para la posicion (joy)."""
    all_raw = np.concatenate([np.abs(ds.joy) for ds in datasets])
    scale = float(np.percentile(all_raw, percentile))
    if scale <= 0:
        scale = 1.0
    for ds in datasets:
        ds.position = (ds.joy / scale).astype(np.float32)
        ds.position_scale = scale
    return scale


def stratified_split(dataset, val_fraction: float, gap: int) -> tuple[list[int], list[int]]:
    """
    Por cada bloque contiguo de la misma task: ultimo val_fraction al val,
    con `gap` ventanas removidas entre train y val dentro del bloque.

    Garantiza que cada task aparezca en train y val.
    """
    n = len(dataset)
    last_positions = dataset.indices              # global CSV idx per window
    win_tasks = dataset.tasks[last_positions]     # task del ultimo sample de cada ventana

    changes = np.where(win_tasks[1:] != win_tasks[:-1])[0] + 1
    boundaries = np.concatenate([[0], changes, [n]])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        block_n = end - start
        block_val_n = int(round(block_n * val_fraction))
        if block_val_n <= 0:
            train_idx.extend(range(start, end))
            continue
        block_val_start = end - block_val_n
        block_train_end = max(start, block_val_start - gap)
        train_idx.extend(range(start, block_train_end))
        val_idx.extend(range(block_val_start, end))
    return train_idx, val_idx


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    preds = preds.astype(np.float64)
    targets = targets.astype(np.float64)
    err = preds - targets
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    p_std = preds.std()
    t_std = targets.std()
    if p_std < 1e-12 or t_std < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.mean((preds - preds.mean()) * (targets - targets.mean())) / (p_std * t_std))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "r2": r2}


def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip,
                epoch: int, total_epochs: int, alpha_pos: float = 0.0,
                smooth_weight: float = 0.0, horizon: int = 1,
                accel_weight: float = 0.0, var_weight: float = 0.0,
                log_every: int = 50):
    """
    Si horizon == 1: pred_vel shape [B], y shape [B, 2]. Modelo actual.
    Si horizon  > 1: pred_vel shape [B, K], y shape [B, K, 2].
        - L_data  = Huber(pred_vel, y_vel)         sobre los K timesteps.
        - L_accel = MSE(Δpred_vel, Δy_vel)         sobre las K-1 diferencias.
    """
    model.train()
    total, n = 0.0, 0
    total_vel, total_pos, total_smooth, total_accel = 0.0, 0.0, 0.0, 0.0
    n_batches = len(loader)
    t_start = time.time()
    use_multitask = alpha_pos > 0.0
    use_smooth = smooth_weight > 0.0 and horizon == 1
    use_accel = accel_weight > 0.0 and horizon > 1
    prev_pred_vel = None                               # state cross-batch para TVD legacy

    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)            # [B, 2] o [B, K, 2]

        if horizon == 1:
            y_vel = y[:, 0]                            # [B]
            y_pos = y[:, 1]                            # [B]
            pred = model(x, horizon=1)                 # [B, output_dim]
            if use_multitask:
                pred_vel = pred[:, 0]
                pred_pos = pred[:, 1]
            else:
                pred_vel = pred.squeeze(-1)
                pred_pos = None
        else:
            y_vel = y[..., 0]                          # [B, K]
            y_pos = y[..., 1]                          # [B, K]
            pred = model(x, horizon=horizon)           # [B, output_dim, K]
            if use_multitask:
                pred_vel = pred[:, 0, :]               # [B, K]
                pred_pos = pred[:, 1, :]               # [B, K]
            else:
                pred_vel = pred[:, 0, :]               # [B, K]
                pred_pos = None

        loss_vel = loss_fn(pred_vel, y_vel)
        if use_multitask:
            loss_pos = loss_fn(pred_pos, y_pos)
            loss = loss_vel + alpha_pos * loss_pos
        else:
            loss_pos = torch.tensor(0.0, device=device)
            loss = loss_vel

        # Sobolev/H1: MSE sobre la derivada (intra-ventana).
        if use_accel:
            d_pred = pred_vel[:, 1:] - pred_vel[:, :-1]     # [B, K-1]
            d_target = y_vel[:, 1:] - y_vel[:, :-1]         # [B, K-1], es gt -> sin grad
            loss_accel = ((d_pred - d_target) ** 2).mean()
            loss = loss + accel_weight * loss_accel
        else:
            loss_accel = torch.tensor(0.0, device=device)

        # Variance matching: penaliza std(pred) lejos de std(target).
        if var_weight > 0:
            loss_var = (pred_vel.flatten().std() - y_vel.flatten().std()) ** 2
            loss = loss + var_weight * loss_var
        else:
            loss_var = torch.tensor(0.0, device=device)

        # TVD legacy cross-batch (solo horizon=1). Asume shuffle=False.
        if use_smooth:
            diffs_intra = (pred_vel[1:] - pred_vel[:-1]).abs()
            if prev_pred_vel is not None:
                cross = (pred_vel[0:1] - prev_pred_vel).abs()
                loss_smooth = torch.cat([cross, diffs_intra]).mean()
            else:
                loss_smooth = diffs_intra.mean() if diffs_intra.numel() > 0 \
                              else torch.tensor(0.0, device=device)
            loss = loss + smooth_weight * loss_smooth
            prev_pred_vel = pred_vel[-1].detach()
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = y.shape[0]
        total += loss.item() * bs
        total_vel += loss_vel.item() * bs
        total_pos += loss_pos.item() * bs
        total_smooth += loss_smooth.item() * bs
        total_accel += loss_accel.item() * bs
        n += bs
        if i % log_every == 0 or i == n_batches:
            elapsed = time.time() - t_start
            sps = n / max(elapsed, 1e-6)
            avg_loss = total / max(n, 1)
            avg_vel = total_vel / max(n, 1)
            extra = ""
            if use_multitask:
                extra += f"  vel {avg_vel:.4f}  pos {total_pos/max(n,1):.4f}"
            if use_accel:
                extra += f"  accel {total_accel/max(n,1):.4f}"
            if use_smooth:
                extra += f"  smooth {total_smooth/max(n,1):.4f}"
            print(
                f"\r  ep {epoch}/{total_epochs}  train  "
                f"batch {i:>4}/{n_batches}  "
                f"loss {avg_loss:.4f}{extra}  "
                f"{sps:>6.0f} samples/s  "
                f"{elapsed:>5.1f}s",
                end="", flush=True,
            )
    print()
    return (total / max(n, 1), total_vel / max(n, 1),
            total_pos / max(n, 1), total_smooth / max(n, 1),
            total_accel / max(n, 1))


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, device,
               epoch: int, total_epochs: int, alpha_pos: float = 0.0,
               horizon: int = 1, log_every: int = 50):
    """
    Evaluation. Reporta metricas SIEMPRE sobre el ultimo timestep predicho
    (lo que se usa en deployment), incluso si horizon > 1 — esto mantiene
    las metricas comparables entre runs con horizon distinto.
    """
    model.eval()
    total, n = 0.0, 0
    total_vel, total_pos = 0.0, 0.0
    preds_vel_list, targets_vel_list = [], []
    preds_pos_list, targets_pos_list = [], []
    n_batches = len(loader)
    t_start = time.time()
    use_multitask = alpha_pos > 0.0
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)            # [B, 2] o [B, K, 2]

        if horizon == 1:
            y_vel = y[:, 0]
            y_pos = y[:, 1]
            pred = model(x, horizon=1)                 # [B, output_dim]
            if use_multitask:
                pred_vel = pred[:, 0]
                pred_pos = pred[:, 1]
            else:
                pred_vel = pred.squeeze(-1)
                pred_pos = None
        else:
            # Solo el ultimo timestep del horizonte se mide
            y_vel = y[:, -1, 0]                        # [B]
            y_pos = y[:, -1, 1]                        # [B]
            pred_full = model(x, horizon=horizon)      # [B, output_dim, K]
            pred = pred_full[..., -1]                  # [B, output_dim]
            if use_multitask:
                pred_vel = pred[:, 0]
                pred_pos = pred[:, 1]
            else:
                pred_vel = pred.squeeze(-1)
                pred_pos = None

        loss_vel = loss_fn(pred_vel, y_vel)
        if use_multitask:
            loss_pos = loss_fn(pred_pos, y_pos)
            loss = loss_vel + alpha_pos * loss_pos
            preds_pos_list.append(pred_pos.detach().cpu().numpy())
            targets_pos_list.append(y_pos.detach().cpu().numpy())
        else:
            loss_pos = torch.tensor(0.0, device=device)
            loss = loss_vel

        bs = y.shape[0]
        total += loss.item() * bs
        total_vel += loss_vel.item() * bs
        total_pos += loss_pos.item() * bs
        n += bs
        preds_vel_list.append(pred_vel.detach().cpu().numpy())
        targets_vel_list.append(y_vel.detach().cpu().numpy())
        if i % log_every == 0 or i == n_batches:
            elapsed = time.time() - t_start
            sps = n / max(elapsed, 1e-6)
            avg_loss = total / max(n, 1)
            print(
                f"\r  ep {epoch}/{total_epochs}  val    "
                f"batch {i:>4}/{n_batches}  "
                f"loss {avg_loss:.4f}  "
                f"{sps:>6.0f} samples/s  "
                f"{elapsed:>5.1f}s",
                end="", flush=True,
            )
    print()
    preds_vel = np.concatenate(preds_vel_list) if preds_vel_list else np.array([])
    targets_vel = np.concatenate(targets_vel_list) if targets_vel_list else np.array([])
    metrics = compute_metrics(preds_vel, targets_vel)
    metrics["loss"] = total / max(n, 1)
    metrics["loss_vel"] = total_vel / max(n, 1)
    metrics["loss_pos"] = total_pos / max(n, 1)
    if use_multitask:
        preds_pos = np.concatenate(preds_pos_list)
        targets_pos = np.concatenate(targets_pos_list)
        pos_metrics = compute_metrics(preds_pos, targets_pos)
        metrics["pos_mae"] = pos_metrics["mae"]
        metrics["pos_r2"] = pos_metrics["r2"]
        metrics["pos_pearson"] = pos_metrics["pearson"]
    return metrics, preds_vel, targets_vel


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"Device: {device}")

    # ---- Validacion de combinaciones ----
    if args.accel_weight > 0.0 and args.horizon < 2:
        raise ValueError("--accel-weight requiere --horizon >= 2")
    if args.smooth_weight > 0.0 and args.horizon > 1:
        raise ValueError(
            "--smooth-weight (legacy TVD cross-batch) no es compatible con --horizon > 1. "
            "Usar --accel-weight para regularizar la derivada."
        )

    # ---- Datasets (1 o varios CSVs) ----
    print(f"\nCargando {len(args.csv)} sesion(es):")
    print(f"Feature set: {args.features}  -> {len(args.features) * 2} canales totales")
    if args.horizon > 1:
        print(f"Horizon: {args.horizon} timesteps (output K, input extendido a "
              f"{args.window_length + args.horizon - 1})")
    config = PreprocessConfig(feature_set=tuple(args.features), sigma_ms=args.sigma_ms)
    datasets = []
    for csv_path in args.csv:
        print(f"  - {csv_path}")
        ds = EMGSessionDataset(
            csv_path=csv_path,
            window_length=args.window_length,
            stride=1,
            exclude_tasks=tuple(args.exclude_tasks),
            config=config,
            horizon=args.horizon,
            verbose=True,
        )
        datasets.append(ds)
    in_channels = datasets[0].features.shape[0]
    for ds in datasets[1:]:
        assert ds.features.shape[0] == in_channels, (
            "Inconsistencia de in_channels entre sesiones"
        )

    # Target scale compartido entre sesiones (velocity y position)
    shared_scale = harmonize_target_scale(datasets, percentile=99.0)
    shared_pos_scale = harmonize_position_scale(datasets, percentile=99.0)
    print(f"\nTarget scale compartido (p99 sobre todas las sesiones):")
    print(f"  velocity: {shared_scale:.4f}")
    print(f"  position: {shared_pos_scale:.4f}")
    use_multitask = args.alpha_pos > 0.0
    output_dim = 2 if use_multitask else 1
    print(f"  task    : {'multi-task (vel + alpha*pos)' if use_multitask else 'single task (vel)'}  "
          f"alpha_pos = {args.alpha_pos}")

    # Split por sesion
    if args.split_mode == "stratified":
        train_idx, val_idx = stratified_split_multi(
            datasets, val_fraction=args.val_fraction, gap=args.gap_windows,
        )
    else:
        # Temporal: aplicado al ConcatDataset entero
        n_total = sum(len(d) for d in datasets)
        train_idx, val_idx = temporal_split(
            n=n_total, val_fraction=args.val_fraction, gap=args.gap_windows,
        )
    print(f"Split mode: {args.split_mode}")
    # Aplicamos stride de muestreo sobre los indices
    train_idx = train_idx[:: args.train_stride]
    val_idx = val_idx[:: args.val_stride]
    print(f"Train windows: {len(train_idx):,}  (stride {args.train_stride})")
    print(f"Val   windows: {len(val_idx):,}  (stride {args.val_stride})")

    # ----- Z-normalizacion opcional por canal -----
    feat_center = feat_scale = None
    if args.feature_norm in ("median_mad", "mean_std"):
        feats_list = []
        for ds in datasets:
            mask = np.ones(ds.features.shape[1], dtype=bool)
            for t in args.exclude_tasks:
                mask &= ds.tasks != t
            feats_list.append(ds.features[:, mask])
        all_train_feats = np.concatenate(feats_list, axis=1)
        if args.feature_norm == "median_mad":
            feat_center = np.median(all_train_feats, axis=1).astype(np.float32)
            mad = np.median(np.abs(all_train_feats - feat_center[:, None]), axis=1).astype(np.float32)
            feat_scale = (mad * 1.4826 + 1e-6).astype(np.float32)
            label_c, label_s = "median", "MAD*1.4826"
        else:
            feat_center = all_train_feats.mean(axis=1).astype(np.float32)
            feat_scale = (all_train_feats.std(axis=1) + 1e-6).astype(np.float32)
            label_c, label_s = "mean", "std"
        print(f"\nFeature norm: {label_c} + {label_s}:")
        for i, name in enumerate(datasets[0].feature_names):
            print(f"  [{i:>2}] {name:<28} {label_c}={feat_center[i]:+.4f}  {label_s}={feat_scale[i]:.4f}")
        for ds in datasets:
            ds.apply_feature_norm(feat_center, feat_scale)

    # Tareas correspondientes a las ventanas de val (lookup con offsets)
    val_tasks = np.array([
        datasets[d_idx].task_at(local_idx)
        for d_idx, local_idx in (lookup_concat(i, datasets) for i in val_idx)
    ])

    concat = ConcatDataset(datasets)
    use_smooth = args.smooth_weight > 0.0
    shuffle_train = not use_smooth
    if use_smooth:
        print(f"Smoothness regularization activa (weight={args.smooth_weight}). "
              f"shuffle desactivado para preservar orden temporal en batches.")
    # Weighted sampling: upweightear tasks especificas usando WeightedRandomSampler.
    # Equivalente a multiplicar la loss por upweight_factor en samples de esas tasks.
    train_sampler = None
    if args.upweight_tasks and args.upweight_factor != 1.0:
        train_tasks = np.array([
            datasets[d_idx].task_at(local_idx)
            for d_idx, local_idx in (lookup_concat(i, datasets) for i in train_idx)
        ])
        sample_weights = np.where(
            np.isin(train_tasks, args.upweight_tasks),
            args.upweight_factor, 1.0
        ).astype(np.float64)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, num_samples=len(train_idx), replacement=True,
        )
        upweighted = int((sample_weights > 1.0).sum())
        print(f"Upweight: {args.upweight_factor}x sobre tasks {args.upweight_tasks} "
              f"({upweighted}/{len(train_idx)} = {upweighted/len(train_idx)*100:.1f}% samples)")

    train_loader = DataLoader(
        Subset(concat, train_idx),
        batch_size=args.batch_size,
        shuffle=shuffle_train if train_sampler is None else False,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(concat, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---- Baselines de control (sobre la velocidad) ----
    def _target_at(global_idx: int) -> float:
        d_idx, local_idx = lookup_concat(global_idx, datasets)
        ds = datasets[d_idx]
        return float(ds.target[ds.indices[local_idx]])

    train_targets = np.array([_target_at(i) for i in train_idx])
    val_targets = np.array([_target_at(i) for i in val_idx])
    bl_mean = compute_metrics(np.full_like(val_targets, train_targets.mean()), val_targets)
    print(f"\nBaseline (predict train mean): "
          f"MAE={bl_mean['mae']:.4f}  RMSE={bl_mean['rmse']:.4f}  R={bl_mean['pearson']:+.3f}")

    # ---- Modelo ----
    model = EMGTCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        output_dim=output_dim,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dilations=args.dilations,
        use_weight_norm=not args.no_weight_norm,
    ).to(device)

    # Fine-tuning: inicializar desde checkpoint si --init-from
    if args.init_from:
        init_ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        init_epoch = init_ckpt.get("epoch", "?")
        init_alpha = init_ckpt.get("alpha_pos", "?")
        print(f"\nInicializando pesos desde {args.init_from} "
              f"(epoch {init_epoch}, alpha_pos {init_alpha})")
        try:
            state = init_ckpt["model_state"]
            # Compat con checkpoints antiguos: block1./block2. -> blocks.0./blocks.1.
            state = {
                (k.replace("block1.", "blocks.0.").replace("block2.", "blocks.1.") if k.startswith(("block1.", "block2.")) else k): v
                for k, v in state.items()
            }
            model.load_state_dict(state)
        except RuntimeError as e:
            raise RuntimeError(
                f"No se puede cargar state_dict del checkpoint. Posible mismatch "
                f"de arquitectura (in_channels, output_dim, hidden, features). "
                f"Error original: {e}"
            )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModelo: EMGTCN  in_channels={in_channels}  hidden={args.hidden_channels}  "
          f"RF={model.receptive_field}  params={n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.lr_min,
        )
        print(f"Scheduler: ReduceLROnPlateau  factor={args.lr_factor}  "
              f"patience={args.lr_patience}  min_lr={args.lr_min}")
    loss_fn = nn.HuberLoss(delta=args.huber_delta)

    # ---- Output dir ----
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    np.save(out_dir / "val_tasks.npy", val_tasks)

    # ---- Training ----
    print(f"\nEntrenando {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}, "
          f"huber_delta={args.huber_delta}")
    print(f"{'ep':>3} {'train':>8} {'val':>8} {'MAE':>8} {'RMSE':>8} {'R':>7} {'R2':>7} {'time':>6}")
    log = []
    best_val = float("inf")
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_vel, train_pos, train_smooth, train_accel = train_epoch(
            model, train_loader, optimizer, loss_fn, device, args.grad_clip,
            epoch=ep, total_epochs=args.epochs,
            alpha_pos=args.alpha_pos, smooth_weight=args.smooth_weight,
            horizon=args.horizon, accel_weight=args.accel_weight,
            var_weight=args.var_weight,
        )
        val_metrics, val_preds, val_targets_arr = eval_epoch(
            model, val_loader, loss_fn, device,
            epoch=ep, total_epochs=args.epochs, alpha_pos=args.alpha_pos,
            horizon=args.horizon,
        )
        elapsed = time.time() - t0

        is_best = val_metrics["loss"] < best_val
        if is_best:
            best_val = val_metrics["loss"]
            best_epoch = ep
            torch.save({
                "model_state": model.state_dict(),
                "config": vars(args),
                "epoch": ep,
                "val_metrics": val_metrics,
                "target_scale": shared_scale,
                "position_scale": shared_pos_scale,
                "alpha_pos": args.alpha_pos,
                "smooth_weight": args.smooth_weight,
                "horizon": args.horizon,
                "accel_weight": args.accel_weight,
                "output_dim": output_dim,
                "fs": float(np.mean([d.fs for d in datasets])),
                "fs_per_session": [d.fs for d in datasets],
                "csv_paths": list(args.csv),
                "in_channels": in_channels,
                "feature_names": list(datasets[0].feature_names),
                "feature_norm": args.feature_norm,
                "feature_center": feat_center.tolist() if feat_center is not None else None,
                "feature_scale": feat_scale.tolist() if feat_scale is not None else None,
            }, out_dir / "best.pt")
            np.savez(
                out_dir / "best_val_preds.npz",
                preds=val_preds, targets=val_targets_arr, tasks=val_tasks,
            )

        if scheduler is not None:
            scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        flag = " *" if is_best else ""
        lr_str = f"  lr={current_lr:.2e}" if scheduler is not None else ""
        print(
            f"{ep:>3} {train_loss:>8.4f} {val_metrics['loss']:>8.4f} "
            f"{val_metrics['mae']:>8.4f} {val_metrics['rmse']:>8.4f} "
            f"{val_metrics['pearson']:>+7.3f} {val_metrics['r2']:>+7.3f} "
            f"{elapsed:>5.1f}s{flag}{lr_str}"
        )

        log.append({
            "epoch": ep,
            "train_loss": train_loss,
            "train_vel": train_vel,
            "train_pos": train_pos,
            "train_smooth": train_smooth,
            "train_accel": train_accel,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_pearson": val_metrics["pearson"],
            "val_r2": val_metrics["r2"],
            "elapsed_s": elapsed,
            "is_best": is_best,
            "lr": current_lr,
        })
        with open(out_dir / "log.json", "w") as f:
            json.dump(log, f, indent=2)

    print(f"\nBest val Huber: {best_val:.4f} en epoch {best_epoch}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")
    print(f"Val preds : {out_dir / 'best_val_preds.npz'}")
    print(f"Log       : {out_dir / 'log.json'}")


if __name__ == "__main__":
    main()
