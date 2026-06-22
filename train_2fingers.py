"""
Training de la TCN para 2 dedos (indice + medio).

Modelo:
    EMGTCN(in_channels=16, output_dim=2, dilations=[1,3,9,27,81,243])
    RF = 729 samples (~1142 ms @ 638 Hz).

Features (16 = 8 por dedo, mismo set que el run reciente del 1-dedo):
    rms_norm, d_rms_norm, rms_slow_norm, d_rms_slow_norm
    (x flexor/extensor, x indice/medio)

Targets (2):
    [vel_ind_norm, vel_med_norm] en el ultimo timestep de la ventana.

Split: random 80/20 sobre las ventanas, con gap >= window_length para evitar
overlap fisico entre train y val. NO es session-level split (eso seria mas
honesto, pero el user pidio random como baseline).

Uso:
    python train_2fingers.py --csv data/raw/P01/P01_S04_*.csv data/raw/P01/P01_S05_*.csv
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

from dataset.emg_dataset_2fingers import (
    EMGSession2FingersDataset, PreprocessConfig2F, DEFAULT_FEATURE_SET_2F,
)
from dataset.preprocessing import ALL_FEATURE_TYPES
from models import EMGTCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, nargs="+",
                   help="path al/los CSV de sesion (formato binary_v2)")
    p.add_argument("--window-length", type=int, default=243,
                   help="RF en samples. Default 243 -> dilations [1,3,9,27,81].")
    p.add_argument("--train-stride", type=int, default=4)
    p.add_argument("--val-stride", type=int, default=4)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--gap-windows", type=int, default=None,
                   help="ventanas descartadas entre train y val para evitar leakage. "
                        "Default = window_length.")
    p.add_argument("--split-mode", default="stratified",
                   choices=["stratified", "temporal", "random_chunked"],
                   help="stratified (recomendado): ultimo val_fraction de cada bloque "
                        "task contiguo. temporal: ultimo val_fraction de la sesion. "
                        "random_chunked (legacy): chunks contiguos asignados random.")
    p.add_argument("--hidden-channels", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--dilations", type=int, nargs="+",
                   default=[1, 3, 9, 27, 81])
    p.add_argument("--sigma-ms", type=float, default=50.0,
                   help="sigma gaussiano para el target (champion=50). Default 50.")
    p.add_argument("--no-weight-norm", action="store_true")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", default="huber", choices=["huber", "l1", "mse"],
                   help="loss de datos. l1 empuja a la mediana (mejor calibracion "
                        "de magnitud), huber/mse a la media.")
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=1,
                   help="K timesteps finales por ventana. >=2 habilita Sobolev.")
    p.add_argument("--accel-weight", type=float, default=0.0,
                   help="Sobolev H1: lambda * MSE(d_pred, d_target). "
                        "Requiere horizon>=2. Penaliza saltos no presentes en el target.")
    p.add_argument("--var-weight", type=float, default=0.0,
                   help="Variance matching: lambda * (std(pred) - std(target))^2 "
                        "promediado por dedo. Fuerza al modelo a capturar la "
                        "energia del sistema, no solo la forma.")
    p.add_argument("--exclude-tasks", nargs="+", default=["transition"],
                   help="Tasks excluidas de train y val. Default: transition.")
    p.add_argument("--target-finger", default="both",
                   choices=["both", "ind", "med"],
                   help="Predecir ambos dedos (both) o monotask sobre uno.")
    p.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURE_SET_2F),
                   choices=list(ALL_FEATURE_TYPES),
                   help="Tipos de feature EMG. Cada uno emite 2 canales (flx, ext) "
                        "por dedo. Default: rms_norm d_rms_norm rms_slow_norm d_rms_slow_norm.")
    p.add_argument("--feature-fingers", default="both",
                   choices=["both", "ind", "med"],
                   help="Dedos cuyas features se incluyen como input. "
                        "Default both (16 canales con default --features).")
    p.add_argument("--feature-norm", default="none",
                   choices=["none", "median_mad", "mean_std"],
                   help="Z-normalizacion por canal sobre train. "
                        "median_mad: (x-median)/(MAD*1.4826), robusto a outliers. "
                        "mean_std: (x-mean)/std clasico. "
                        "Stats se guardan en el checkpoint para inferencia.")
    p.add_argument("--eval-every", type=int, default=1,
                   help="Correr eval cada N epochs (1 = cada epoch). Util cuando "
                        "eval-on-CPU es lento. La ultima epoch siempre se evalua.")
    p.add_argument("--eval-device", default="same", choices=["same", "cpu"],
                   help="Dispositivo para eval. 'cpu' mueve el modelo a CPU antes "
                        "de cada eval y lo devuelve al device de train despues. "
                        "Workaround para el bug MPS+eval mode.")
    p.add_argument("--clip-value", type=float, default=10.0,
                   help="Clip post-normalizacion a [-clip_value, +clip_value]. "
                        "0 = desactivado. El valor optimo depende del tipo de "
                        "normalizacion (median_mad ~10, mean_std ~3-5).")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--run-name", default="tcn_2f_s04s05")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
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
    """Indices 0..n-1. Last val_fraction es val; los `gap` ultimos del train
    se descartan para evitar leakage del RF."""
    val_start = int(n * (1 - val_fraction))
    train_end = max(0, val_start - gap)
    return list(range(0, train_end)), list(range(val_start, n))


def lookup_concat(global_idx: int, datasets) -> tuple[int, int]:
    """Devuelve (dataset_idx, local_idx) para un indice global del ConcatDataset."""
    offset = 0
    for d, ds in enumerate(datasets):
        if global_idx < offset + len(ds):
            return d, global_idx - offset
        offset += len(ds)
    raise IndexError(global_idx)


def stratified_split(dataset, val_fraction: float, gap: int) -> tuple[list[int], list[int]]:
    """Por cada bloque contiguo de la misma task: ultimo val_fraction al val,
    con `gap` ventanas removidas entre train y val dentro del bloque.
    Garantiza que cada task aparezca en train y val."""
    n = len(dataset)
    last_positions = dataset.indices
    win_tasks = dataset.tasks[last_positions]
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


def stratified_split_multi(datasets, val_fraction: float, gap: int) -> tuple[list[int], list[int]]:
    """Aplica stratified_split por sesion y ofsetea los indices contra ConcatDataset."""
    train_idx: list[int] = []
    val_idx: list[int] = []
    offset = 0
    for ds in datasets:
        tr, vl = stratified_split(ds, val_fraction, gap)
        train_idx.extend(i + offset for i in tr)
        val_idx.extend(i + offset for i in vl)
        offset += len(ds)
    return train_idx, val_idx


def random_split_with_gap(dataset, val_fraction: float, gap_samples: int,
                          chunk_samples: int = None,
                          seed: int = 42) -> tuple[list[int], list[int]]:
    """
    Split random por CHUNKS contiguos. Los chunks son bloques de samples
    consecutivos en el CSV; cada chunk se asigna random a train o val con
    probabilidad val_fraction. Como los chunks son contiguos, la zona de
    exclusion entre train y val solo afecta las fronteras (perdemos ~gap
    samples por chunk-boundary), no toda la cobertura.

    chunk_samples controla el tamano de cada bloque (default = 5*gap).
    Mas grande -> menos pero mas largos chunks; mas chico -> mas chunks
    pero relativamente mas "waste" por las fronteras.
    """
    if chunk_samples is None:
        chunk_samples = gap_samples * 5

    last_positions = dataset.indices              # [n_windows] last_idx en samples
    if len(last_positions) == 0:
        return [], []

    n_samples_total = int(last_positions[-1] + 1)
    n_chunks = max(1, n_samples_total // chunk_samples)
    chunk_boundaries = np.linspace(0, n_samples_total, n_chunks + 1, dtype=np.int64)

    rng = np.random.default_rng(seed)
    chunk_is_val = rng.random(n_chunks) < val_fraction

    # Para cada window, decidir train/val segun el chunk del last_idx.
    chunk_of_window = np.searchsorted(chunk_boundaries[1:], last_positions, side="right")
    chunk_of_window = np.clip(chunk_of_window, 0, n_chunks - 1)
    is_val = chunk_is_val[chunk_of_window]

    # Aplicar gap en las fronteras: descartar windows del chunk train que
    # esten a <= gap del primer/ultimo sample de un chunk val adyacente.
    val_sample_set = []
    for c in range(n_chunks):
        if chunk_is_val[c]:
            val_sample_set.append((chunk_boundaries[c], chunk_boundaries[c + 1]))

    if val_sample_set:
        val_intervals = np.array(val_sample_set, dtype=np.int64)  # [k, 2]
        # window descartada si esta a <= gap del [start, end] de algun val chunk.
        # Esto es: existe v tal que (last_idx >= v_start - gap) AND (last_idx < v_end + gap)
        # Optimizamos con searchsorted sobre los bordes.
        keep_train = np.ones(len(last_positions), dtype=bool)
        for v_start, v_end in val_intervals:
            in_gap = (last_positions >= v_start - gap_samples) & \
                     (last_positions < v_end + gap_samples)
            keep_train &= ~(in_gap & ~is_val)
        train_mask = (~is_val) & keep_train
    else:
        train_mask = ~is_val

    train_idx = np.where(train_mask)[0].tolist()
    val_idx = np.where(is_val)[0].tolist()
    return train_idx, val_idx


def random_split_with_gap_multi(datasets, val_fraction: float, gap_samples: int,
                                seed: int = 42) -> tuple[list[int], list[int]]:
    """Aplica random_split_with_gap por sesion y ofsetea para ConcatDataset."""
    train_idx, val_idx = [], []
    offset = 0
    for i, ds in enumerate(datasets):
        tr, vl = random_split_with_gap(ds, val_fraction, gap_samples, seed=seed + i)
        train_idx.extend(j + offset for j in tr)
        val_idx.extend(j + offset for j in vl)
        offset += len(ds)
    return train_idx, val_idx


def compute_metrics(preds: np.ndarray, targets: np.ndarray,
                    finger_names=("ind", "med")) -> dict:
    """preds, targets: [N, F] -> metricas por dedo. F = len(finger_names)."""
    out = {}
    for i, finger in enumerate(finger_names):
        p = preds[:, i].astype(np.float64)
        t = targets[:, i].astype(np.float64)
        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        p_std = p.std(); t_std = t.std()
        pearson = float(np.mean((p - p.mean()) * (t - t.mean())) / (p_std * t_std + 1e-12))
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out[f"{finger}_mae"] = mae
        out[f"{finger}_rmse"] = rmse
        out[f"{finger}_pearson"] = pearson
        out[f"{finger}_r2"] = r2
    return out


def variance_matching_loss(pred, target):
    """L = mean over fingers of (std(pred_f) - std(target_f))^2.
    pred y target en formato [B, 2] (h=1) o [B, K, 2] (h>1)."""
    if pred.dim() == 3:
        # [B, K, 2] -> aplanar B*K para std por dedo
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
    losses = []
    for i in range(pred.shape[-1]):
        s_p = pred[:, i].std()
        s_t = target[:, i].std()
        losses.append((s_p - s_t) ** 2)
    return torch.stack(losses).mean()


def train_epoch(model, loader, optimizer, loss_fn, device, grad_clip,
                epoch: int, total_epochs: int, horizon: int = 1,
                accel_weight: float = 0.0, var_weight: float = 0.0,
                log_every: int = 50):
    """
    horizon == 1: pred [B, 2], y [B, 2]. Solo loss_data.
    horizon  > 1: pred [B, 2, K], y [B, K, 2]. Loss_data + accel_weight*L_accel.
        L_accel = MSE(d_pred, d_target) sobre las K-1 diferencias.
    """
    use_accel = horizon > 1 and accel_weight > 0
    use_var = var_weight > 0
    model.train()
    total, total_data, total_accel, total_var, n = 0.0, 0.0, 0.0, 0.0, 0
    n_batches = len(loader)
    t_start = time.time()
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if horizon == 1:
            pred = model(x, horizon=1)                # [B, 2]
            loss_data = loss_fn(pred, y)
            loss_accel = torch.tensor(0.0, device=device)
            pred_for_var = pred
        else:
            pred = model(x, horizon=horizon)          # [B, 2, K]
            pred_t = pred.permute(0, 2, 1).contiguous()   # [B, K, 2] - match y
            loss_data = loss_fn(pred_t, y)
            if use_accel:
                d_pred = pred_t[:, 1:, :] - pred_t[:, :-1, :]
                d_y    = y[:, 1:, :] - y[:, :-1, :]
                loss_accel = ((d_pred - d_y) ** 2).mean()
            else:
                loss_accel = torch.tensor(0.0, device=device)
            pred_for_var = pred_t
        loss_var = (variance_matching_loss(pred_for_var, y) if use_var
                    else torch.tensor(0.0, device=device))
        loss = loss_data + accel_weight * loss_accel + var_weight * loss_var
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = y.shape[0]
        total += loss.item() * bs
        total_data += loss_data.item() * bs
        total_accel += loss_accel.item() * bs
        total_var += loss_var.item() * bs
        n += bs
        if i % log_every == 0 or i == n_batches:
            sps = n / max(time.time() - t_start, 1e-6)
            extra = ""
            if use_accel: extra += f"  accel {total_accel/n:.4f}"
            if use_var:   extra += f"  var {total_var/n:.4f}"
            print(f"\r  ep {epoch}/{total_epochs}  train  batch {i}/{n_batches}  "
                  f"loss {total/n:.4f}  data {total_data/n:.4f}{extra}  "
                  f"{sps:>5.0f} samples/s", end="", flush=True)
    print()
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, horizon: int = 1,
             accel_weight: float = 0.0,
             finger_names=("ind", "med")):
    """Evalua sobre el LAST timestep de cada ventana cuando horizon>1
    (para comparar contra los mismos R^2/MAE que se reportan en horizon=1)."""
    use_accel = horizon > 1 and accel_weight > 0
    model.eval()
    all_preds, all_targets = [], []
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if horizon == 1:
            pred = model(x, horizon=1)                  # [B, output_dim]
            loss_data = loss_fn(pred, y)
            preds_last = pred
            y_last = y
        else:
            pred = model(x, horizon=horizon)            # [B, output_dim, K]
            pred_t = pred.permute(0, 2, 1).contiguous() # [B, K, output_dim]
            loss_data = loss_fn(pred_t, y)
            if use_accel:
                d_pred = pred_t[:, 1:, :] - pred_t[:, :-1, :]
                d_y    = y[:, 1:, :] - y[:, :-1, :]
                loss_data = loss_data + accel_weight * ((d_pred - d_y) ** 2).mean()
            preds_last = pred_t[:, -1, :]               # [B, output_dim]
            y_last = y[:, -1, :]                        # [B, output_dim]
        total += loss_data.item() * y.shape[0]; n += y.shape[0]
        all_preds.append(preds_last.detach().cpu().numpy())
        all_targets.append(y_last.detach().cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets, finger_names=finger_names)
    metrics["loss"] = total / max(n, 1)
    return metrics, preds, targets


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}")
    print(f"sigma_ms: {args.sigma_ms}")
    print(f"window_length: {args.window_length}, dilations: {args.dilations}")

    # ----- Datasets (1 por CSV, stride=1, igual que train.py) -----
    pp = PreprocessConfig2F(sigma_ms=args.sigma_ms,
                            feature_set=tuple(args.features))
    datasets = []
    for csv in args.csv:
        print(f"\nLoading {csv}")
        ds = EMGSession2FingersDataset(
            csv, window_length=args.window_length,
            stride=1, config=pp, verbose=True,
            horizon=args.horizon,
            exclude_tasks=tuple(args.exclude_tasks),
            target_finger=args.target_finger,
            feature_fingers=args.feature_fingers,
        )
        datasets.append(ds)

    # ----- Split por sesion (stratified default, temporal, o random_chunked) -----
    gap = args.gap_windows if args.gap_windows is not None else args.window_length
    if args.split_mode == "stratified":
        train_idx, val_idx = stratified_split_multi(
            datasets, val_fraction=args.val_fraction, gap=gap,
        )
    elif args.split_mode == "temporal":
        n_total = sum(len(d) for d in datasets)
        train_idx, val_idx = temporal_split(
            n=n_total, val_fraction=args.val_fraction, gap=gap,
        )
    else:  # random_chunked (legacy)
        train_idx, val_idx = random_split_with_gap_multi(
            datasets, args.val_fraction, gap_samples=gap, seed=args.seed,
        )
    print(f"\nSplit mode: {args.split_mode}")
    # Stride de muestreo sobre los indices (igual que train.py)
    train_idx = train_idx[:: args.train_stride]
    val_idx = val_idx[:: args.val_stride]
    print(f"  train windows: {len(train_idx):,}  (stride {args.train_stride})")
    print(f"  val   windows: {len(val_idx):,}  (stride {args.val_stride})")

    # ----- Z-normalizacion opcional por canal -----
    # Stats calculados solo sobre los SAMPLES de train (filtrados por exclude_tasks).
    # Aplicados in-place a TODOS los datasets (train y val). Guardados en config.
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
            mad = np.median(np.abs(all_train_feats - feat_center[:, None]),
                            axis=1).astype(np.float32)
            feat_scale = (mad * 1.4826 + 1e-6).astype(np.float32)
            label_c, label_s = "median", "MAD*1.4826"
        else:  # mean_std
            feat_center = all_train_feats.mean(axis=1).astype(np.float32)
            feat_scale = (all_train_feats.std(axis=1) + 1e-6).astype(np.float32)
            label_c, label_s = "mean", "std"
        print(f"\nFeature norm: {label_c} + {label_s} (excluyendo {args.exclude_tasks}):")
        for i, name in enumerate(datasets[0].feature_names):
            print(f"  [{i:>2}] {name:<28} {label_c}={feat_center[i]:+.4f}  "
                  f"{label_s}={feat_scale[i]:.4f}")
        for ds in datasets:
            ds.apply_feature_norm(feat_center, feat_scale)
        if args.clip_value > 0:
            n_total_feat = sum(ds.features.size for ds in datasets)
            n_clipped = sum(int(np.sum(np.abs(ds.features) > args.clip_value))
                            for ds in datasets)
            print(f"Clip features post-norm a ±{args.clip_value}: "
                  f"{n_clipped:,} de {n_total_feat:,} samples afectados "
                  f"({n_clipped/max(n_total_feat,1)*100:.3f}%)")
            for ds in datasets:
                np.clip(ds.features, -args.clip_value, args.clip_value, out=ds.features)

    concat = ConcatDataset(datasets)
    train_set = Subset(concat, train_idx)
    val_set = Subset(concat, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    # ----- Model -----
    finger_names = ("ind", "med") if args.target_finger == "both" else (args.target_finger,)
    out_dim = len(finger_names)
    in_channels = datasets[0].features.shape[0]   # dinamico segun features/fingers
    model = EMGTCN(
        in_channels=in_channels,
        hidden_channels=args.hidden_channels,
        output_dim=out_dim,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dilations=args.dilations,
        use_weight_norm=not args.no_weight_norm,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: EMGTCN(in=16, hidden={args.hidden_channels}, out={out_dim}, "
          f"target_finger={args.target_finger}, dilations={args.dilations})")
    print(f"  RF: {model.receptive_field} samples")
    print(f"  Params: {n_params:,}")

    # ----- Loss / Optim -----
    loss_fn = {
        "huber": nn.HuberLoss(delta=args.huber_delta),
        "l1":    nn.L1Loss(),
        "mse":   nn.MSELoss(),
    }[args.loss]
    print(f"data loss: {args.loss}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    # ----- Harmonizar target_scale entre sesiones -----
    # Cada dataset calculo su propio p99 de |vel|. Para que el modelo vea una
    # escala consistente en train/val (y para que inference use la misma escala
    # despues), recomputamos un scale global sobre todas las sesiones.
    def _harmonize(datasets, attr_scale_name: str, raw_target_idx: int):
        """attr_scale_name: 'target_scale_ind' o 'target_scale_med'.
           raw_target_idx: 0 = ind, 1 = med en target [2, T] pre-norm."""
        all_raw_norm = []  # ya estan en escala normalizada por dataset
        for ds in datasets:
            cur_scale = getattr(ds, attr_scale_name)
            # devolver a unidad original y concatenar
            all_raw_norm.append(np.abs(ds.target[raw_target_idx]) * cur_scale)
        all_raw = np.concatenate(all_raw_norm)
        global_scale = float(np.percentile(all_raw, pp.target_scale_percentile))
        if global_scale <= 0:
            global_scale = 1.0
        for ds in datasets:
            cur_scale = getattr(ds, attr_scale_name)
            ds.target[raw_target_idx] = (ds.target[raw_target_idx] * cur_scale
                                          / global_scale).astype(np.float32)
            setattr(ds, attr_scale_name, global_scale)
        return global_scale

    target_scale_ind = _harmonize(datasets, "target_scale_ind", 0)
    target_scale_med = _harmonize(datasets, "target_scale_med", 1)
    print(f"\nGlobal target_scale: ind={target_scale_ind:.2f}  med={target_scale_med:.2f}")

    # ----- Output paths -----
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    config_out = {
        **vars(args),
        "in_channels": in_channels,
        "output_dim": out_dim,
        "target_finger": args.target_finger,
        "receptive_field": model.receptive_field,
        "n_params": n_params,
        "target_scale_ind": target_scale_ind,
        "target_scale_med": target_scale_med,
        "feature_names": list(datasets[0].feature_names),
        "feature_center": feat_center.tolist() if feat_center is not None else None,
        "feature_scale":  feat_scale.tolist() if feat_scale is not None else None,
        "feature_norm":   args.feature_norm,
        "clip_value":     args.clip_value,
        "fs": float(datasets[0].fs),
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)

    # ----- Train loop -----
    eval_dev = torch.device("cpu") if args.eval_device == "cpu" else device
    if eval_dev != device:
        print(f"eval_device: {eval_dev}  (train: {device})")
    best_val_r2 = -np.inf
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device,
                                 args.grad_clip, epoch, args.epochs,
                                 horizon=args.horizon,
                                 accel_weight=args.accel_weight,
                                 var_weight=args.var_weight)
        # Eval cada N epochs (ultimo siempre se evalua).
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        if not do_eval:
            print(f"  ep {epoch}/{args.epochs}  train_loss {train_loss:.4f}  "
                  f"(skip eval)")
            history.append({"epoch": epoch, "train_loss": train_loss})
            continue
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        if eval_dev != device:
            model.to(eval_dev)
        val = evaluate(model, val_loader, loss_fn, eval_dev,
                       horizon=args.horizon,
                       accel_weight=args.accel_weight,
                       finger_names=finger_names)[0]
        if eval_dev != device:
            model.to(device)
        avg_r2 = float(np.mean([val[f"{f}_r2"] for f in finger_names]))
        parts = [f"  ep {epoch}/{args.epochs}  val_loss {val['loss']:.4f}"]
        for f in finger_names:
            parts.append(f"  {f} r2={val[f'{f}_r2']:+.3f} "
                         f"mae={val[f'{f}_mae']:.3f} pearson={val[f'{f}_pearson']:+.3f}")
        print("  |".join(parts))
        history.append({"epoch": epoch, "train_loss": train_loss, **val})
        if avg_r2 > best_val_r2:
            best_val_r2 = avg_r2
            torch.save({
                "model_state": model.state_dict(),
                "config": config_out,
                "in_channels": in_channels,
                "output_dim": out_dim,
                "target_scale_ind": target_scale_ind,
                "target_scale_med": target_scale_med,
                "target_finger": args.target_finger,
                "epoch": epoch,
                "val": val,
            }, out_dir / "best.pt")
            print(f"  -> best checkpoint (avg r2 = {avg_r2:+.3f})")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nDone. Best avg r2: {best_val_r2:+.3f}. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
