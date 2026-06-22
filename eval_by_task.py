"""
Calcula R^2, Pearson, MAE por task para un checkpoint sobre uno o mas CSVs.

Uso:
    python eval_by_task.py \\
        --checkpoint checkpoints/tcn_2f_s04s05_h64_rf243_var2_sobolev_50ep/best.pt \\
        --csv data/raw/P01/P01_S05_20260618_172229.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from dataset.emg_dataset_2fingers import EMGSession2FingersDataset, PreprocessConfig2F
from models import EMGTCN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True, nargs="+")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--exclude-tasks", nargs="+", default=["transition"])
    p.add_argument("--warmup-s", type=float, default=1.0,
                   help="excluir primeros N segundos (transient de filtros).")
    return p.parse_args()


def metrics(pred, target):
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    sp, st = pred.std(), target.std()
    if sp < 1e-12 or st < 1e-12:
        return {"mae": mae, "pearson": 0.0, "r2": 0.0, "std_p": float(sp), "std_t": float(st)}
    pearson = float(np.mean((pred - pred.mean()) * (target - target.mean())) / (sp * st))
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return {"mae": mae, "pearson": pearson, "r2": r2,
            "std_p": float(sp), "std_t": float(st), "n": len(pred)}


@torch.no_grad()
def predict_full_sequence(model, features, device):
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    h = x
    for block in model.blocks:
        h = block(h)
    if model.has_final_conv:
        h = model.final_drop(model.final_act(model.final_conv(h)))
    h = model.head(h)
    return h[0].cpu().numpy()   # [output_dim, T]


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = EMGTCN(
        in_channels=cfg.get("in_channels", 16),
        hidden_channels=cfg["hidden_channels"],
        output_dim=cfg.get("output_dim", 2),
        kernel_size=cfg["kernel_size"],
        dropout=cfg["dropout"],
        dilations=cfg["dilations"],
        use_weight_norm=False,
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state"])

    sigma_ms = float(cfg.get("sigma_ms", 50.0))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  RF: {model.receptive_field}  hidden: {cfg['hidden_channels']}  "
          f"sigma_ms: {sigma_ms}")

    # Accumular preds/targets/tasks sobre todos los CSVs
    all_pred_ind, all_pred_med = [], []
    all_targ_ind, all_targ_med = [], []
    all_tasks = []

    for csv in args.csv:
        print(f"\nLoading {csv}")
        ds = EMGSession2FingersDataset(
            csv, window_length=cfg["window_length"], stride=1,
            config=PreprocessConfig2F(sigma_ms=sigma_ms),
            verbose=False,
        )
        warmup = int(round(ds.fs * args.warmup_s))
        preds_norm = predict_full_sequence(model, ds.features, device)  # [2, T]
        # ds.target shape [2, T] ya normalizado.
        # Mascarar warmup + tasks excluidos por sample
        valid = np.ones(ds.features.shape[1], dtype=bool)
        valid[:warmup] = False
        for tname in args.exclude_tasks:
            valid &= ds.tasks != tname

        all_pred_ind.append(preds_norm[0, valid])
        all_pred_med.append(preds_norm[1, valid])
        all_targ_ind.append(ds.target[0, valid])
        all_targ_med.append(ds.target[1, valid])
        all_tasks.append(ds.tasks[valid])

    pred_ind = np.concatenate(all_pred_ind)
    pred_med = np.concatenate(all_pred_med)
    targ_ind = np.concatenate(all_targ_ind)
    targ_med = np.concatenate(all_targ_med)
    tasks = np.concatenate(all_tasks)

    # ----- Metrics por task -----
    rows = []
    for task in sorted(np.unique(tasks)):
        mask = tasks == task
        m_ind = metrics(pred_ind[mask], targ_ind[mask])
        m_med = metrics(pred_med[mask], targ_med[mask])
        rows.append({
            "task": task,
            "n": int(mask.sum()),
            "ind_r2": m_ind["r2"],
            "ind_p":  m_ind["pearson"],
            "ind_mae":m_ind["mae"],
            "ind_std_ratio": m_ind["std_p"] / max(m_ind["std_t"], 1e-12),
            "med_r2": m_med["r2"],
            "med_p":  m_med["pearson"],
            "med_mae":m_med["mae"],
            "med_std_ratio": m_med["std_p"] / max(m_med["std_t"], 1e-12),
        })

    # Overall
    overall_ind = metrics(pred_ind, targ_ind)
    overall_med = metrics(pred_med, targ_med)

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}")
    pd.set_option("display.width", 180)
    print("\n=== Metricas por task (excluye warmup + exclude_tasks) ===")
    print(df.to_string(index=False))

    print(f"\n=== Overall ===")
    print(f"  ind: r2={overall_ind['r2']:+.3f}  pearson={overall_ind['pearson']:+.3f}  "
          f"mae={overall_ind['mae']:.3f}  std_ratio={overall_ind['std_p']/overall_ind['std_t']:.3f}")
    print(f"  med: r2={overall_med['r2']:+.3f}  pearson={overall_med['pearson']:+.3f}  "
          f"mae={overall_med['mae']:.3f}  std_ratio={overall_med['std_p']/overall_med['std_t']:.3f}")


if __name__ == "__main__":
    main()
