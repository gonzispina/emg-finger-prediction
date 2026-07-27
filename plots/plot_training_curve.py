"""
Curvas de entrenamiento: train_loss vs val_loss a lo largo de los epochs.
Usa el log del champion actual (h48, adapsig, attn04, clip300, S02-S14).
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

LOG_PATH = "checkpoints/tcn_s02s03s08s09s10s11s12s13s14_rawrms_RF567_h48_adapsig50_80_attn04_clip300_plateau_100ep/log.json"

with open(LOG_PATH) as f:
    log = json.load(f)

epochs      = [e["epoch"]      for e in log]
train_loss  = [e["train_loss"] for e in log]
val_loss    = [e["val_loss"]   for e in log]
val_r2      = [e["val_r2"]     for e in log]
val_pearson = [e["val_pearson"] for e in log]

best_ep = max(log, key=lambda x: x["val_r2"])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle("Curvas de entrenamiento — TCN 1 dedo (S02–S14, 100 epochs, h48)", fontsize=13)

ax1.plot(epochs, train_loss, label="train loss", color="steelblue", lw=1.5)
ax1.plot(epochs, val_loss,   label="val loss",   color="tomato",    lw=1.5)
ax1.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--", label=f"best (ep {best_ep['epoch']})")
ax1.set_ylabel("Huber Loss")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_r2, color="darkorange", lw=1.5)
ax2.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--")
ax2.axhline(best_ep["val_r2"], color="gray", lw=0.8, ls=":",
            label=f"R²={best_ep['val_r2']:.3f}")
ax2.set_ylabel("R² (val)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

ax3.plot(epochs, val_pearson, color="mediumpurple", lw=1.5)
ax3.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--")
ax3.axhline(best_ep["val_pearson"], color="gray", lw=0.8, ls=":",
            label=f"Pearson={best_ep['val_pearson']:.3f}")
ax3.set_ylabel("Pearson r (val)")
ax3.set_xlabel("Epoch")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/training_curve.png", dpi=150, bbox_inches="tight")
print(f"Guardado: plots/training_curve.png  "
      f"(best ep{best_ep['epoch']}, R²={best_ep['val_r2']:.3f}, Pearson={best_ep['val_pearson']:.3f})")
plt.show()
