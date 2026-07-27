"""
Curvas de entrenamiento del modelo TCN 2 dedos (S04-S07, 100ep).
"""
import json
import numpy as np
import matplotlib.pyplot as plt

LOG = "checkpoints/tcn_2f_S04S05S06S07_rms2_h48_rf567_100ep/history.json"

with open(LOG) as f:
    log = json.load(f)

# Solo épocas que tienen métricas de val
log_val    = [e for e in log if "ind_r2" in e]
epochs     = [e["epoch"]       for e in log_val]
train_loss = [e["train_loss"]  for e in log_val]
val_loss   = [e["loss"]        for e in log_val]
ind_r2     = [e["ind_r2"]      for e in log_val]
med_r2     = [e["med_r2"]      for e in log_val]
ind_pear   = [e["ind_pearson"] for e in log_val]
med_pear   = [e["med_pearson"] for e in log_val]

avg_r2   = [(a + b) / 2 for a, b in zip(ind_r2, med_r2)]
best_ep  = max(log_val, key=lambda x: (x["ind_r2"] + x["med_r2"]) / 2)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
fig.suptitle("Curvas de entrenamiento — TCN 2 dedos (S04–S07, 100 epochs)", fontsize=13)

ax1.plot(epochs, train_loss, color="steelblue", lw=1.5, label="train loss")
ax1.plot(epochs, val_loss,   color="tomato",    lw=1.5, label="val loss")
ax1.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--",
            label=f"best (ep {best_ep['epoch']})")
ax1.set_ylabel("Huber Loss"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, ind_r2,  color="steelblue",   lw=1.5, label=f"R² índice  (final={ind_r2[-1]:.3f})")
ax2.plot(epochs, med_r2,  color="darkorange",  lw=1.5, label=f"R² medio   (final={med_r2[-1]:.3f})")
ax2.plot(epochs, avg_r2,  color="black",       lw=1.2, ls="--",
         label=f"R² promedio (final={avg_r2[-1]:.3f})")
ax2.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--")
ax2.set_ylabel("R² (val)"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

ax3.plot(epochs, ind_pear, color="steelblue",  lw=1.5, label=f"Pearson índice  (final={ind_pear[-1]:.3f})")
ax3.plot(epochs, med_pear, color="darkorange", lw=1.5, label=f"Pearson medio   (final={med_pear[-1]:.3f})")
ax3.axvline(best_ep["epoch"], color="green", lw=1.2, ls="--")
ax3.set_ylabel("Pearson r (val)"); ax3.set_xlabel("Epoch")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/training_curve_2f.png", dpi=150, bbox_inches="tight")
print(f"Guardado: plots/training_curve_2f.png")
print(f"Best epoch {best_ep['epoch']}: ind R²={best_ep['ind_r2']:.3f}, med R²={best_ep['med_r2']:.3f}, "
      f"avg R²={(best_ep['ind_r2']+best_ep['med_r2'])/2:.3f}")
plt.show()
