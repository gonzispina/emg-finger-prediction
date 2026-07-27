"""
Predicción vs ground truth del modelo TCN campeón (sig050, 200ep, S02+S03).
Usa best_val_preds.npz — no requiere correr el modelo.

Genera:
  1. pred_vs_target_tasks.png  — segmento temporal por task (fast_ext, fast_flex, free_controlled)
  2. pred_scatter.png          — scatter pred vs target coloreado por task
"""
import numpy as np
import matplotlib.pyplot as plt
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)

NPZ = "checkpoints/tcn_s02s03_RF567_sobolev01_200ep/best_val_preds.npz"

d       = np.load(NPZ, allow_pickle=True)
preds   = d["preds"].astype(float)
targets = d["targets"].astype(float)
tasks   = d["tasks"]

# ── FIGURA 1: segmentos temporales por task ────────────────────────────────────
SHOW_TASKS = ["fast_flex", "fast_ext", "free_controlled", "slow_flex", "slow_ext"]
COLORS = {
    "fast_flex":      ("tomato",     "lightcoral"),
    "fast_ext":       ("steelblue",  "lightblue"),
    "free_controlled":("darkorange", "peachpuff"),
    "slow_flex":      ("mediumpurple","thistle"),
    "slow_ext":       ("seagreen",   "lightgreen"),
}
N_SHOW = 600   # samples a mostrar por task (~1s a 605Hz)

fig, axes = plt.subplots(len(SHOW_TASKS), 1, figsize=(13, 2.8 * len(SHOW_TASKS)))
fig.suptitle("Predicción TCN vs ground truth  (val set, S02+S03, σ=50ms)",
             fontsize=13, y=1.01)

for ax, task in zip(axes, SHOW_TASKS):
    mask = tasks == task
    p = preds[mask][:N_SHOW]
    t = targets[mask][:N_SHOW]
    r2 = r2_score(targets[mask], preds[mask])
    t_ax = np.arange(len(p)) / 605 * 1000  # ms

    c_pred, c_target = COLORS[task]
    ax.plot(t_ax, t, color=c_target, lw=1.5, alpha=0.9, label="Ground truth")
    ax.plot(t_ax, p, color=c_pred,   lw=1.5, alpha=0.9, label=f"Predicción  R²={r2:.3f}")
    ax.axhline(0, color="gray", lw=0.7, ls=":")
    ax.set_ylabel("Vel. norm.", fontsize=9)
    ax.set_title(task.replace("_", " "), fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, t_ax[-1])

axes[-1].set_xlabel("Tiempo (ms)", fontsize=10)
plt.tight_layout()
plt.savefig("plots/pred_vs_target_tasks.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/pred_vs_target_tasks.png")
plt.close()

# ── FIGURA 2: scatter pred vs target ──────────────────────────────────────────
SCATTER_TASKS = {
    "fast_flex":       ("tomato",      "Flexión rápida"),
    "fast_ext":        ("steelblue",   "Extensión rápida"),
    "free_controlled": ("darkorange",  "Libre"),
    "slow_flex":       ("mediumpurple","Flexión lenta"),
    "slow_ext":        ("seagreen",    "Extensión lenta"),
}

fig, ax = plt.subplots(figsize=(7, 7))
for task, (color, label) in SCATTER_TASKS.items():
    mask = tasks == task
    r2 = r2_score(targets[mask], preds[mask])
    ax.scatter(targets[mask][::4], preds[mask][::4],
               color=color, alpha=0.15, s=4, label=f"{label}  R²={r2:.3f}")

lim = 1.6
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5, label="Predicción perfecta")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("Ground truth (vel. norm.)", fontsize=11)
ax.set_ylabel("Predicción TCN (vel. norm.)", fontsize=11)
r2_all = r2_score(targets, preds)
ax.set_title(f"Scatter predicción vs ground truth\nR² global = {r2_all:.3f}", fontsize=12)
ax.legend(fontsize=8, markerscale=4)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plots/pred_scatter.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/pred_scatter.png")
plt.close()
