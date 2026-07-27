"""
R² por task — champion (h48, S02-S14, adapsig, attn04, clip300).
Evaluado sobre su val holdout.
"""
import numpy as np
import matplotlib.pyplot as plt

tasks = ["fast_flex", "fast_ext", "free_controlled", "microcorrections",
         "slow_flex", "slow_ext", "slow_cycles", "holds_intermediate"]
labels = [t.replace("_", "\n") for t in tasks]
x = np.arange(len(tasks))

r2_champion = [0.842, 0.712, 0.768, 0.449, 0.604, 0.587, 0.577, 0.101]

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(x, r2_champion, 0.6, color="steelblue", alpha=0.9)

ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("R²")
ax.set_ylim(-0.05, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("R² por task — TCN Champion  (h48, S02–S14, val holdout)", fontsize=12)
ax.grid(True, axis="y", alpha=0.3)

ax.text(0.99, 0.97, f"val R² = 0.688",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

plt.tight_layout()
plt.savefig("plots/r2_by_task.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/r2_by_task.png")
plt.show()
