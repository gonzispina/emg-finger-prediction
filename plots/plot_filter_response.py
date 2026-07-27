"""
Respuesta en frecuencia del filtro Butterworth pasa banda 70-300 Hz.
Muestra ganancia en dB y fase, con región de paso resaltada.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

FS = 638.0
LOW_HZ = 70.0
HIGH_HZ = 300.0
ORDER = 4

b, a = signal.butter(ORDER, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=FS)
w, h = signal.freqz(b, a, worN=4096, fs=FS)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
fig.suptitle(f"Butterworth pasa banda {LOW_HZ:.0f}–{HIGH_HZ:.0f} Hz  (orden {ORDER})", fontsize=13)

# Ganancia
gain_db = 20 * np.log10(np.abs(h) + 1e-12)
ax1.plot(w, gain_db, color="steelblue", lw=2)
ax1.axvspan(LOW_HZ, HIGH_HZ, alpha=0.12, color="green", label="Banda de paso")
ax1.axvline(50, color="red", lw=1, ls="--", label="50 Hz (ruido red)")
ax1.set_ylabel("Ganancia (dB)")
ax1.set_ylim(-80, 5)
ax1.axhline(-3, color="gray", lw=0.8, ls=":")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, FS / 2)

# Fase
phase_deg = np.angle(h, deg=True)
ax2.plot(w, phase_deg, color="darkorange", lw=2)
ax2.axvspan(LOW_HZ, HIGH_HZ, alpha=0.12, color="green")
ax2.axvline(50, color="red", lw=1, ls="--")
ax2.set_ylabel("Fase (grados)")
ax2.set_xlabel("Frecuencia (Hz)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plots/filter_response.png", dpi=150, bbox_inches="tight")
print("Guardado: plots/filter_response.png")
plt.show()
