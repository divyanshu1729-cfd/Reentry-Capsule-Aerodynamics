
"""
===========================================================
Figure 1

Variation of density with altitude

Reproduces Figure 1 of
NACA Report 1381
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from atmosphere import density

# Altitude (ft)
y = np.linspace(0.0, 280000.0, 500)

rho = density(y)

plt.figure(figsize=(5.5,5.5))
plt.minorticks_on()
plt.semilogy(
    y/1000.0,
    rho,
    linewidth=2,
    label = r"$\rho=\rho_0 e^{-\beta y}$"
)
plt.legend()

plt.grid(which="major", linestyle="-")
plt.grid(which="minor", linestyle=":")
plt.tight_layout(pad=1.2)

plt.xlabel(r"Altitude (×10$^{-3}$ ft)")
plt.ylabel(r"Density, $\rho$ (slug ft$^{-3}$")
plt.ylim(1e-7, 1e-2)
ticks = np.arange(0, 281, 40)

plt.xticks(ticks)

plt.grid(True, which="both")

plt.tight_layout()

plt.savefig("../figures/Figure01_Density.png", dpi=300)

plt.show()
