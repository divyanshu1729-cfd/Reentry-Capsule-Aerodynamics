"""
=========================================================
Figure 4
Universal Functions F', F'', F'''
Allen & Eggers (NACA Report 1381)

Author:
    Divyanshu
    OpenAI ChatGPT

=========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Independent variable
# x = βΔy
# --------------------------------------------------------

x = np.linspace(-3.0, 5.5, 800)

# --------------------------------------------------------
# Universal functions
# --------------------------------------------------------

F1 = np.exp(-0.5*np.exp(-x))

F2 = np.exp(-x) * np.exp(1.0 - np.exp(-x))

F3 = np.exp(-0.5*x) * np.exp(0.5*(1.0 - np.exp(-x)))

# --------------------------------------------------------
# Plot
# --------------------------------------------------------

plt.figure(figsize=(7.2,6.2))

plt.plot(
    x,
    F1,
    color='black',
    linewidth=2.5,
    label=r"$F'(\beta\Delta y)$"
)

plt.plot(
    x,
    F2,
    color='black',
    linewidth=2.5,
    linestyle='-',
    label=r"$F''(\beta\Delta y)$"
)

plt.plot(
    x,
    F3,
    color='black',
    linewidth=2.5,
    linestyle='--',
    dashes=(10,6),
    label=r"$F'''(\beta\Delta y)$"
)

# --------------------------------------------------------
# Axes
# --------------------------------------------------------

plt.xlim(-3,5.0)
plt.ylim(0,1.0)

plt.xticks(np.arange(-3,6,1))
plt.yticks(np.arange(0,1.1,0.2))

plt.xlabel(r'$\beta\Delta y$',fontsize=16)
plt.ylabel(r"$F'(\beta\Delta y),\;F''(\beta\Delta y),\;F'''(\beta\Delta y)$",
           fontsize=16)

# --------------------------------------------------------
# Grid
# --------------------------------------------------------

plt.minorticks_on()

plt.grid(True,
         which='major',
         linestyle='-',
         linewidth=0.9,
         alpha=0.8)

plt.grid(True,
         which='minor',
         linestyle=':',
         linewidth=0.5,
         alpha=0.5)

# --------------------------------------------------------
# Labels positioned like the report
# --------------------------------------------------------

plt.text(3.25,0.92,r"$F'(\beta\Delta y)$",fontsize=14)

plt.text(3.45,0.36,r"$F'''(\beta\Delta y)$",fontsize=14)

plt.text(4.55,0.08,r"$F''(\beta\Delta y)$",fontsize=14)

plt.tight_layout()

plt.savefig(
    "../figures/Figure4_Functions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
