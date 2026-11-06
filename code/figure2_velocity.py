"""
Figure 2
Velocity variation with altitude
Reproduction of NACA Report 1381
"""

import numpy as np
import matplotlib.pyplot as plt

import ballistics2

# --------------------------------------------------
# Ballistic body
# --------------------------------------------------

Cd = 0.92
A = np.pi / 4.0
m = 7.9742

# --------------------------------------------------
# Altitude for analytical solution
# --------------------------------------------------

y = np.linspace(0.0, 280000.0, 600)

fig, ax = plt.subplots(figsize=(7, 6))

for Ve in [10000.0, 20000.0, 30000.0]:

    # -------------------------
    # Analytical (dashed)
    # -------------------------

    V_approx = ballistics2.velocity_approx(
        y,
        Ve,
        Cd,
        A,
        m
    )

    ax.plot(
        y / 1000.0,
        V_approx / 1000.0,
        "--",
        linewidth=2,
        label=f"Approx. {Ve/1000:.0f}×10³ ft/s"
    )

    # -------------------------
    # Numerical (solid)
    # -------------------------

    sol = ballistics2.solve_trajectory(
        Ve,
        Cd=Cd,
        A=A,
        m=m
    )

    altitude = sol.y[0]
    velocity = sol.y[1]

    ax.plot(
        altitude / 1000.0,
        velocity / 1000.0,
        linewidth=2
    )

# --------------------------------------------------

# --------------------------------------------------
# Axis labels
# --------------------------------------------------

ax.set_xlabel("Altitude (ft × $10^{-3}$)", fontsize=12)
ax.set_ylabel("Velocity (ft/s × $10^{-3}$)", fontsize=12)

# --------------------------------------------------
# Axis limits
# --------------------------------------------------

ax.set_xlim(50, 200)
ax.set_ylim(0, 30)

# --------------------------------------------------
# Tick marks (match NACA Figure 2)
# --------------------------------------------------

ax.set_xticks(np.arange(0, 201, 50))
ax.set_yticks(np.arange(0, 31, 5))

# --------------------------------------------------
# Major and minor grids
# --------------------------------------------------

ax.minorticks_on()

ax.grid(which="major",
        linestyle="-",
        linewidth=0.8,
        alpha=0.8)

ax.grid(which="minor",
        linestyle=":",
        linewidth=0.5,
        alpha=0.5)

ax.tick_params(axis='both',
               which='major',
               direction='in',
               length=6)

ax.tick_params(axis='both',
               which='minor',
               direction='in',
               length=3)

ax.legend()

plt.tight_layout()

plt.savefig("../figures/Figure02_Velocity.png", dpi=600)

plt.show()
