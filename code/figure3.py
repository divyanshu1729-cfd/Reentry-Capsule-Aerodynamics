"""
===========================================================
Project  : Re-entry Vehicle Analysis
Reference: NACA Report 1381

Figure 3
Variation of Deceleration with Altitude

Authors:
    Divyanshu
    OpenAI ChatGPT
===========================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from ballistics2 import (
    solve_trajectory,
    velocity_approx,
    numerical_deceleration,
    density
)
# -------------------------------------------------------
# Sphere properties
# -------------------------------------------------------

Cd = 0.92
A  = np.pi*(1.0**2)/4.0
m  = 7.9742

entry_velocities = [10000,20000,30000]
colors = ["tab:blue","tab:green","tab:red"]

plt.figure(figsize=(8,8))

for Ve, color in zip(entry_velocities, colors):

    # --------------------------------------------
    # Numerical solution
    # --------------------------------------------

    sol = solve_trajectory(
        Ve,
        Cd=Cd,
        A=A,
        m=m
    )

    altitude = sol.y[0] / 1000.0
    velocity = sol.y[1]

    decel_num = numerical_deceleration(
        sol.y[0],
        velocity,
        Cd,
        A,
        m
    )

    decel_num = numerical_deceleration(
        sol.y[0],
        velocity,
        Cd,
        A,
        m
    )

    plt.plot(
    altitude,
    decel_num,
    color=color,
    linewidth=2.8,
    label=f"$V_E$ = {Ve:,} ft/s"
)

    # --------------------------------------------
    # Approximation (Eq. 9 + Eq. 10)
    # --------------------------------------------

    y = np.linspace(0,200000,600)

    V = velocity_approx(
        y,
        Ve,
        Cd,
        A,
        m
    )

    rho = density(y)

    decel = Cd*rho*A*V**2/(2*m*32.174)

    plt.plot(
        y/1000,
        decel,
        "--",
        color=color,
        lw=2.3
    )

# -------------------------------------------------------
# Formatting
# -------------------------------------------------------

plt.xlim(0,200)
plt.ylim(0,250)

plt.xticks(np.arange(0,201,50))
plt.yticks(np.arange(0,251,50))

plt.minorticks_on()

plt.grid(which="major",linewidth=0.8)
plt.grid(which="minor",linestyle=":",alpha=0.5)

plt.xlabel("Altitude (ft × $10^{-3}$)",fontsize=14)

plt.ylabel(r"Deceleration, $-\frac{dV/dt}{g}$",fontsize=14)

plt.legend()

plt.tight_layout()

plt.savefig("../figures/Figure3_Deceleration.png",dpi=600)

plt.show()
