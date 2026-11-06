"""
============================================================
Project  : Re-entry Vehicle Analysis
Reference: NACA Report 1381 (Allen & Eggers)

File:
    ballistic.py

Purpose:
    Implements the analytical ballistic-entry equations
    (Equations 8 and 9).

Author:
    Divyanshu

Assisted by:
    OpenAI ChatGPT (GPT-5.5), Anthropic Claude
============================================================
"""

import numpy as np

from constants import (
    RHO0,
    BETA,
    ENTRY_VELOCITIES,
)

# ----------------------------------------------------------
# Ballistic body
# ----------------------------------------------------------

# 1-ft diameter iron sphere

DIAMETER = 1.0          # ft
RADIUS = DIAMETER/2.0

AREA = np.pi*RADIUS**2

VOLUME = (4.0/3.0)*np.pi*RADIUS**3

# Density of solid iron

RHO_IRON = 490.0/32.174        # slug/ft^3

MASS = RHO_IRON*VOLUME

# Hypersonic sphere drag coefficient

CD = 0.92

# ----------------------------------------------------------
# Ballistic parameter
# ----------------------------------------------------------

K = CD*RHO0*AREA/(2.0*MASS)

B = K/BETA

# ----------------------------------------------------------
# Velocity
# ----------------------------------------------------------

def velocity(y_ft, Ve):

    """
    Equation (9)

    Parameters
    ----------
    y_ft : altitude (ft)

    Ve : entry velocity (ft/s)
    """

    return Ve*np.exp(-B*np.exp(-BETA*y_ft))


# ----------------------------------------------------------
# Deceleration
# ----------------------------------------------------------

def deceleration(y_ft, Ve):

    """
    Equation (8)
    Gravity neglected
    """

    rho = RHO0*np.exp(-BETA*y_ft)

    V = velocity(y_ft, Ve)

    return K*rho*V**2


# ----------------------------------------------------------
# Utility
# ----------------------------------------------------------

def print_summary():

    print("\nAllen-Eggers Ballistic Body\n")

    print(f"Diameter      : {DIAMETER:.3f} ft")
    print(f"Area          : {AREA:.6f} ft²")
    print(f"Volume        : {VOLUME:.6f} ft³")
    print(f"Mass          : {MASS:.4f} slugs")
    print(f"Cd            : {CD:.3f}")
    print(f"K             : {K:.8f}")
    print(f"B             : {B:.5f}")


if __name__ == "__main__":

    print_summary()
