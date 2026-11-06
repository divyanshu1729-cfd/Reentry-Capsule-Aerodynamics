"""
shock_relations.py
==================

Classical oblique-shock relations for a perfect gas.

These relations provide the post-shock thermodynamic state
required before integrating the Taylor–Maccoll equations.

Author : Divyanshu
Project: Allen–Eggers Reentry Analysis
"""

import numpy as np
from constants import GAMMA


# ==========================================================
# Normal Mach number
# ==========================================================

def normal_mach(M1, beta):
    """
    Upstream normal Mach number.

    Parameters
    ----------
    M1 : float
        Freestream Mach number.

    beta : float
        Shock angle (radians).
    """
    return M1 * np.sin(beta)


# ==========================================================
# Pressure ratio
# ==========================================================

def pressure_ratio(M1n):
    """
    p2 / p1
    """
    g = GAMMA

    return 1.0 + (2.0 * g / (g + 1.0)) * (M1n**2 - 1.0)


# ==========================================================
# Density ratio
# ==========================================================

def density_ratio(M1n):
    """
    rho2 / rho1
    """
    g = GAMMA

    return ((g + 1.0) * M1n**2) / (
        (g - 1.0) * M1n**2 + 2.0
    )


# ==========================================================
# Temperature ratio
# ==========================================================

def temperature_ratio(M1n):
    """
    T2 / T1
    """
    return pressure_ratio(M1n) / density_ratio(M1n)


# ==========================================================
# Downstream normal Mach
# ==========================================================

def downstream_normal_mach(M1n):
    """
    M2n
    """
    g = GAMMA

    numerator = 1.0 + 0.5 * (g - 1.0) * M1n**2

    denominator = (
        g * M1n**2
        - 0.5 * (g - 1.0)
    )

    return np.sqrt(numerator / denominator)


# ==========================================================
# Downstream Mach number
# ==========================================================

def downstream_mach(M1, beta, theta):
    """
    M2 immediately behind the oblique shock.
    """

    M2n = downstream_normal_mach(
        normal_mach(M1, beta)
    )

    return M2n / np.sin(beta - theta)


# ==========================================================
# Stagnation-pressure ratio
# ==========================================================

def stagnation_pressure_ratio(M1n):
    """
    p02 / p01

    Exact normal-shock relation.
    """

    g = GAMMA

    term1 = (
        ((g + 1.0) * M1n**2)
        /
        ((g - 1.0) * M1n**2 + 2.0)
    ) ** (g / (g - 1.0))

    term2 = (
        (g + 1.0)
        /
        (2.0 * g * M1n**2 - (g - 1.0))
    ) ** (1.0 / (g - 1.0))

    return term1 * term2


# ==========================================================
# Verification
# ==========================================================

if __name__ == "__main__":

    from taylor_maccoll import (
        theta_beta_mach,
        find_shock_angle,
    )

    print("=" * 60)
    print("OBLIQUE SHOCK VERIFICATION")
    print("=" * 60)

    M1 = 5.0

    theta_deg = 20.0

    beta_deg = find_shock_angle(
        mach=M1,
        theta_c_deg=theta_deg
    )

    beta = np.deg2rad(beta_deg)

    theta = theta_beta_mach(
        beta,
        M1
    )

    M1n = normal_mach(
        M1,
        beta
    )

    print()

    print(f"Freestream Mach      = {M1:.4f}")
    print(f"Shock angle          = {beta_deg:.4f} deg")
    print(f"Flow deflection      = {np.rad2deg(theta):.4f} deg")

    print()

    print(f"M1n                  = {M1n:.6f}")

    print(f"Pressure ratio       = {pressure_ratio(M1n):.6f}")

    print(f"Density ratio        = {density_ratio(M1n):.6f}")

    print(f"Temperature ratio    = {temperature_ratio(M1n):.6f}")

    print(f"Downstream Mach      = {downstream_mach(M1,beta,theta):.6f}")

    print(f"p02/p01              = {stagnation_pressure_ratio(M1n):.6f}")
