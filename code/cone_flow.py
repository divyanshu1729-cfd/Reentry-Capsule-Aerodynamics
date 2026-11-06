"""
cone_flow.py
============

Conical flow utilities built on the Taylor–Maccoll solution.

This module converts the Taylor–Maccoll velocity solution into
engineering quantities:

    • Shock angle
    • Surface velocity
    • Surface Mach number
    • Surface pressure ratio
    • Pressure coefficient
    • Pressure drag coefficient

Author : Divyanshu
Project: Allen–Eggers Reentry Analysis
"""

import numpy as np

from constants import GAMMA
from taylor_maccoll import (
    find_shock_angle,
    integrate_cone,
)

from shock_relations import (
    normal_mach,
    stagnation_pressure_ratio,
)
# ==========================================================
# Surface velocity
# ==========================================================

def surface_velocity(M1, theta_c_deg):
    """
    Returns the nondimensional velocity components
    at the cone surface.

    Parameters
    ----------
    M1 : float
        Freestream Mach number.

    theta_c_deg : float
        Cone half-angle (deg).

    Returns
    -------
    Vr : float

    Vtheta : float

    V : float
        Magnitude of nondimensional velocity.
    """

    beta_deg = find_shock_angle(
        mach=M1,
        theta_c_deg=theta_c_deg
    )

    beta = np.deg2rad(beta_deg)

    sol = integrate_cone(beta, M1)

    Vr = sol.y[0, -1]
    Vtheta = sol.y[1, -1]

    V = np.sqrt(Vr**2 + Vtheta**2)

    return Vr, Vtheta, V


# ==========================================================
# Velocity → Mach number
# ==========================================================

def velocity_to_mach(V):
    """
    Converts nondimensional Taylor–Maccoll velocity
    into local Mach number.

    Parameters
    ----------
    V : float

    Returns
    -------
    M : float
    """

    return np.sqrt(
        (2.0 * V**2) /
        ((GAMMA - 1.0) * (1.0 - V**2))
    )


# ==========================================================
# Surface Mach number
# ==========================================================

def surface_mach(M1, theta_c_deg):
    """
    Local Mach number at cone surface.
    """

    _, _, V = surface_velocity(
        M1,
        theta_c_deg
    )

    return velocity_to_mach(V)


# ==========================================================
# Pressure ratio
# ==========================================================

# ==========================================================
# Cone-surface pressure ratio
# ==========================================================

def pressure_ratio(M1, theta_c_deg):
    """
    Compute the cone-wall static-pressure ratio

        pw / p_inf

    including:

    1. Total-pressure loss across the conical shock
       using the normal component of freestream Mach number.

    2. Isentropic compression from immediately behind the
       shock to the cone surface through the Taylor-Maccoll
       flow field.

    Parameters
    ----------
    M1 : float
        Freestream Mach number.

    theta_c_deg : float
        Cone semi-vertex angle (half-angle), degrees.

    Returns
    -------
    float
        Wall-to-freestream static-pressure ratio, pw/p_inf.
    """

    gamma = GAMMA

    # Shock angle corresponding to the specified cone
    beta_deg = find_shock_angle(
        mach=M1,
        theta_c_deg=theta_c_deg
    )

    beta = np.deg2rad(beta_deg)

    # Freestream Mach component normal to the shock
    M1n = normal_mach(M1, beta)

    # Total-pressure loss across the shock
    p02_p01 = stagnation_pressure_ratio(M1n)

    # Mach number at the cone surface
    Mw = surface_mach(M1, theta_c_deg)

    # Static-to-stagnation-pressure conversion:
    #
    # pw/p_inf =
    # (p02/p01) *
    # [(1 + (gamma-1)/2 M1²) /
    #  (1 + (gamma-1)/2 Mw²)]^(gamma/(gamma-1))

    numerator = (
        1.0
        + 0.5 * (gamma - 1.0) * M1**2
    )

    denominator = (
        1.0
        + 0.5 * (gamma - 1.0) * Mw**2
    )

    return (
        p02_p01
        * (numerator / denominator)
        ** (gamma / (gamma - 1.0))
    )
# ==========================================================
# Pressure coefficient
# ==========================================================

def pressure_coefficient(M1, theta_c_deg):
    """
    Pressure coefficient.

        Cp =
        2/(γM²)
        (ps/p∞ − 1)
    """

    ratio = pressure_ratio(
        M1,
        theta_c_deg
    )

    return (
        2.0 /
        (GAMMA * M1**2)
    ) * (ratio - 1.0)


# ==========================================================
# Pressure drag coefficient
# ==========================================================

def pressure_drag(M1, theta_c_deg):

    return pressure_coefficient(
        M1,
        theta_c_deg
    )

# ==========================================================
# Complete cone-surface / boundary-layer-edge state
# ==========================================================

# ==========================================================
# Complete cone-surface / boundary-layer-edge state
# ==========================================================

def cone_surface_state(M1, theta_c_deg):
    """
    Return the inviscid state immediately behind the conical
    shock and at the cone surface.

    The cone-surface inviscid state is the boundary-layer-edge
    state used for subsequent skin-friction calculations.
    """

    gamma = GAMMA

    # ------------------------------------------------------
    # Shock angle and Taylor-Maccoll solution
    # ------------------------------------------------------

    beta_deg = find_shock_angle(
        mach=M1,
        theta_c_deg=theta_c_deg,
    )

    beta = np.deg2rad(beta_deg)

    sol = integrate_cone(
        beta,
        M1,
    )

    # Taylor-Maccoll velocity immediately behind the shock
    Vr2 = sol.y[0, 0]
    Vtheta2 = sol.y[1, 0]

    V2_dimensionless = np.hypot(
        Vr2,
        Vtheta2,
    )

    # Convert V/Vmax to Mach number
    M2 = velocity_to_mach(
        V2_dimensionless
    )

    # Surface state from the terminal event
    if (
        sol.y_events
        and len(sol.y_events[0]) > 0
    ):
        Vrw, Vthetaw = sol.y_events[0][0]
    else:
        Vrw = sol.y[0, -1]
        Vthetaw = sol.y[1, -1]

    Vw_dimensionless = np.hypot(
        Vrw,
        Vthetaw,
    )

    Mw = velocity_to_mach(
        Vw_dimensionless
    )

    # ------------------------------------------------------
    # Normal-shock ratios immediately behind shock
    # ------------------------------------------------------

    M1n = M1 * np.sin(beta)

    p2_pinf = (
        1.0
        + 2.0 * gamma
        / (gamma + 1.0)
        * (M1n**2 - 1.0)
    )

    rho2_rhoinf = (
        (gamma + 1.0) * M1n**2
        / (
            (gamma - 1.0) * M1n**2
            + 2.0
        )
    )

    T2_Tinf = (
        p2_pinf
        / rho2_rhoinf
    )

    # ------------------------------------------------------
    # Isentropic turning from shock to cone surface
    # ------------------------------------------------------

    stagnation_factor_2 = (
        1.0
        + 0.5
        * (gamma - 1.0)
        * M2**2
    )

    stagnation_factor_w = (
        1.0
        + 0.5
        * (gamma - 1.0)
        * Mw**2
    )

    Tw_T2 = (
        stagnation_factor_2
        / stagnation_factor_w
    )

    pw_p2 = Tw_T2 ** (
        gamma
        / (gamma - 1.0)
    )

    rhow_rho2 = Tw_T2 ** (
        1.0
        / (gamma - 1.0)
    )

    # ------------------------------------------------------
    # Surface ratios relative to freestream
    # ------------------------------------------------------

    pw_pinf = (
        p2_pinf
        * pw_p2
    )

    rhow_rhoinf = (
        rho2_rhoinf
        * rhow_rho2
    )

    Tw_Tinf = (
        T2_Tinf
        * Tw_T2
    )

    Cp = (
        2.0
        / (gamma * M1**2)
        * (pw_pinf - 1.0)
    )

    return {
        "beta_deg": beta_deg,

        "Vr2": Vr2,
        "Vtheta2": Vtheta2,
        "M2": M2,

        "Vrw": Vrw,
        "Vthetaw": Vthetaw,
        "Mw": Mw,

        "p2_pinf": p2_pinf,
        "rho2_rhoinf": rho2_rhoinf,
        "T2_Tinf": T2_Tinf,

        "pw_pinf": pw_pinf,
        "rhow_rhoinf": rhow_rhoinf,
        "Tw_Tinf": Tw_Tinf,

        "Cp": Cp,
        "Cd_pressure": Cp,
    }
# ==========================================================
# Verification
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CONE FLOW MODULE VERIFICATION")
    print("=" * 60)

    M = 5.0
    theta = 20.0

    print()
    print(f"Mach number          : {M:.2f}")
    print(f"Cone half-angle      : {theta:.2f} deg")

    print()
    print("Calculating complete cone-surface state...")

    state = cone_surface_state(
        M,
        theta,
    )

    print()
    print("Complete surface state")
    print("----------------------")
    print(f"Shock angle beta      = {state['beta_deg']:.8f} deg")
    print(f"Mach behind shock     = {state['M2']:.8f}")
    print(f"Surface Mach          = {state['Mw']:.8f}")

    print()
    print(f"p2/p_inf              = {state['p2_pinf']:.8f}")
    print(f"rho2/rho_inf          = {state['rho2_rhoinf']:.8f}")
    print(f"T2/T_inf              = {state['T2_Tinf']:.8f}")

    print()
    print(f"pw/p_inf              = {state['pw_pinf']:.8f}")
    print(f"rhow/rho_inf          = {state['rhow_rhoinf']:.8f}")
    print(f"Tw/T_inf              = {state['Tw_Tinf']:.8f}")
    print(f"Cp                    = {state['Cp']:.8f}")
    print(f"Pressure Cd           = {state['Cd_pressure']:.8f}")
