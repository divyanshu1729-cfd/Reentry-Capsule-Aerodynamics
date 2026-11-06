"""
=============================================================
Taylor-Maccoll Solver for Conical Supersonic Flow
=============================================================

Project
-------
Re-entry Vehicle Analysis
(NACA Report 1381)

Purpose
-------
Computes inviscid conical flow over a circular cone using
the Taylor-Maccoll equations.

All cone angles are SEMI-VERTEX ANGLES
(half-apex angles).

References
----------
Taylor (1933)
Maccoll (1937)
Kopal (1947)
Anderson:
Modern Compressible Flow,
Chapter 10

Author
------
Divyanshu
OpenAI ChatGPT
=============================================================
"""

import numpy as np

from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ==========================================================
# Gas properties
# ==========================================================

GAMMA = 1.4


# ==========================================================
# Oblique Shock Relations
# ==========================================================

def theta_beta_mach(beta, M1, gamma=GAMMA):
    """
    Theta-Beta-Mach relation.

    Parameters
    ----------
    beta : float
        Shock angle (rad)

    M1 : float
        Freestream Mach number

    Returns
    -------
    theta : float
        Flow deflection angle (rad)
    """

    numerator = (
        2.0
        / np.tan(beta)
        * (M1**2 * np.sin(beta)**2 - 1.0)
    )

    denominator = (
        M1**2
        * (gamma + np.cos(2.0 * beta))
        + 2.0
    )

    return np.arctan(numerator / denominator)


# ==========================================================
# Normal Shock
# ==========================================================

def normal_mach(M1, beta):

    return M1 * np.sin(beta)


def downstream_normal_mach(Mn1, gamma=GAMMA):

    return np.sqrt(
        (1.0 + 0.5 * (gamma - 1.0) * Mn1**2)
        /
        (gamma * Mn1**2 - 0.5 * (gamma - 1.0))
    )


def downstream_mach(M1, beta, theta, gamma=GAMMA):

    Mn1 = normal_mach(M1, beta)

    Mn2 = downstream_normal_mach(Mn1, gamma)

    return Mn2 / np.sin(beta - theta)

# ==========================================================
# Velocity Immediately Behind Shock
# ==========================================================

def initial_velocity(beta, M1, gamma=GAMMA):
    """
    Vr and Vtheta immediately behind the shock.

    Returned in nondimensional form
    V / Vmax.
    """

    theta = theta_beta_mach(beta, M1, gamma)

    M2 = downstream_mach(M1, beta, theta, gamma)

    q = np.sqrt(
        (gamma - 1.0)
        /
        (
            2.0
            +
            (gamma - 1.0) * M2**2
        )
    )

    V = M2 * q

    Vr = V * np.cos(beta - theta)

    Vtheta = -V * np.sin(beta - theta)

    return Vr, Vtheta

# ==========================================================
# Taylor-Maccoll Equation
# ==========================================================

def taylor_maccoll(theta, state, gamma=GAMMA):

    Vr, Vtheta = state

    A = (
        (gamma - 1.0)
        / 2.0
    ) * (
        1.0
        - Vr**2
        - Vtheta**2
    )

    dVr = Vtheta

    numerator = (
        Vtheta**2 * Vr
        -
        A
        * (
            2.0 * Vr
            +
            Vtheta / np.tan(theta)
        )
    )

    denominator = (
        A
        -
        Vtheta**2
    )

    dVtheta = numerator / denominator

    return [dVr, dVtheta]

# ==========================================================
# Integrate Between Shock and Cone
# ==========================================================
# ==========================================================
# Cone Surface Event
# ==========================================================

def cone_surface(theta, state):
    """
    Stop integration when Vtheta = 0,
    corresponding to the cone surface.
    """

    Vr, Vtheta = state

    return Vtheta


cone_surface.terminal = True
cone_surface.direction = 1
# ==========================================================
# Integrate Between Shock and Cone
# ==========================================================

def integrate_cone(beta, M1):

    theta0 = beta

    Vr0, Vtheta0 = initial_velocity(beta, M1)

    solution = solve_ivp(
        taylor_maccoll,
        (theta0, 1.0e-4),
        [Vr0, Vtheta0],
        events=cone_surface,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        max_step=np.deg2rad(0.1)
    )

    return solution
# ==========================================================
# Module Verification
# ==========================================================
# ==========================================================
# Cone Angle Residual
# ==========================================================
from scipy.optimize import brentq

def find_shock_angle(mach, theta_c_deg):
    """
    Determine the attached shock half-angle corresponding to
    a specified cone half-angle.

    Parameters
    ----------
    mach : float
        Freestream Mach number.

    theta_c_deg : float
        Cone half-angle (degrees).

    Returns
    -------
    beta_deg : float
        Shock half-angle (degrees).
    """

    theta_target = np.deg2rad(theta_c_deg)

    mu = np.arcsin(1.0 / mach)

    beta_values = np.linspace(
        mu + 1.0e-6,
        np.deg2rad(89.5),
        250,
    )

    beta_left = None
    f_left = None

    for beta in beta_values:

        sol = integrate_cone(beta, mach)

        # Integration failed to reach cone surface
        if len(sol.t_events[0]) == 0:
            continue

        theta_surface = sol.t_events[0][0]

        f = theta_surface - theta_target

        if beta_left is None:
            beta_left = beta
            f_left = f
            continue

        # Root bracket located
        if f * f_left < 0:

            beta_root = brentq(
                lambda b:
                    integrate_cone(b, mach).t_events[0][0]
                    - theta_target,
                beta_left,
                beta,
                xtol=1e-8,
                rtol=1e-8
            )

            return np.rad2deg(beta_root)

        beta_left = beta
        f_left = f

    raise RuntimeError(
        f"Could not bracket a shock angle for θc={theta_c_deg:.2f}°"
    )
    
    def residual(beta):

        sol = integrate_cone(beta, mach)

        if len(sol.t_events[0]) == 0:
            return 1.0

        theta_surface = sol.t_events[0][0]

        return theta_surface - theta_target

    beta = brentq(
        residual,
        mu + np.deg2rad(0.05),
        np.deg2rad(60.0),
        xtol=1e-8,
        rtol=1e-8
    )

    return np.rad2deg(beta)
    
def cone_angle_residual(beta, M1, theta_target):

    sol = integrate_cone(beta, M1)

    if len(sol.t_events[0]) == 0:
        return np.nan

    theta_computed = sol.t_events[0][0]

    return theta_computed - theta_target

# ==========================================================
# Shock Angle Solver
# ==========================================================


    
if __name__ == "__main__":

    print("="*60)
    print("TAYLOR-MACCOLL MODULE VERIFICATION")
    print("="*60)

    M1 = 5.0
    beta_deg = 30.0

    beta = np.deg2rad(beta_deg)

    theta = theta_beta_mach(beta, M1)

    print(f"\nFreestream Mach = {M1:.2f}")
    print(f"Shock angle     = {beta_deg:.2f} deg")
    print(f"Flow deflection = {np.rad2deg(theta):.4f} deg")

    Vr, Vtheta = initial_velocity(beta, M1)

    print("\nInitial conditions behind shock")
    print(f"Vr     = {Vr:.6f}")
    print(f"Vtheta = {Vtheta:.6f}")

    sol = integrate_cone(beta, M1)

    print("\nIntegration")

    print(sol.message)

    if len(sol.t_events[0]) > 0:

        cone_angle = np.rad2deg(sol.t_events[0][0])

        print(f"\nCone surface located at {cone_angle:.4f} deg")

    else:

        print("\nCone surface not detected.")

        print(f"Steps  : {len(sol.t)}")

        print("\nFinal state")

        print(f"Vr     = {sol.y[0,-1]:.6f}")
        print(f"Vtheta = {sol.y[1,-1]:.6f}")

    print()

    print("="*60)
    print("ROOT FINDER TEST")
    print("="*60)

    theta_c = 20.0

    beta = find_shock_angle(
        mach=5.0,
        theta_c_deg=theta_c
    )

    print(f"Cone half-angle  = {theta_c:.2f} deg")
    print(f"Shock half-angle = {beta:.4f} deg")
