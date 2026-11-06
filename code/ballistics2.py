"""
===========================================================
Project  : Re-entry Vehicle Analysis
Reference: NACA Report 1381 (Allen & Eggers)

File:
    ballistics.py

Purpose:
    Implements the analytical and numerical ballistic
    equations used throughout the report.

Contents
--------
1. Atmospheric density
2. Approximate analytical solution (Eq. 9)
3. Approximate deceleration (Eq. 10)
4. Complete equation of motion (Eq. 3)
5. Numerical trajectory solver

Author:
    Divyanshu

Acknowledgement:
    Developed with technical assistance from OpenAI ChatGPT.
===========================================================
"""

import numpy as np
from scipy.integrate import solve_ivp

from constants import (
    RHO0,
    BETA,
    G
)

# ==========================================================
# Atmospheric Density
# Equation (2)
# ==========================================================

def density(y):
    """
    Atmospheric density.

    Parameters
    ----------
    y : float or ndarray
        Altitude (ft)

    Returns
    -------
    rho : float or ndarray
        Density (slug/ft³)
    """

    return RHO0 * np.exp(-BETA * y)


# ==========================================================
# Equation (9)
# Approximate Velocity (gravity neglected)
# ==========================================================

def velocity_approx(y, Ve, Cd, A, m):
    """
    Equation (9)

    Parameters
    ----------
    y  : altitude (ft)
    Ve : entry velocity (ft/s)
    Cd : drag coefficient
    A  : reference area (ft²)
    m  : mass (slug)

    Returns
    -------
    Velocity (ft/s)
    """

    K = Cd * RHO0 * A / (2.0 * BETA * m)

    return Ve * np.exp(-K * np.exp(-BETA * y))


# ==========================================================
# Equation (10)
# Approximate Deceleration (gravity neglected)
# ==========================================================

def deceleration_approx(y, Ve, Cd, A, m):
    """
    Returns

        -(dV/dt)/g

    dimensionless (g units)
    """

    rho = density(y)

    V = velocity_approx(y, Ve, Cd, A, m)

    decel = Cd * rho * A * V**2 / (2.0 * m)

    return decel / G


# ==========================================================
# Equation (3)
# Complete Equation of Motion
# ==========================================================

def ballistic_ode(t, state, Cd, A, m):
    """
    State vector

        state[0] = altitude (ft)

        state[1] = velocity (ft/s)
    """

    y = state[0]
    V = state[1]

    rho = density(y)

    drag = Cd * rho * A * V**2 / (2.0 * m)

    dydt = -V

    dVdt = G - drag

    return [dydt, dVdt]


# ==========================================================
# Stop Integration at Ground
# ==========================================================

def ground_event(t, state, Cd, A, m):

    return state[0]

ground_event.terminal = True
ground_event.direction = -1


# ==========================================================
# Numerical Solver
# ==========================================================

def solve_trajectory(
        Ve,
        y0=280000.0,
        Cd=0.92,
        A=np.pi/4.0,
        m=7.9742,
        tmax=250.0):
    """
    Solves Equation (3)

    Parameters
    ----------
    Ve : float
        Entry velocity (ft/s)

    Returns
    -------
    scipy solution object
    """

    initial_state = [y0, Ve]

    solution = solve_ivp(
        ballistic_ode,
        (0.0, tmax),
        initial_state,
        args=(Cd, A, m),
        events=ground_event,
        dense_output=True,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.25
    )

    return solution


# ==========================================================
# Numerical Deceleration
# ==========================================================

def numerical_deceleration(y, V, Cd, A, m):
    """
    Computes

        -(dV/dt)/g

    from the numerical solution.
    """

    rho = density(y)

    drag = Cd * rho * A * V**2 / (2.0 * m)

    dVdt = G - drag

    return -dVdt / G
