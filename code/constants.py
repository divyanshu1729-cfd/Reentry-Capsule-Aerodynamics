"""
===========================================================
Project  : Re-entry Vehicle Analysis
Reference: NACA Report 1381 (Allen & Eggers)

File:
    constants.py

Purpose:
    Defines the physical and atmospheric constants used
    throughout the analytical implementation.

Author:
    Divyanshu
===========================================================
"""

import numpy as np

# =========================================================
# Standard Atmosphere (Imperial Units)
# =========================================================

# Sea-level density (slug/ft^3)
RHO0 = 0.0034

# Exponential atmosphere constant (1/ft)
BETA = 1.0 / 22000.0

# =========================================================
# Gravity
# =========================================================

# Gravitational acceleration (ft/s^2)
G = 32.174

# =========================================================
# Thermodynamic Properties of Air
# =========================================================

# Ratio of specific heats
GAMMA = 1.4

# Gas constant for air
# Units: ft·lbf/(slug·°R)
R = 1716.0

# Specific heat at constant pressure
CP = GAMMA * R / (GAMMA - 1.0)

# =========================================================
# Entry Velocities (NACA Report 1381)
# =========================================================

ENTRY_VELOCITIES = np.array([
    10000.0,
    20000.0,
    30000.0
])

# =========================================================
# Entry Angles
# =========================================================

# Vertical entry
THETA_VERTICAL = np.deg2rad(90.0)

# Standard entry used in heating analysis
THETA_STANDARD = np.deg2rad(30.0)
