"""
===========================================================
Project  : Re-entry Vehicle Analysis
Reference: NACA Report 1381

File:
    atmosphere.py

Purpose:
    Implements the exponential atmospheric density model
    used throughout Allen & Eggers (1958).

    rho = rho0 * exp(-beta*y)

Author:
    Divyanshu
===========================================================
"""

import numpy as np

from constants import RHO0
from constants import BETA


def density(altitude):
    """
    Atmospheric density.

    Parameters
    ----------
    altitude : float or ndarray
        Altitude above sea level (ft)

    Returns
    -------
    rho : float or ndarray
        Density (slug/ft^3)
    """

    return RHO0 * np.exp(-BETA * altitude)
