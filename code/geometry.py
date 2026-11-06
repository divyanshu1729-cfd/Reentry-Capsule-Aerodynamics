"""
============================================================
Project : Re-entry Vehicle Analysis
Reference : NACA Report 1381

File:
    geometry.py

Purpose
-------
Geometric utilities for conical re-entry vehicles.

Supports
--------
• Fixed-base-area class
• Fixed-volume class
• Ballistic coefficient

Cone angles are specified as FULL cone angles
(as used in NACA Report 1381).

Author
------
Divyanshu
OpenAI ChatGPT
============================================================
"""

import numpy as np


# ==========================================================
# Fixed Base Area Geometry
# ==========================================================

def base_radius_from_area(A):
    """
    Base radius from base area.

    Parameters
    ----------
    A : float
        Base area (ft²)

    Returns
    -------
    float
        Base radius (ft)
    """
    return np.sqrt(A / np.pi)


def cone_height(Rb, cone_angle_deg):
    """
    Cone height.

    Parameters
    ----------
    Rb : float
        Base radius (ft)

    cone_angle_deg : float
        Full cone angle (degrees)

    Returns
    -------
    float
        Cone height (ft)
    """
    theta = np.deg2rad(cone_angle_deg / 2.0)

    return Rb / np.tan(theta)


def slant_height(Rb, cone_angle_deg):
    """
    Slant height.
    """

    theta = np.deg2rad(cone_angle_deg / 2.0)

    return Rb / np.sin(theta)


def lateral_area(Rb, cone_angle_deg):
    """
    Lateral surface area.
    """

    return np.pi * Rb * slant_height(Rb, cone_angle_deg)


def cone_volume(Rb, cone_angle_deg):
    """
    Cone volume.
    """

    H = cone_height(Rb, cone_angle_deg)

    return np.pi * Rb**2 * H / 3.0


# ==========================================================
# Fixed Volume Geometry
# ==========================================================

def base_radius_from_volume(V, cone_angle_deg):
    """
    Base radius for a fixed-volume cone.

    Parameters
    ----------
    V : float
        Cone volume (ft³)

    cone_angle_deg : float
        Full cone angle (degrees)

    Returns
    -------
    float
        Base radius (ft)
    """

    theta = np.deg2rad(cone_angle_deg / 2.0)

    return ((3.0 * V * np.tan(theta)) / np.pi) ** (1.0 / 3.0)


# ==========================================================
# Ballistic Coefficient
# ==========================================================

def ballistic_coefficient(m, Cd, A):
    """
    Ballistic coefficient.

    Parameters
    ----------
    m : float
        Mass (slug)

    Cd : float
        Drag coefficient

    A : float
        Reference area (ft²)

    Returns
    -------
    float
        m/(Cd*A)
    """

    return m / (Cd * A)


# ==========================================================
# Module Verification
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GEOMETRY MODULE TEST")
    print("=" * 60)

    # ------------------------------------------------------
    # Fixed Base Area
    # ------------------------------------------------------

    A = 10.0

    print("\nFIXED BASE AREA = 10 ft²")
    print("-" * 60)

    for angle in [10, 20, 30, 40, 60, 80, 100]:

        R = base_radius_from_area(A)
        H = cone_height(R, angle)
        L = slant_height(R, angle)
        S = lateral_area(R, angle)
        V = cone_volume(R, angle)

        print(f"\nCone Angle = {angle:3d}°")
        print(f"Base Radius   = {R:8.4f} ft")
        print(f"Height        = {H:8.4f} ft")
        print(f"Slant Height  = {L:8.4f} ft")
        print(f"Surface Area  = {S:8.4f} ft²")
        print(f"Volume        = {V:8.4f} ft³")

    # ------------------------------------------------------
    # Fixed Volume
    # ------------------------------------------------------

    V = 16.34

    print("\n")
    print("=" * 60)
    print("FIXED VOLUME = 16.34 ft³")
    print("-" * 60)

    for angle in [10, 20, 30, 40, 60, 80, 100]:

        R = base_radius_from_volume(V, angle)
        H = cone_height(R, angle)
        L = slant_height(R, angle)
        A = np.pi * R**2
        S = lateral_area(R, angle)

        print(f"\nCone Angle = {angle:3d}°")
        print(f"Base Radius   = {R:8.4f} ft")
        print(f"Height        = {H:8.4f} ft")
        print(f"Slant Height  = {L:8.4f} ft")
        print(f"Base Area     = {A:8.4f} ft²")
        print(f"Surface Area  = {S:8.4f} ft²")

    # ------------------------------------------------------
    # Cross-check
    # ------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CROSS-CHECK")
    print("=" * 60)

    angle = 40.0

    R = base_radius_from_area(10.0)
    Vcalc = cone_volume(R, angle)

    print(f"\nCone Angle = {angle:.0f}°")
    print(f"Volume for Fixed-Area Class = {Vcalc:.4f} ft³")
    print("Reference Fixed-Volume Class = 16.3400 ft³")
    print(f"Difference = {abs(Vcalc - 16.34):.6f} ft³")
