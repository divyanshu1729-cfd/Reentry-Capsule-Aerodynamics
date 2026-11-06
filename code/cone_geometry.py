"""
Geometry utilities for the NACA Report 1381 conical-missile families.

Two families are required:

1. Fixed base area:
       A_b = 10 ft^2

2. Fixed volume:
       V = 16.34 ft^3

The aerodynamic solver uses the cone semi-vertex angle
(cone half-angle), measured between the axis and cone surface.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConeGeometry:
    """Geometric properties of a sharp circular cone."""

    half_angle_deg: float
    base_radius_ft: float
    base_area_ft2: float
    length_ft: float
    lateral_area_ft2: float
    volume_ft3: float


def _validate_half_angle(half_angle_deg: float) -> float:
    """
    Validate a conventional sharp-cone semi-vertex angle.

    The limiting values 0 and 90 degrees are singular:
    - theta -> 0 gives an infinitely long cone;
    - theta -> 90 gives zero cone length.
    """

    angle = float(half_angle_deg)

    if not np.isfinite(angle):
        raise ValueError("Cone half-angle must be finite.")

    if not 0.0 < angle < 90.0:
        raise ValueError(
            "Cone half-angle must satisfy 0 < theta_c < 90 degrees."
        )

    return angle


def geometry_from_radius(
    half_angle_deg: float,
    base_radius_ft: float,
) -> ConeGeometry:
    """
    Construct cone geometry from half-angle and base radius.
    """

    angle = _validate_half_angle(half_angle_deg)
    radius = float(base_radius_ft)

    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Base radius must be positive and finite.")

    theta = np.deg2rad(angle)

    base_area = np.pi * radius**2
    length = radius / np.tan(theta)
    slant_length = radius / np.sin(theta)
    lateral_area = np.pi * radius * slant_length
    volume = base_area * length / 3.0

    return ConeGeometry(
        half_angle_deg=angle,
        base_radius_ft=radius,
        base_area_ft2=base_area,
        length_ft=length,
        lateral_area_ft2=lateral_area,
        volume_ft3=volume,
    )


def fixed_base_area_geometry(
    half_angle_deg: float,
    base_area_ft2: float = 10.0,
) -> ConeGeometry:
    """
    Geometry for the Figure 5 fixed-base-area family.

    Parameters
    ----------
    half_angle_deg
        Cone semi-vertex angle, degrees.

    base_area_ft2
        Fixed frontal/base area. NACA Report 1381 uses 10 ft^2.
    """

    area = float(base_area_ft2)

    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Base area must be positive and finite.")

    radius = np.sqrt(area / np.pi)

    return geometry_from_radius(
        half_angle_deg=half_angle_deg,
        base_radius_ft=radius,
    )


def fixed_volume_geometry(
    half_angle_deg: float,
    volume_ft3: float = 16.34,
) -> ConeGeometry:
    """
    Geometry for the Figure 6 fixed-volume family.

    From

        V = pi*r_b^3 / (3*tan(theta_c)),

    the base radius is

        r_b = [3*V*tan(theta_c)/pi]^(1/3).
    """

    angle = _validate_half_angle(half_angle_deg)
    volume = float(volume_ft3)

    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("Volume must be positive and finite.")

    theta = np.deg2rad(angle)

    radius = np.cbrt(
        3.0 * volume * np.tan(theta) / np.pi
    )

    geometry = geometry_from_radius(
        half_angle_deg=angle,
        base_radius_ft=radius,
    )

    # Guard against accidental algebra or unit errors.
    if not np.isclose(
        geometry.volume_ft3,
        volume,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise RuntimeError("Fixed-volume geometry check failed.")

    return geometry


def print_geometry(
    title: str,
    geometry: ConeGeometry,
) -> None:
    """Print one geometry in a readable form."""

    print(title)
    print("-" * len(title))
    print(
        f"Cone half-angle : "
        f"{geometry.half_angle_deg:.6f} deg"
    )
    print(
        f"Base radius     : "
        f"{geometry.base_radius_ft:.6f} ft"
    )
    print(
        f"Base area       : "
        f"{geometry.base_area_ft2:.6f} ft^2"
    )
    print(
        f"Axial length    : "
        f"{geometry.length_ft:.6f} ft"
    )
    print(
        f"Lateral area    : "
        f"{geometry.lateral_area_ft2:.6f} ft^2"
    )
    print(
        f"Volume          : "
        f"{geometry.volume_ft3:.6f} ft^3"
    )


if __name__ == "__main__":

    TEST_ANGLE_DEG = 20.0

    fixed_area = fixed_base_area_geometry(
        half_angle_deg=TEST_ANGLE_DEG,
        base_area_ft2=10.0,
    )

    fixed_volume = fixed_volume_geometry(
        half_angle_deg=TEST_ANGLE_DEG,
        volume_ft3=16.34,
    )

    print("=" * 60)
    print("NACA 1381 CONE-GEOMETRY VERIFICATION")
    print("=" * 60)
    print()

    print_geometry(
        "Fixed base area: A_b = 10 ft^2",
        fixed_area,
    )

    print()

    print_geometry(
        "Fixed volume: V = 16.34 ft^3",
        fixed_volume,
    )
