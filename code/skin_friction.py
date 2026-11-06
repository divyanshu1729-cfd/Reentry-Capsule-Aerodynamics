"""
Approximate turbulent skin-friction drag for the NACA 1381
conical-missile calculations.

Model:
    - Exponential atmosphere
    - Sutherland viscosity law
    - Compressibility-corrected turbulent flat-plate correlation
    - Reynolds number based on cone axial length

This is a first-pass engineering model. It is not yet a literal
reproduction of references 8 and 9 cited by NACA Report 1381.
"""

from dataclasses import dataclass

import numpy as np

from cone_geometry import ConeGeometry


# ----------------------------------------------------------
# Atmospheric and gas constants
# ----------------------------------------------------------

RHO0_SLUG_FT3 = 0.0034
BETA_FT_INV = 1.0 / 22_000.0

GAMMA = 1.4
R_AIR_FTLBF_SLUG_R = 1716.0

# Sutherland constants for air in imperial units
MU_REF_SLUG_FT_S = 3.737e-7
T_REF_R = 518.67
SUTHERLAND_R = 198.72


@dataclass(frozen=True)
class SkinFrictionResult:
    density_slug_ft3: float
    viscosity_slug_ft_s: float
    reynolds_number: float
    cf_incompressible: float
    compressibility_factor: float
    cf_average: float
    cd_friction: float


# ----------------------------------------------------------
# Atmosphere
# ----------------------------------------------------------

def exponential_density(
    altitude_ft: float,
    rho0: float = RHO0_SLUG_FT3,
    beta: float = BETA_FT_INV,
) -> float:
    """
    Exponential atmospheric-density model:

        rho = rho0 exp(-beta*y)
    """

    altitude = float(altitude_ft)

    if altitude < 0.0:
        raise ValueError("Altitude cannot be negative.")

    return rho0 * np.exp(-beta * altitude)


# ----------------------------------------------------------
# Sutherland viscosity
# ----------------------------------------------------------

def sutherland_viscosity(
    temperature_R: float,
) -> float:
    """
    Dynamic viscosity of air using Sutherland's law.

        mu/mu_ref =
        (T/T_ref)^(3/2)
        (T_ref + S)/(T + S)
    """

    temperature = float(temperature_R)

    if temperature <= 0.0:
        raise ValueError("Absolute temperature must be positive.")

    temperature_ratio = temperature / T_REF_R

    return (
        MU_REF_SLUG_FT_S
        * temperature_ratio**1.5
        * (T_REF_R + SUTHERLAND_R)
        / (temperature + SUTHERLAND_R)
    )


# ----------------------------------------------------------
# Turbulent flat-plate friction correlation
# ----------------------------------------------------------

def turbulent_flat_plate_cf(
    reynolds_number: float,
    mach: float,
) -> tuple[float, float, float]:
    """
    Average turbulent flat-plate skin-friction coefficient.

    Incompressible Prandtl-Schlichting relation:

        Cf0 = 0.455 / [log10(Re)]^2.58

    Compressibility correction:

        F_M = (1 + 0.144 M^2)^(-0.65)

        Cf = Cf0 * F_M

    Returns
    -------
    cf0
        Incompressible value.

    compressibility_factor
        Multiplicative Mach correction.

    cf
        Compressibility-corrected average coefficient.
    """

    reynolds = float(reynolds_number)
    mach_number = float(mach)

    if reynolds <= 1.0:
        raise ValueError("Reynolds number must exceed one.")

    if mach_number < 0.0:
        raise ValueError("Mach number cannot be negative.")

    cf0 = 0.455 / np.log10(reynolds)**2.58

    compressibility_factor = (
        1.0 + 0.144 * mach_number**2
    )**(-0.65)

    cf = cf0 * compressibility_factor

    return cf0, compressibility_factor, cf


# ----------------------------------------------------------
# Cone skin-friction drag
# ----------------------------------------------------------

def cone_skin_friction_drag(
    geometry,
    freestream_density_slug_ft3,
    freestream_temperature_R,
    surface_state,
    gas_constant=R_AIR_FTLBF_SLUG_R,
):
    """
    Estimate turbulent skin-friction drag using the inviscid
    cone-surface state as the boundary-layer-edge state.

    Parameters
    ----------
    geometry
        ConeGeometry instance.

    freestream_density_slug_ft3
        Local atmospheric freestream density.

    freestream_temperature_R
        Local atmospheric freestream temperature.

    surface_state
        Dictionary returned by cone_surface_state().

    gas_constant
        Air gas constant in ft*lbf/(slug*deg R).
    """

    theta = np.deg2rad(
        geometry.half_angle_deg
    )

    # Boundary-layer length of run along cone surface
    slant_length = (
        geometry.base_radius_ft
        / np.sin(theta)
    )

    # Inviscid boundary-layer-edge properties
    density_edge = (
        freestream_density_slug_ft3
        * surface_state["rhow_rhoinf"]
    )

    temperature_edge = (
        freestream_temperature_R
        * surface_state["Tw_Tinf"]
    )

    mach_edge = surface_state["Mw"]

    speed_of_sound_edge = np.sqrt(
        GAMMA
        * gas_constant
        * temperature_edge
    )

    velocity_edge = (
        mach_edge
        * speed_of_sound_edge
    )

    viscosity_edge = sutherland_viscosity(
        temperature_edge
    )

    reynolds = (
        density_edge
        * velocity_edge
        * slant_length
        / viscosity_edge
    )

    cf0, mach_factor, cf_average = turbulent_flat_plate_cf(
        reynolds_number=reynolds,
        mach=mach_edge,
    )

    # Axial projection of tangential friction force
    projection_ratio = (
        geometry.lateral_area_ft2
        / geometry.base_area_ft2
        * np.cos(theta)
    )

    cd_friction = (
        cf_average
        * projection_ratio
    )

    return SkinFrictionResult(
        density_slug_ft3=density_edge,
        viscosity_slug_ft_s=viscosity_edge,
        reynolds_number=reynolds,
        cf_incompressible=cf0,
        compressibility_factor=mach_factor,
        cf_average=cf_average,
        cd_friction=cd_friction,
    )
# ----------------------------------------------------------
# Verification
# ----------------------------------------------------------

if __name__ == "__main__":

    from cone_geometry import fixed_base_area_geometry
    from entry_conditions import entrance_condition
    from cone_flow import cone_surface_state

    geometry = fixed_base_area_geometry(
        half_angle_deg=20.0,
        base_area_ft2=10.0,
    )

    entrance = entrance_condition(
        velocity_ft_s=30_000.0,
        temperature_R=500.0,
    )

    surface = cone_surface_state(
        entrance.mach,
        geometry.half_angle_deg,
    )

    altitude_ft = 40.0 * 5280.0

    rho_inf = exponential_density(
        altitude_ft
    )

    result = cone_skin_friction_drag(
        geometry=geometry,
        freestream_density_slug_ft3=rho_inf,
        freestream_temperature_R=500.0,
        surface_state=surface,
    )

    theta = np.deg2rad(
        geometry.half_angle_deg
    )

    slant_length = (
        geometry.base_radius_ft
        / np.sin(theta)
    )

    rho_edge = (
        rho_inf
        * surface["rhow_rhoinf"]
    )

    temperature_edge = (
        500.0
        * surface["Tw_Tinf"]
    )

    velocity_edge = (
        surface["Mw"]
        * np.sqrt(
            GAMMA
            * R_AIR_FTLBF_SLUG_R
            * temperature_edge
        )
    )

    print("=" * 64)
    print("CONE SKIN-FRICTION VERIFICATION")
    print("=" * 64)

    print(f"Altitude             : {40.0:.2f} miles")
    print(
        f"Freestream velocity  : "
        f"{entrance.velocity_ft_s:,.0f} ft/s"
    )
    print(
        f"Freestream Mach      : "
        f"{entrance.mach:.8f}"
    )
    print(
        f"Cone half-angle      : "
        f"{geometry.half_angle_deg:.4f} deg"
    )
    print(
        f"Cone axial length    : "
        f"{geometry.length_ft:.6f} ft"
    )
    print(
        f"Cone slant length    : "
        f"{slant_length:.6f} ft"
    )
    print(
        f"Cone lateral area    : "
        f"{geometry.lateral_area_ft2:.6f} ft^2"
    )

    print()
    print("Boundary-layer-edge state")
    print("-------------------------")
    print(
        f"Edge density         : "
        f"{rho_edge:.8e} slug/ft^3"
    )
    print(
        f"Edge temperature     : "
        f"{temperature_edge:.8f} deg R"
    )
    print(
        f"Edge Mach number     : "
        f"{surface['Mw']:.8f}"
    )
    print(
        f"Edge velocity        : "
        f"{velocity_edge:.8f} ft/s"
    )

    print()
    print("Skin-friction calculation")
    print("-------------------------")
    print(
        f"Dynamic viscosity    : "
        f"{result.viscosity_slug_ft_s:.8e} slug/(ft s)"
    )
    print(
        f"Reynolds number      : "
        f"{result.reynolds_number:.8e}"
    )
    print(
        f"Incompressible Cf    : "
        f"{result.cf_incompressible:.8e}"
    )
    print(
        f"Mach correction      : "
        f"{result.compressibility_factor:.8e}"
    )
    print(
        f"Average turbulent Cf : "
        f"{result.cf_average:.8e}"
    )
    print(
        f"Friction drag Cd,f   : "
        f"{result.cd_friction:.8e}"
    )
