"""
Allen-Eggers total convective heat transferred by impact.

Implements NACA Report 1381 equations:

    B = Cd*rho0*A / (beta*m*sin(theta_E))

    Q = (m*VE^2/4)
        * (Cf_equiv*S)/(Cd*A)
        * (1 - exp(-B))

The skin-friction coefficient is evaluated at the altitude
where the altitude rate of total heat input is maximal.

Units
-----
velocity     : ft/s
altitude     : ft
area         : ft^2
mass         : slug
weight       : lbf
heat         : ft*lbf
temperature  : deg R
"""

from dataclasses import dataclass

import numpy as np

from cone_flow import cone_surface_state
from cone_geometry import (
    ConeGeometry,
    fixed_base_area_geometry,
)
from entry_conditions import speed_of_sound
from skin_friction import (
    BETA_FT_INV,
    RHO0_SLUG_FT3,
    cone_skin_friction_drag,
    exponential_density,
)


# ==========================================================
# Constants
# ==========================================================

G_FT_S2 = 32.174

ENTRY_ANGLE_DEG = 30.0
T_INF_R = 500.0


# ==========================================================
# Result container
# ==========================================================

@dataclass(frozen=True)
class HeatInputResult:
    weight_lb: float
    mass_slug: float

    cd_pressure: float
    cd_friction: float
    cd_total: float

    ballistic_parameter_B: float

    peak_altitude_ft: float
    peak_velocity_ft_s: float
    peak_mach: float

    impact_velocity_ft_s: float
    reynolds_number: float
    cf_equivalent: float

    friction_heating_ratio: float
    total_heat_ft_lbf: float

    iterations: int


# ==========================================================
# Ballistic relations
# ==========================================================

def ballistic_parameter(
    cd_total: float,
    reference_area_ft2: float,
    mass_slug: float,
    entry_angle_deg: float,
    rho0: float = RHO0_SLUG_FT3,
    beta: float = BETA_FT_INV,
) -> float:
    """
    Allen-Eggers parameter:

        B = Cd*rho0*A /
            (beta*m*sin(theta_E))
    """

    theta_entry = np.deg2rad(
        abs(entry_angle_deg)
    )

    if mass_slug <= 0.0:
        raise ValueError("Mass must be positive.")

    if cd_total <= 0.0:
        raise ValueError(
            "Total drag coefficient must be positive."
        )

    if reference_area_ft2 <= 0.0:
        raise ValueError(
            "Reference area must be positive."
        )

    if np.sin(theta_entry) <= 0.0:
        raise ValueError(
            "Entry-angle magnitude must exceed zero."
        )

    return (
        cd_total
        * rho0
        * reference_area_ft2
        / (
            beta
            * mass_slug
            * np.sin(theta_entry)
        )
    )


def heating_peak_conditions(
    entrance_velocity_ft_s: float,
    B: float,
    beta: float = BETA_FT_INV,
) -> tuple[float, float]:
    """
    Altitude and velocity for maximum dQ/dy.

    For B > 1:

        y_peak = ln(B)/beta
        V_peak = VE*exp(-1/2)

    If B <= 1, the theoretical maximum lies below sea
    level, so the largest in-flight value occurs at impact:

        y_peak = 0
        V_peak = VE*exp(-B/2)
    """

    if B > 1.0:

        altitude_ft = np.log(B) / beta

        velocity_ft_s = (
            entrance_velocity_ft_s
            * np.exp(-0.5)
        )

    else:

        altitude_ft = 0.0

        velocity_ft_s = (
            entrance_velocity_ft_s
            * np.exp(-0.5 * B)
        )

    return altitude_ft, velocity_ft_s


# ==========================================================
# Friction evaluation at one trajectory state
# ==========================================================

def friction_at_state(
    geometry: ConeGeometry,
    velocity_ft_s: float,
    altitude_ft: float,
    temperature_R: float = T_INF_R,
):
    """
    Calculate the cone skin friction at a specified
    trajectory velocity and altitude.
    """

    sound_speed = speed_of_sound(
        temperature_R
    )

    mach = velocity_ft_s / sound_speed

    surface = cone_surface_state(
        mach,
        geometry.half_angle_deg,
    )

    rho_inf = exponential_density(
        altitude_ft
    )

    friction = cone_skin_friction_drag(
        geometry=geometry,
        freestream_density_slug_ft3=rho_inf,
        freestream_temperature_R=temperature_R,
        surface_state=surface,
    )

    return mach, surface, friction


# ==========================================================
# Finite missile weight
# ==========================================================

def finite_weight_heat_input(
    geometry: ConeGeometry,
    entrance_velocity_ft_s: float,
    weight_lb: float,
    entry_angle_deg: float = ENTRY_ANGLE_DEG,
    temperature_R: float = T_INF_R,
    tolerance: float = 1.0e-8,
    max_iterations: int = 50,
) -> HeatInputResult:
    """
    Calculate total heat input for a finite missile weight.

    Cd_total and skin friction are coupled because the
    heating-peak altitude depends on Cd_total, while the
    friction coefficient depends on conditions at that
    altitude. A fixed-point iteration is therefore used.
    """

    if weight_lb <= 0.0:
        raise ValueError(
            "Finite weight must be greater than zero."
        )

    mass_slug = weight_lb / G_FT_S2

    entry_mach = (
        entrance_velocity_ft_s
        / speed_of_sound(temperature_R)
    )

    entry_surface = cone_surface_state(
        entry_mach,
        geometry.half_angle_deg,
    )

    cd_pressure = entry_surface[
        "Cd_pressure"
    ]

    # Initial estimate: pressure drag only
    cd_total = cd_pressure

    iterations = 0

    for iteration in range(
        1,
        max_iterations + 1,
    ):

        iterations = iteration

        B = ballistic_parameter(
            cd_total=cd_total,
            reference_area_ft2=(
                geometry.base_area_ft2
            ),
            mass_slug=mass_slug,
            entry_angle_deg=entry_angle_deg,
        )

        peak_altitude_ft, peak_velocity_ft_s = (
            heating_peak_conditions(
                entrance_velocity_ft_s,
                B,
            )
        )

        peak_mach, surface, friction = (
            friction_at_state(
                geometry=geometry,
                velocity_ft_s=peak_velocity_ft_s,
                altitude_ft=peak_altitude_ft,
                temperature_R=temperature_R,
            )
        )

        cd_new = (
            cd_pressure
            + friction.cd_friction
        )

        relative_change = (
            abs(cd_new - cd_total)
            / max(abs(cd_new), 1.0e-12)
        )

        if relative_change < tolerance:
            cd_total = cd_new
            break

        # Relaxation improves stability for slender cones.
        cd_total = (
            0.5 * cd_total
            + 0.5 * cd_new
        )

    else:
        raise RuntimeError(
            "Heat-input drag iteration did not converge."
        )

    # Recalculate the final state using converged Cd.
    B = ballistic_parameter(
        cd_total=cd_total,
        reference_area_ft2=geometry.base_area_ft2,
        mass_slug=mass_slug,
        entry_angle_deg=entry_angle_deg,
    )

    peak_altitude_ft, peak_velocity_ft_s = (
        heating_peak_conditions(
            entrance_velocity_ft_s,
            B,
        )
    )

    peak_mach, surface, friction = (
        friction_at_state(
            geometry=geometry,
            velocity_ft_s=peak_velocity_ft_s,
            altitude_ft=peak_altitude_ft,
            temperature_R=temperature_R,
        )
    )

    cd_total = (
        cd_pressure
        + friction.cd_friction
    )

    # Integrated equivalent friction coefficient.
    #
    # This is Cf'*S, without the axial cosine projection.
    cfS = (
        friction.cf_average
        * geometry.lateral_area_ft2
    )

    drag_area = (
        cd_total
        * geometry.base_area_ft2
    )

    friction_heating_ratio = (
        cfS / drag_area
    )

    # Numerically stable evaluation of 1 - exp(-B).
    energy_loss_factor = (
        -np.expm1(-B)
    )

    total_heat = (
        0.25
        * mass_slug
        * entrance_velocity_ft_s**2
        * friction_heating_ratio
        * energy_loss_factor
    )

    impact_velocity = (
        entrance_velocity_ft_s
        * np.exp(-0.5 * B)
    )

    return HeatInputResult(
        weight_lb=weight_lb,
        mass_slug=mass_slug,

        cd_pressure=cd_pressure,
        cd_friction=friction.cd_friction,
        cd_total=cd_total,

        ballistic_parameter_B=B,

        peak_altitude_ft=peak_altitude_ft,
        peak_velocity_ft_s=peak_velocity_ft_s,
        peak_mach=peak_mach,

        impact_velocity_ft_s=impact_velocity,
        reynolds_number=friction.reynolds_number,
        cf_equivalent=friction.cf_average,

        friction_heating_ratio=(
            friction_heating_ratio
        ),

        total_heat_ft_lbf=total_heat,

        iterations=iterations,
    )


# ==========================================================
# Zero-weight limiting case
# ==========================================================

def zero_weight_heat_input() -> float:
    """
    As m approaches zero, the total heat delivered to the
    missile approaches zero.
    """

    return 0.0


# ==========================================================
# Infinite-weight limiting case
# ==========================================================

def infinite_weight_heat_input(
    geometry: ConeGeometry,
    entrance_velocity_ft_s: float,
    entry_angle_deg: float = ENTRY_ANGLE_DEG,
    temperature_R: float = T_INF_R,
) -> HeatInputResult:
    """
    Relatively-heavy-missile limit, B -> 0:

        Q_inf =
        Cf'*S*rho0*VE^2 /
        [4*beta*sin(theta_E)]

    The maximum altitude rate of heat input occurs at
    sea level in this limiting case.
    """

    entry_mach, surface, friction = (
        friction_at_state(
            geometry=geometry,
            velocity_ft_s=(
                entrance_velocity_ft_s
            ),
            altitude_ft=0.0,
            temperature_R=temperature_R,
        )
    )

    cd_pressure = surface["Cd_pressure"]

    cd_total = (
        cd_pressure
        + friction.cd_friction
    )

    cfS = (
        friction.cf_average
        * geometry.lateral_area_ft2
    )

    theta_entry = np.deg2rad(
        abs(entry_angle_deg)
    )

    total_heat = (
        cfS
        * RHO0_SLUG_FT3
        * entrance_velocity_ft_s**2
        / (
            4.0
            * BETA_FT_INV
            * np.sin(theta_entry)
        )
    )

    friction_heating_ratio = (
        cfS
        / (
            cd_total
            * geometry.base_area_ft2
        )
    )

    return HeatInputResult(
        weight_lb=np.inf,
        mass_slug=np.inf,

        cd_pressure=cd_pressure,
        cd_friction=friction.cd_friction,
        cd_total=cd_total,

        ballistic_parameter_B=0.0,

        peak_altitude_ft=0.0,
        peak_velocity_ft_s=(
            entrance_velocity_ft_s
        ),
        peak_mach=entry_mach,

        impact_velocity_ft_s=(
            entrance_velocity_ft_s
        ),
        reynolds_number=(
            friction.reynolds_number
        ),
        cf_equivalent=(
            friction.cf_average
        ),

        friction_heating_ratio=(
            friction_heating_ratio
        ),

        total_heat_ft_lbf=total_heat,

        iterations=1,
    )


# ==========================================================
# Verification case
# ==========================================================

if __name__ == "__main__":

    # Current report-angle convention:
    # report angle 40 deg -> half-angle 20 deg.
    geometry = fixed_base_area_geometry(
        half_angle_deg=20.0,
        base_area_ft2=10.0,
    )

    entrance_velocity = 30_000.0

    finite_weights = (
        1_000.0,
        5_000.0,
        10_000.0,
    )

    print("=" * 78)
    print("ALLEN-EGGERS TOTAL HEAT-INPUT VERIFICATION")
    print("=" * 78)

    print()
    print("Geometry")
    print("--------")
    print(
        f"Cone half-angle       : "
        f"{geometry.half_angle_deg:.4f} deg"
    )
    print(
        f"Reference area        : "
        f"{geometry.base_area_ft2:.6f} ft^2"
    )
    print(
        f"Lateral area          : "
        f"{geometry.lateral_area_ft2:.6f} ft^2"
    )
    print(
        f"Entrance velocity     : "
        f"{entrance_velocity:,.0f} ft/s"
    )

    print()
    print("Zero-weight limit")
    print("-----------------")
    print(
        f"Q                     : "
        f"{zero_weight_heat_input():.8e} ft-lbf"
    )

    for weight in finite_weights:

        result = finite_weight_heat_input(
            geometry=geometry,
            entrance_velocity_ft_s=(
                entrance_velocity
            ),
            weight_lb=weight,
        )

        print()
        print(f"Weight = {weight:,.0f} lb")
        print("-" * 30)

        print(
            f"Mass                  : "
            f"{result.mass_slug:.8f} slug"
        )
        print(
            f"Pressure Cd           : "
            f"{result.cd_pressure:.8f}"
        )
        print(
            f"Friction Cd           : "
            f"{result.cd_friction:.8f}"
        )
        print(
            f"Total Cd              : "
            f"{result.cd_total:.8f}"
        )
        print(
            f"Ballistic parameter B : "
            f"{result.ballistic_parameter_B:.8f}"
        )

        print(
            f"Peak altitude         : "
            f"{result.peak_altitude_ft:,.2f} ft"
        )
        print(
            f"Peak velocity         : "
            f"{result.peak_velocity_ft_s:,.2f} ft/s"
        )
        print(
            f"Peak Mach             : "
            f"{result.peak_mach:.8f}"
        )
        print(
            f"Impact velocity       : "
            f"{result.impact_velocity_ft_s:,.2f} ft/s"
        )

        print(
            f"Reynolds number       : "
            f"{result.reynolds_number:.8e}"
        )
        print(
            f"Equivalent Cf         : "
            f"{result.cf_equivalent:.8e}"
        )
        print(
            f"Cf'S/(CdA)            : "
            f"{result.friction_heating_ratio:.8e}"
        )

        print(
            f"Total heat            : "
            f"{result.total_heat_ft_lbf:.8e} ft-lbf"
        )
        print(
            f"Total heat            : "
            f"{result.total_heat_ft_lbf / 1.0e6:.6f} "
            f"million ft-lbf"
        )
        print(
            f"Iterations            : "
            f"{result.iterations}"
        )

    heavy = infinite_weight_heat_input(
        geometry=geometry,
        entrance_velocity_ft_s=entrance_velocity,
    )

    print()
    print("Infinite-weight limit")
    print("---------------------")
    print(
        f"Reynolds number       : "
        f"{heavy.reynolds_number:.8e}"
    )
    print(
        f"Equivalent Cf         : "
        f"{heavy.cf_equivalent:.8e}"
    )
    print(
        f"Total heat            : "
        f"{heavy.total_heat_ft_lbf:.8e} ft-lbf"
    )
    print(
        f"Total heat            : "
        f"{heavy.total_heat_ft_lbf / 1.0e6:.6f} "
        f"million ft-lbf"
    )
