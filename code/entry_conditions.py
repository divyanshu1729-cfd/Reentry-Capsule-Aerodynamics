"""
Entrance conditions for the NACA Report 1381 calculations.

The report uses:
    T_inf = 500 deg R
    V_E   = 10,000, 20,000, and 30,000 ft/s

The aerodynamic cone solver requires Mach number.
"""

from dataclasses import dataclass

import numpy as np


GAMMA = 1.4

# Imperial gas constant for air:
# ft*lbf/(slug*deg R)
R_AIR = 1716.0

# Report entrance temperature
T_INF_R = 500.0


@dataclass(frozen=True)
class EntranceCondition:
    velocity_ft_s: float
    temperature_R: float
    speed_of_sound_ft_s: float
    mach: float


def speed_of_sound(
    temperature_R: float,
    gamma: float = GAMMA,
    gas_constant: float = R_AIR,
) -> float:
    """
    Compute the perfect-gas speed of sound:

        a = sqrt(gamma*R*T)
    """

    temperature = float(temperature_R)

    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Absolute temperature must be positive and finite."
        )

    if gamma <= 1.0:
        raise ValueError("Gamma must be greater than one.")

    if gas_constant <= 0.0:
        raise ValueError("Gas constant must be positive.")

    return np.sqrt(
        gamma
        * gas_constant
        * temperature
    )


def entrance_condition(
    velocity_ft_s: float,
    temperature_R: float = T_INF_R,
) -> EntranceCondition:
    """
    Construct one entrance condition.
    """

    velocity = float(velocity_ft_s)

    if not np.isfinite(velocity) or velocity <= 0.0:
        raise ValueError(
            "Entrance velocity must be positive and finite."
        )

    sound_speed = speed_of_sound(temperature_R)

    return EntranceCondition(
        velocity_ft_s=velocity,
        temperature_R=temperature_R,
        speed_of_sound_ft_s=sound_speed,
        mach=velocity / sound_speed,
    )


def report_angle_to_half_angle(
    report_cone_angle_deg: float,
) -> float:
    """
    Convert the cone angle shown on the NACA 1381 plots
    to the semi-vertex angle used by the cone-flow solver.

        theta_c = cone_angle_report / 2
    """

    report_angle = float(report_cone_angle_deg)

    if not np.isfinite(report_angle):
        raise ValueError("Report cone angle must be finite.")

    if not 0.0 < report_angle < 180.0:
        raise ValueError(
            "Report cone angle must satisfy "
            "0 < angle < 180 degrees."
        )

    return 0.5 * report_angle


if __name__ == "__main__":

    entrance_velocities = (
        10_000.0,
        20_000.0,
        30_000.0,
    )

    print("=" * 60)
    print("NACA 1381 ENTRANCE CONDITIONS")
    print("=" * 60)

    for velocity in entrance_velocities:

        state = entrance_condition(velocity)

        print()
        print(
            f"Entrance velocity : "
            f"{state.velocity_ft_s:,.0f} ft/s"
        )
        print(
            f"Temperature       : "
            f"{state.temperature_R:.2f} deg R"
        )
        print(
            f"Speed of sound    : "
            f"{state.speed_of_sound_ft_s:.6f} ft/s"
        )
        print(
            f"Mach number       : "
            f"{state.mach:.8f}"
        )

    print()
    print("Angle-convention example")
    print("------------------------")

    report_angle = 40.0
    half_angle = report_angle_to_half_angle(report_angle)

    print(
        f"Report cone angle : {report_angle:.2f} deg"
    )
    print(
        f"Cone half-angle   : {half_angle:.2f} deg"
    )
