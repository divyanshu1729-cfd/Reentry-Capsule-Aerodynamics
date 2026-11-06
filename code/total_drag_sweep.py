"""
Total-drag sweep for the two NACA Report 1381 cone families.

Computes:
    Cd_pressure
    Cd_friction
    Cd_total = Cd_pressure + Cd_friction

Families:
    1. Fixed base area: 10 ft^2
    2. Fixed volume: 16.34 ft^3

The aerodynamic state depends only on Mach number and cone
half-angle, so it is calculated once and reused for both families.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np

from cone_flow import cone_surface_state
from cone_geometry import (
    fixed_base_area_geometry,
    fixed_volume_geometry,
)
from entry_conditions import (
    entrance_condition,
    report_angle_to_half_angle,
)
from skin_friction import (
    cone_skin_friction_drag,
    exponential_density,
)


# ==========================================================
# Report conditions
# ==========================================================

ENTRANCE_VELOCITIES_FT_S = (
    10_000.0,
    20_000.0,
    30_000.0,
)

ALTITUDE_FT = 40.0 * 5280.0
T_INF_R = 500.0

FIXED_BASE_AREA_FT2 = 10.0
FIXED_VOLUME_FT3 = 16.34


# ==========================================================
# Aerodynamic-state cache
# ==========================================================

def build_surface_state_cache(
    report_angles_deg,
):
    """
    Calculate each Taylor-Maccoll state only once.

    Returns
    -------
    dictionary keyed by:
        (entrance_velocity_ft_s, report_angle_deg)
    """

    cache = {}

    for velocity in ENTRANCE_VELOCITIES_FT_S:

        entrance = entrance_condition(
            velocity_ft_s=velocity,
            temperature_R=T_INF_R,
        )

        for report_angle in report_angles_deg:

            half_angle = report_angle_to_half_angle(
                report_angle
            )

            print(
                f"Calculating V={velocity:,.0f} ft/s, "
                f"report angle={report_angle:.2f} deg, "
                f"half-angle={half_angle:.2f} deg"
            )

            try:
                state = cone_surface_state(
                    entrance.mach,
                    half_angle,
                )

            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
            ) as error:

                print(f"  Failed: {error}")
                state = None

            cache[(velocity, report_angle)] = state

    return cache


# ==========================================================
# One geometry-family sweep
# ==========================================================

def sweep_family(
    family_name,
    report_angles_deg,
    state_cache,
):
    """
    Compute pressure, friction, and total drag for one family.
    """

    rho_inf = exponential_density(
        ALTITUDE_FT
    )

    results = {
        velocity: {
            "report_angle_deg": [],
            "half_angle_deg": [],
            "cd_pressure": [],
            "cd_friction": [],
            "cd_total": [],
            "friction_fraction": [],
            "reynolds_number": [],
        }
        for velocity in ENTRANCE_VELOCITIES_FT_S
    }

    for velocity in ENTRANCE_VELOCITIES_FT_S:

        for report_angle in report_angles_deg:

            half_angle = report_angle_to_half_angle(
                report_angle
            )

            state = state_cache[
                (velocity, report_angle)
            ]

            if state is None:
                continue

            if family_name == "fixed_area":

                geometry = fixed_base_area_geometry(
                    half_angle_deg=half_angle,
                    base_area_ft2=FIXED_BASE_AREA_FT2,
                )

            elif family_name == "fixed_volume":

                geometry = fixed_volume_geometry(
                    half_angle_deg=half_angle,
                    volume_ft3=FIXED_VOLUME_FT3,
                )

            else:
                raise ValueError(
                    f"Unknown geometry family: {family_name}"
                )

            friction = cone_skin_friction_drag(
                geometry=geometry,
                freestream_density_slug_ft3=rho_inf,
                freestream_temperature_R=T_INF_R,
                surface_state=state,
            )

            cd_pressure = state["Cd_pressure"]
            cd_friction = friction.cd_friction
            cd_total = cd_pressure + cd_friction

            friction_fraction = (
                cd_friction / cd_total
                if cd_total > 0.0
                else np.nan
            )

            family_result = results[velocity]

            family_result["report_angle_deg"].append(
                report_angle
            )
            family_result["half_angle_deg"].append(
                half_angle
            )
            family_result["cd_pressure"].append(
                cd_pressure
            )
            family_result["cd_friction"].append(
                cd_friction
            )
            family_result["cd_total"].append(
                cd_total
            )
            family_result["friction_fraction"].append(
                friction_fraction
            )
            family_result["reynolds_number"].append(
                friction.reynolds_number
            )

    return results


# ==========================================================
# CSV output
# ==========================================================

def save_family_csv(
    filename,
    family_name,
    results,
):
    """Save one geometry-family sweep."""

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "family",
                "velocity_ft_s",
                "report_angle_deg",
                "half_angle_deg",
                "cd_pressure",
                "cd_friction",
                "cd_total",
                "friction_fraction",
                "reynolds_number",
            ]
        )

        for velocity, data in results.items():

            number_of_points = len(
                data["report_angle_deg"]
            )

            for index in range(number_of_points):

                writer.writerow(
                    [
                        family_name,
                        velocity,
                        data["report_angle_deg"][index],
                        data["half_angle_deg"][index],
                        data["cd_pressure"][index],
                        data["cd_friction"][index],
                        data["cd_total"][index],
                        data["friction_fraction"][index],
                        data["reynolds_number"][index],
                    ]
                )


# ==========================================================
# Plotting
# ==========================================================

def plot_total_drag(
    results,
    title,
    output_filename,
):
    """Plot total drag for all entrance velocities."""

    plot_styles = (
        {
            "linestyle": "-",
            "marker": "o",
        },
        {
            "linestyle": "--",
            "marker": "s",
        },
        {
            "linestyle": ":",
            "marker": "^",
        },
    )

    plt.figure(
        figsize=(8.0, 5.5)
    )

    for velocity, style in zip(
        ENTRANCE_VELOCITIES_FT_S,
        plot_styles,
    ):

        data = results[velocity]

        angles = np.asarray(
            data["report_angle_deg"]
        )

        cd_total = np.asarray(
            data["cd_total"]
        )

        entrance = entrance_condition(
            velocity_ft_s=velocity,
            temperature_R=T_INF_R,
        )

        plt.plot(
            angles,
            cd_total,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=4,
            linewidth=1.8,
            markersize=4.5,
            label=(
                f"$V_E={velocity:,.0f}$ ft/s "
                f"($M={entrance.mach:.2f}$)"
            ),
        )

    plt.xlabel(
        "Cone angle shown in report, degrees"
    )

    plt.ylabel(
        r"Total drag coefficient, $C_D$"
    )

    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_filename,
        dpi=300,
    )

    plt.show()


# ==========================================================
# Numerical summary
# ==========================================================

def print_summary(
    family_name,
    results,
):
    """Print values near report cone angle = 40 degrees."""

    print()
    print("=" * 72)
    print(f"{family_name.upper()} SUMMARY")
    print("=" * 72)

    for velocity in ENTRANCE_VELOCITIES_FT_S:

        data = results[velocity]

        angles = np.asarray(
            data["report_angle_deg"]
        )

        index = int(
            np.argmin(
                np.abs(angles - 40.0)
            )
        )

        print()
        print(
            f"Velocity              : "
            f"{velocity:,.0f} ft/s"
        )
        print(
            f"Report cone angle     : "
            f"{angles[index]:.2f} deg"
        )
        print(
            f"Cone half-angle       : "
            f"{data['half_angle_deg'][index]:.2f} deg"
        )
        print(
            f"Pressure drag Cd,p    : "
            f"{data['cd_pressure'][index]:.8f}"
        )
        print(
            f"Friction drag Cd,f    : "
            f"{data['cd_friction'][index]:.8f}"
        )
        print(
            f"Total drag Cd         : "
            f"{data['cd_total'][index]:.8f}"
        )
        print(
            f"Friction contribution : "
            f"{100.0 * data['friction_fraction'][index]:.4f} %"
        )
        print(
            f"Reynolds number       : "
            f"{data['reynolds_number'][index]:.8e}"
        )


# ==========================================================
# Main program
# ==========================================================

def main():

    # Begin at 4 degrees because the zero-angle limit is
    # geometrically and numerically singular.
    report_angles_deg = np.linspace(
        4.0,
        100.0,
        49,
    )

    print("=" * 72)
    print("BUILDING TAYLOR-MACCOLL SURFACE-STATE CACHE")
    print("=" * 72)

    state_cache = build_surface_state_cache(
        report_angles_deg
    )

    print()
    print("Calculating fixed-base-area family...")

    fixed_area_results = sweep_family(
        family_name="fixed_area",
        report_angles_deg=report_angles_deg,
        state_cache=state_cache,
    )

    print("Calculating fixed-volume family...")

    fixed_volume_results = sweep_family(
        family_name="fixed_volume",
        report_angles_deg=report_angles_deg,
        state_cache=state_cache,
    )

    save_family_csv(
        filename="total_drag_fixed_area.csv",
        family_name="fixed_area",
        results=fixed_area_results,
    )

    save_family_csv(
        filename="total_drag_fixed_volume.csv",
        family_name="fixed_volume",
        results=fixed_volume_results,
    )

    print_summary(
        family_name="fixed area",
        results=fixed_area_results,
    )

    print_summary(
        family_name="fixed volume",
        results=fixed_volume_results,
    )

    plot_total_drag(
        results=fixed_area_results,
        title=(
            "Total cone drag — fixed base area "
            r"$A_b=10\ \mathrm{ft^2}$"
        ),
        output_filename=(
            "total_drag_fixed_area.png"
        ),
    )

    plot_total_drag(
        results=fixed_volume_results,
        title=(
            "Total cone drag — fixed volume "
            r"$V=16.34\ \mathrm{ft^3}$"
        ),
        output_filename=(
            "total_drag_fixed_volume.png"
        ),
    )

    print()
    print("Saved:")
    print("  total_drag_fixed_area.csv")
    print("  total_drag_fixed_volume.csv")
    print("  total_drag_fixed_area.png")
    print("  total_drag_fixed_volume.png")


if __name__ == "__main__":
    main()
