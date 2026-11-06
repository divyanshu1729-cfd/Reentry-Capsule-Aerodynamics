"""
Pressure-drag sweep for the NACA Report 1381 cone families.

This stage computes only inviscid pressure drag using the
validated Taylor-Maccoll cone-flow solver.

Skin-friction drag and heat transfer are added later.
"""

import numpy as np
import matplotlib.pyplot as plt

from entry_conditions import (
    entrance_condition,
    report_angle_to_half_angle,
)
from cone_flow import pressure_drag


ENTRANCE_VELOCITIES_FT_S = (
    10_000.0,
    20_000.0,
    30_000.0,
)


def compute_pressure_drag_curve(
    entrance_velocity_ft_s,
    report_angles_deg,
):
    """
    Compute cone pressure-drag coefficient over a range of
    report cone angles.

    Parameters
    ----------
    entrance_velocity_ft_s : float
        Entry velocity in ft/s.

    report_angles_deg : array-like
        Included cone angles plotted by NACA Report 1381.

    Returns
    -------
    mach : float
        Entry Mach number.

    cd_pressure : numpy.ndarray
        Pressure-drag coefficient at each cone angle.
        Failed points are returned as NaN.
    """

    condition = entrance_condition(
        entrance_velocity_ft_s
    )

    mach = condition.mach

    cd_pressure = np.full(
        len(report_angles_deg),
        np.nan,
        dtype=float,
    )

    for index, report_angle_deg in enumerate(
        report_angles_deg
    ):
        half_angle_deg = report_angle_to_half_angle(
            report_angle_deg
        )

        try:
            cd_pressure[index] = pressure_drag(
                mach,
                half_angle_deg,
            )

        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
        ) as error:
            print(
                "Taylor-Maccoll calculation failed at "
                f"V={entrance_velocity_ft_s:,.0f} ft/s, "
                f"report angle={report_angle_deg:.2f} deg, "
                f"half-angle={half_angle_deg:.2f} deg: "
                f"{error}"
            )

    return mach, cd_pressure


def main():

    report_angles_deg = np.linspace(
        2.0,
        100.0,
        50,
    )

    print("=" * 72)
    print("NACA 1381 PRESSURE-DRAG SWEEP")
    print("=" * 72)

    plt.figure(figsize=(8.0, 5.5))

    plot_styles = (
        {"linestyle": "-",  "marker": "o"},
        {"linestyle": "--", "marker": "s"},
        {"linestyle": ":",  "marker": "^"},
    )

    for velocity, style in zip(
        ENTRANCE_VELOCITIES_FT_S,
        plot_styles,
    ):

        mach, cd_pressure = compute_pressure_drag_curve(
            entrance_velocity_ft_s=velocity,
            report_angles_deg=report_angles_deg,
        )

        valid = np.isfinite(cd_pressure)

        print()
        print(f"Velocity : {velocity:,.0f} ft/s")
        print(f"Mach     : {mach:.8f}")
        print(
            f"Valid points: "
            f"{np.count_nonzero(valid)}/"
            f"{len(report_angles_deg)}"
        )

        if np.any(valid):
            print(
                f"Cd,p range: "
                f"{np.nanmin(cd_pressure):.6f} "
                f"to {np.nanmax(cd_pressure):.6f}"
            )

        plt.plot(
            report_angles_deg[valid],
            cd_pressure[valid],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=5,
            linewidth=1.8,
            markersize=4.5,
            label=(
                f"$V_E={velocity:,.0f}$ ft/s "
                f"($M={mach:.2f}$)"
            ),
        )

    plt.xlabel("Cone angle shown in report, degrees")
    plt.ylabel(r"Pressure drag coefficient, $C_{D,p}$")
    plt.title(
        "Taylor-Maccoll pressure drag at entrance conditions"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = "pressure_drag_sweep.png"

    plt.savefig(
        output_file,
        dpi=300,
    )

    print()
    print(f"Saved plot: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
