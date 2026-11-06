"""
Baseline reconstruction of NACA Report 1381 Figures 5 and 6.

Figure 5:
    Fixed cone base area, Ab = 10 ft^2

Figure 6:
    Fixed cone volume, V = 16.34 ft^3

The horizontal axis is the report cone angle. The Taylor-Maccoll
solver receives the corresponding cone semi-vertex angle:

    theta_c = theta_report / 2

This is Baseline Model A:
    - Taylor-Maccoll pressure drag
    - Current compressible turbulent skin-friction correlation
    - Allen-Eggers total heat-input relation

The script writes each completed case immediately to CSV, allowing
an interrupted calculation to resume.
"""


import csv
import time
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import impact_heat
from cone_geometry import (
    fixed_base_area_geometry,
    fixed_volume_geometry,
)
from entry_conditions import report_angle_to_half_angle


# ==========================================================
# Calculation configuration
# ==========================================================

ENTRANCE_VELOCITIES_FT_S = (
    10_000.0,
    20_000.0,
    30_000.0,
)

WEIGHT_CASES = (
    ("0", 0.0),
    ("1000", 1_000.0),
    ("5000", 5_000.0),
    ("10000", 10_000.0),
    ("inf", np.inf),
)

# Twenty-five points:
# 4, 8, 12, ..., 100 degrees.
#
# This is adequate for the first complete reconstruction.
REPORT_ANGLES_DEG = np.linspace(
    4.0,
    100.0,
    25,
)

ENTRY_ANGLE_DEG = 30.0
T_INF_R = 500.0

FIXED_BASE_AREA_FT2 = 10.0
FIXED_VOLUME_FT3 = 16.34

# Set True only when deliberately discarding previous checkpoints.
RESET_OUTPUT = False

OUTPUT_FILES = {
    "fixed_area": {
        "csv": Path("figure5_fixed_area.csv"),
        "png": Path("figure5_fixed_area.png"),
    },
    "fixed_volume": {
        "csv": Path("figure6_fixed_volume.csv"),
        "png": Path("figure6_fixed_volume.png"),
    },
}


# ==========================================================
# Taylor-Maccoll cache
# ==========================================================

# impact_heat repeatedly requests identical surface states:
# entry states and, for B > 1, the common VE*exp(-1/2) peak state.
# Caching considerably reduces runtime.

_original_cone_surface_state = impact_heat.cone_surface_state


@lru_cache(maxsize=16_384)
def cached_cone_surface_state(
    mach: float,
    half_angle_deg: float,
):
    return _original_cone_surface_state(
        float(mach),
        float(half_angle_deg),
    )


impact_heat.cone_surface_state = cached_cone_surface_state


# ==========================================================
# CSV structure
# ==========================================================

FIELDNAMES = (
    "case_key",
    "family",
    "velocity_ft_s",
    "report_angle_deg",
    "half_angle_deg",
    "weight_label",
    "weight_lb",
    "base_area_ft2",
    "volume_ft3",
    "length_ft",
    "lateral_area_ft2",
    "mass_slug",
    "cd_pressure",
    "cd_friction",
    "cd_total",
    "ballistic_parameter_B",
    "peak_altitude_ft",
    "peak_velocity_ft_s",
    "peak_mach",
    "impact_velocity_ft_s",
    "reynolds_number",
    "cf_equivalent",
    "friction_heating_ratio",
    "total_heat_ft_lbf",
    "total_heat_1e8_ft_lbf",
    "iterations",
    "status",
    "error",
)


def make_case_key(
    family: str,
    velocity_ft_s: float,
    report_angle_deg: float,
    weight_label: str,
) -> str:
    """Generate a stable checkpoint key."""

    return (
        f"{family}|"
        f"{velocity_ft_s:.0f}|"
        f"{report_angle_deg:.6f}|"
        f"{weight_label}"
    )


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load existing checkpoint rows."""

    if not csv_path.exists():
        return []

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        return list(csv.DictReader(csv_file))


def append_row(
    csv_path: Path,
    row: dict,
) -> None:
    """Append one completed or failed case immediately."""

    write_header = not csv_path.exists()

    with csv_path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        if write_header:
            writer.writeheader()

        writer.writerow(row)


# ==========================================================
# Geometry
# ==========================================================

def geometry_for_family(
    family: str,
    half_angle_deg: float,
):
    """Construct the selected NACA cone family."""

    if family == "fixed_area":
        return fixed_base_area_geometry(
            half_angle_deg=half_angle_deg,
            base_area_ft2=FIXED_BASE_AREA_FT2,
        )

    if family == "fixed_volume":
        return fixed_volume_geometry(
            half_angle_deg=half_angle_deg,
            volume_ft3=FIXED_VOLUME_FT3,
        )

    raise ValueError(
        f"Unknown geometry family: {family}"
    )


# ==========================================================
# Case evaluation
# ==========================================================

def blank_result_fields() -> dict:
    """Fields used when a calculation fails."""

    return {
        "mass_slug": np.nan,
        "cd_pressure": np.nan,
        "cd_friction": np.nan,
        "cd_total": np.nan,
        "ballistic_parameter_B": np.nan,
        "peak_altitude_ft": np.nan,
        "peak_velocity_ft_s": np.nan,
        "peak_mach": np.nan,
        "impact_velocity_ft_s": np.nan,
        "reynolds_number": np.nan,
        "cf_equivalent": np.nan,
        "friction_heating_ratio": np.nan,
        "total_heat_ft_lbf": np.nan,
        "total_heat_1e8_ft_lbf": np.nan,
        "iterations": 0,
    }


def evaluate_case(
    family: str,
    velocity_ft_s: float,
    report_angle_deg: float,
    weight_label: str,
    weight_lb: float,
) -> dict:
    """Evaluate one geometry/velocity/weight combination."""

    half_angle_deg = report_angle_to_half_angle(
        report_angle_deg
    )

    geometry = geometry_for_family(
        family,
        half_angle_deg,
    )

    row = {
        "case_key": make_case_key(
            family,
            velocity_ft_s,
            report_angle_deg,
            weight_label,
        ),
        "family": family,
        "velocity_ft_s": velocity_ft_s,
        "report_angle_deg": report_angle_deg,
        "half_angle_deg": half_angle_deg,
        "weight_label": weight_label,
        "weight_lb": weight_lb,
        "base_area_ft2": geometry.base_area_ft2,
        "volume_ft3": geometry.volume_ft3,
        "length_ft": geometry.length_ft,
        "lateral_area_ft2": (
            geometry.lateral_area_ft2
        ),
    }

    row.update(blank_result_fields())

    try:

        # --------------------------------------------------
        # Zero-weight limiting curve
        # --------------------------------------------------

        if weight_label == "0":

            row.update(
                {
                    "mass_slug": 0.0,
                    "ballistic_parameter_B": np.inf,
                    "total_heat_ft_lbf": 0.0,
                    "total_heat_1e8_ft_lbf": 0.0,
                    "iterations": 0,
                }
            )

        # --------------------------------------------------
        # Infinite-weight limiting curve
        # --------------------------------------------------

        elif weight_label == "inf":

            result = impact_heat.infinite_weight_heat_input(
                geometry=geometry,
                entrance_velocity_ft_s=velocity_ft_s,
                entry_angle_deg=ENTRY_ANGLE_DEG,
                temperature_R=T_INF_R,
            )

            row.update(
                {
                    "mass_slug": result.mass_slug,
                    "cd_pressure": result.cd_pressure,
                    "cd_friction": result.cd_friction,
                    "cd_total": result.cd_total,
                    "ballistic_parameter_B": (
                        result.ballistic_parameter_B
                    ),
                    "peak_altitude_ft": (
                        result.peak_altitude_ft
                    ),
                    "peak_velocity_ft_s": (
                        result.peak_velocity_ft_s
                    ),
                    "peak_mach": result.peak_mach,
                    "impact_velocity_ft_s": (
                        result.impact_velocity_ft_s
                    ),
                    "reynolds_number": (
                        result.reynolds_number
                    ),
                    "cf_equivalent": (
                        result.cf_equivalent
                    ),
                    "friction_heating_ratio": (
                        result.friction_heating_ratio
                    ),
                    "total_heat_ft_lbf": (
                        result.total_heat_ft_lbf
                    ),
                    "total_heat_1e8_ft_lbf": (
                        result.total_heat_ft_lbf / 1.0e8
                    ),
                    "iterations": result.iterations,
                }
            )

        # --------------------------------------------------
        # Finite-weight curves
        # --------------------------------------------------

        else:

            result = impact_heat.finite_weight_heat_input(
                geometry=geometry,
                entrance_velocity_ft_s=velocity_ft_s,
                weight_lb=weight_lb,
                entry_angle_deg=ENTRY_ANGLE_DEG,
                temperature_R=T_INF_R,

                # Sufficient for baseline figure generation.
                tolerance=1.0e-5,
                max_iterations=30,
            )

            row.update(
                {
                    "mass_slug": result.mass_slug,
                    "cd_pressure": result.cd_pressure,
                    "cd_friction": result.cd_friction,
                    "cd_total": result.cd_total,
                    "ballistic_parameter_B": (
                        result.ballistic_parameter_B
                    ),
                    "peak_altitude_ft": (
                        result.peak_altitude_ft
                    ),
                    "peak_velocity_ft_s": (
                        result.peak_velocity_ft_s
                    ),
                    "peak_mach": result.peak_mach,
                    "impact_velocity_ft_s": (
                        result.impact_velocity_ft_s
                    ),
                    "reynolds_number": (
                        result.reynolds_number
                    ),
                    "cf_equivalent": (
                        result.cf_equivalent
                    ),
                    "friction_heating_ratio": (
                        result.friction_heating_ratio
                    ),
                    "total_heat_ft_lbf": (
                        result.total_heat_ft_lbf
                    ),
                    "total_heat_1e8_ft_lbf": (
                        result.total_heat_ft_lbf / 1.0e8
                    ),
                    "iterations": result.iterations,
                }
            )

        row["status"] = "ok"
        row["error"] = ""

    except Exception as error:

        row["status"] = "failed"
        row["error"] = (
            f"{type(error).__name__}: {error}"
        )

    return row


# ==========================================================
# Family calculation with resume support
# ==========================================================

def calculate_family(
    family: str,
) -> list[dict[str, str]]:
    """Calculate or resume one complete report figure."""

    csv_path = OUTPUT_FILES[family]["csv"]

    if RESET_OUTPUT and csv_path.exists():
        csv_path.unlink()

    existing_rows = load_rows(csv_path)

    completed_keys = {
        row["case_key"]
        for row in existing_rows
        if row.get("status") == "ok"
    }

    total_cases = (
        len(ENTRANCE_VELOCITIES_FT_S)
        * len(REPORT_ANGLES_DEG)
        * len(WEIGHT_CASES)
    )

    completed_count = len(completed_keys)

    print()
    print("=" * 78)
    print(f"CALCULATING {family.upper()}")
    print("=" * 78)
    print(
        f"Previously completed cases: "
        f"{completed_count}/{total_cases}"
    )

    start_time = time.time()

    for velocity in ENTRANCE_VELOCITIES_FT_S:

        for report_angle in REPORT_ANGLES_DEG:

            for weight_label, weight_lb in WEIGHT_CASES:

                key = make_case_key(
                    family,
                    velocity,
                    report_angle,
                    weight_label,
                )

                if key in completed_keys:
                    continue

                case_number = completed_count + 1

                print(
                    f"[{case_number:03d}/{total_cases}] "
                    f"{family}, "
                    f"VE={velocity:,.0f} ft/s, "
                    f"angle={report_angle:.1f} deg, "
                    f"W={weight_label} lb"
                )

                case_start = time.time()

                row = evaluate_case(
                    family=family,
                    velocity_ft_s=velocity,
                    report_angle_deg=report_angle,
                    weight_label=weight_label,
                    weight_lb=weight_lb,
                )

                append_row(
                    csv_path,
                    row,
                )

                if row["status"] == "ok":
                    completed_keys.add(key)
                    completed_count += 1

                    print(
                        "    "
                        f"Q={float(row['total_heat_1e8_ft_lbf']):.6f}"
                        " x10^8 ft-lbf, "
                        f"time={time.time() - case_start:.2f} s"
                    )

                else:
                    print(
                        f"    FAILED: {row['error']}"
                    )

    elapsed = time.time() - start_time

    print()
    print(
        f"{family} calculation elapsed time: "
        f"{elapsed / 60.0:.2f} min"
    )

    return load_rows(csv_path)


# ==========================================================
# Plotting
# ==========================================================

def weight_legend_label(
    weight_label: str,
) -> str:
    """Display label for one missile-weight curve."""

    labels = {
        "0": r"$W=0$ lb",
        "1000": r"$W=1{,}000$ lb",
        "5000": r"$W=5{,}000$ lb",
        "10000": r"$W=10{,}000$ lb",
        "inf": r"$W=\infty$",
    }

    return labels[weight_label]


def plot_family(
    family: str,
    rows: list[dict[str, str]],
) -> None:
    """Generate a three-panel Figure 5 or Figure 6."""

    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    styles = {
        "0": {
            "linestyle": "-",
            "marker": None,
        },
        "1000": {
            "linestyle": "-",
            "marker": "o",
        },
        "5000": {
            "linestyle": "--",
            "marker": "s",
        },
        "10000": {
            "linestyle": "-.",
            "marker": "^",
        },
        "inf": {
            "linestyle": ":",
            "marker": "D",
        },
    }

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(8.5, 11.0),
        sharex=True,
    )

    for panel_index, (
        axis,
        velocity,
    ) in enumerate(
        zip(
            axes,
            ENTRANCE_VELOCITIES_FT_S,
        )
    ):

        for weight_label, _ in WEIGHT_CASES:

            selected = [
                row
                for row in valid_rows
                if (
                    float(row["velocity_ft_s"])
                    == velocity
                    and row["weight_label"]
                    == weight_label
                )
            ]

            selected.sort(
                key=lambda row: float(
                    row["report_angle_deg"]
                )
            )

            if not selected:
                continue

            angles = np.asarray(
                [
                    float(row["report_angle_deg"])
                    for row in selected
                ]
            )

            heat = np.asarray(
                [
                    float(
                        row[
                            "total_heat_1e8_ft_lbf"
                        ]
                    )
                    for row in selected
                ]
            )

            style = styles[weight_label]

            axis.plot(
                angles,
                heat,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=3,
                linewidth=1.7,
                markersize=4.0,
                label=weight_legend_label(
                    weight_label
                ),
            )

        panel_letter = chr(
            ord("a") + panel_index
        )

        axis.set_title(
            f"({panel_letter}) "
            f"$V_E={velocity:,.0f}$ ft/s"
        )

        axis.set_ylabel(
            r"$Q\;(10^8\ \mathrm{ft\cdot lbf})$"
        )

        axis.set_xlim(
            0.0,
            100.0,
        )

        axis.set_ylim(
            bottom=0.0,
        )

        axis.grid(True)

    axes[-1].set_xlabel(
        "Cone angle shown in report, degrees"
    )

    axes[0].legend(
        loc="best",
        ncol=2,
    )

    if family == "fixed_area":

        figure.suptitle(
            "Baseline reconstruction of NACA Report 1381 Figure 5\n"
            r"Fixed base area: $A_b=10\ \mathrm{ft^2}$"
        )

    else:

        figure.suptitle(
            "Baseline reconstruction of NACA Report 1381 Figure 6\n"
            r"Fixed volume: $V=16.34\ \mathrm{ft^3}$"
        )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )

    png_path = OUTPUT_FILES[family]["png"]

    figure.savefig(
        png_path,
        dpi=300,
    )

    plt.close(figure)

    print(f"Saved plot: {png_path}")


# ==========================================================
# Summary
# ==========================================================

def print_optimum_summary(
    family: str,
    rows: list[dict[str, str]],
) -> None:
    """Print maximum baseline heat and corresponding angle."""

    valid_rows = [
        row
        for row in rows
        if row.get("status") == "ok"
    ]

    print()
    print("=" * 78)
    print(f"{family.upper()} OPTIMUM-ANGLE SUMMARY")
    print("=" * 78)

    for velocity in ENTRANCE_VELOCITIES_FT_S:

        print()
        print(
            f"Entrance velocity: "
            f"{velocity:,.0f} ft/s"
        )

        for weight_label, _ in WEIGHT_CASES:

            selected = [
                row
                for row in valid_rows
                if (
                    float(row["velocity_ft_s"])
                    == velocity
                    and row["weight_label"]
                    == weight_label
                )
            ]

            if not selected:
                print(
                    f"  W={weight_label:>5}: "
                    "no valid points"
                )
                continue

            optimum = max(
                selected,
                key=lambda row: float(
                    row["total_heat_ft_lbf"]
                ),
            )

            print(
                f"  W={weight_label:>5} lb: "
                f"Qmax="
                f"{float(optimum['total_heat_1e8_ft_lbf']):.6f}"
                f" x10^8 ft-lbf at "
                f"{float(optimum['report_angle_deg']):.1f} deg"
            )


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    print("=" * 78)
    print("NACA REPORT 1381 FIGURES 5 AND 6 — BASELINE MODEL A")
    print("=" * 78)
    print(
        f"Report-angle points per velocity: "
        f"{len(REPORT_ANGLES_DEG)}"
    )
    print(
        "Checkpointing is enabled; completed cases "
        "will be skipped on rerun."
    )

    fixed_area_rows = calculate_family(
        "fixed_area"
    )

    plot_family(
        "fixed_area",
        fixed_area_rows,
    )

    print_optimum_summary(
        "fixed_area",
        fixed_area_rows,
    )

    fixed_volume_rows = calculate_family(
        "fixed_volume"
    )

    plot_family(
        "fixed_volume",
        fixed_volume_rows,
    )

    print_optimum_summary(
        "fixed_volume",
        fixed_volume_rows,
    )

    print()
    print("=" * 78)
    print("CALCULATION COMPLETE")
    print("=" * 78)

    print("Generated:")
    print("  figure5_fixed_area.csv")
    print("  figure5_fixed_area.png")
    print("  figure6_fixed_volume.csv")
    print("  figure6_fixed_volume.png")

    print()
    print(
        "Taylor-Maccoll cache statistics: "
        f"{cached_cone_surface_state.cache_info()}"
    )


if __name__ == "__main__":
    main()
