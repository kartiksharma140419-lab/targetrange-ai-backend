"""
Ballistic Mathematical Engine for TargetRange-AI.

Computes Mean Point of Impact (MPI), spatial scaling, linear deviations,
and scope correction clicks from detected bullet-hole pixel coordinates.
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_mpi(shots: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """
    Return the mathematical centroid of all detected bullet holes.

    Args:
        shots: Sequence of (x, y) pixel coordinates for each hole center.

    Returns:
        (x_mpi, y_mpi) — average pixel coordinates.
    """
    n = len(shots)
    x_mpi = sum(s[0] for s in shots) / n
    y_mpi = sum(s[1] for s in shots) / n
    return x_mpi, y_mpi


def compute_scaling_factor(
    known_bullseye_diameter_inches: float,
    bullseye_pixel_diameter: float,
) -> float:
    """
    Derive the spatial scaling factor: physical inches per image pixel.

    SF = known_bullseye_diameter_inches / bullseye_pixel_diameter

    Args:
        known_bullseye_diameter_inches: Real-world diameter of the detected
            bullseye ring in inches.
        bullseye_pixel_diameter: Pixel diameter of the same ring as found by
            the vision pipeline.

    Returns:
        Inches-per-pixel scaling factor (float).

    Raises:
        ValueError: If bullseye_pixel_diameter is zero or negative.
    """
    if bullseye_pixel_diameter <= 0:
        raise ValueError(
            f"bullseye_pixel_diameter must be positive; got {bullseye_pixel_diameter}"
        )
    return known_bullseye_diameter_inches / bullseye_pixel_diameter


def compute_linear_deviations(
    x_mpi: float,
    y_mpi: float,
    x_center: float,
    y_center: float,
    scaling_factor: float,
) -> tuple[float, float]:
    """
    Convert pixel offsets between MPI and target center to physical inches.

    Note on Y-axis convention:
        Pixel Y increases downward; target elevation increases upward.
        Delta_Y_inches is therefore (y_center - y_mpi) * SF so that a shot
        above center (y_mpi < y_center in pixel space) yields a positive
        (high) elevation reading.

    Args:
        x_mpi: MPI x-coordinate in pixels.
        y_mpi: MPI y-coordinate in pixels.
        x_center: Bullseye center x-coordinate in pixels.
        y_center: Bullseye center y-coordinate in pixels.
        scaling_factor: Inches per pixel.

    Returns:
        (delta_x_inches, delta_y_inches):
            delta_x_inches > 0 → MPI is to the right of center.
            delta_y_inches > 0 → MPI is above center.
    """
    delta_x_inches = (x_mpi - x_center) * scaling_factor
    delta_y_inches = (y_center - y_mpi) * scaling_factor
    return delta_x_inches, delta_y_inches


def compute_scope_corrections(
    delta_x_inches: float,
    delta_y_inches: float,
    distance_yards: float,
    click_value_moa: float,
) -> dict:
    """
    Translate physical deviations into scope adjustment clicks.

    MOA Reference: 1 MOA ≈ 1.047 inches at 100 yards.  For small-arms
    zeroing, the industry convention simplifies this to exactly 1 inch per
    100 yards, which is what most scope manufacturers use for click-value
    labelling.  This implementation follows that convention so that the
    output matches what the shooter reads on their turrets.

        MOA_Factor      = distance_yards / 100.0
        Required_MOA_X  = delta_x_inches / MOA_Factor
        Required_MOA_Y  = delta_y_inches / MOA_Factor
        Clicks_Windage  = round(Required_MOA_X / click_value_moa)
        Clicks_Elevation= round(Required_MOA_Y / click_value_moa)

    Direction convention:
        Windage  — positive clicks → RIGHT, negative → LEFT.
        Elevation— positive clicks → UP,   negative → DOWN.
        Zero clicks → "NONE".

    Args:
        delta_x_inches: Horizontal deviation of MPI from center (inches).
        delta_y_inches: Vertical deviation of MPI from center (inches).
        distance_yards: Shooting distance in yards.
        click_value_moa: MOA value per scope click (e.g. 0.25 for ¼-MOA).

    Returns:
        dict with keys:
            "elevation": {"clicks": int, "direction": "UP"|"DOWN"|"NONE"}
            "windage":   {"clicks": int, "direction": "RIGHT"|"LEFT"|"NONE"}

    Raises:
        ValueError: If distance_yards or click_value_moa is non-positive.
    """
    if distance_yards <= 0:
        raise ValueError(f"distance_yards must be positive; got {distance_yards}")
    if click_value_moa <= 0:
        raise ValueError(f"click_value_moa must be positive; got {click_value_moa}")

    moa_factor = distance_yards / 100.0

    required_moa_x = delta_x_inches / moa_factor
    required_moa_y = delta_y_inches / moa_factor

    clicks_windage = round(required_moa_x / click_value_moa)
    clicks_elevation = round(required_moa_y / click_value_moa)

    def _windage_direction(clicks: int) -> str:
        if clicks > 0:
            return "RIGHT"
        if clicks < 0:
            return "LEFT"
        return "NONE"

    def _elevation_direction(clicks: int) -> str:
        if clicks > 0:
            return "UP"
        if clicks < 0:
            return "DOWN"
        return "NONE"

    return {
        "elevation": {
            "clicks": clicks_elevation,
            "direction": _elevation_direction(clicks_elevation),
        },
        "windage": {
            "clicks": clicks_windage,
            "direction": _windage_direction(clicks_windage),
        },
    }
