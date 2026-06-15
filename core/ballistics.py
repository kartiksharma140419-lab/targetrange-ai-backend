"""
Ballistic Mathematical Engine — TargetRange-AI.

Public API
----------
calculate_zeroing_vectors(shot_group, target_center, target_distance_meters)
    Top-level zeroing function.  Takes raw detection dicts from the vision
    pipeline, computes MPI, derives physical deviations in centimetres, and
    outputs exact integer scope-click corrections with directional strings.

Legacy helpers (used by main.py pipeline)
------------------------------------------
compute_mpi                — centroid of (x, y) pixel tuples
compute_scaling_factor     — inches-per-pixel from a known bullseye diameter
compute_linear_deviations  — pixel offsets → physical inches
compute_scope_corrections  — inches → MOA click counts with directions

Physical constants embedded in calculate_zeroing_vectors
---------------------------------------------------------
  Canvas scale       : 50.0 cm / 640 px   (50 × 50 cm backing card on 640 × 640 frame)
  Click value @ 100m : 0.7275 cm / click  (¼-MOA tactical scope standard)
  Click scaling      : linear with target_distance_meters / 100.0
"""

from __future__ import annotations

import math
from typing import Sequence


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Physical width/height of the target backing card in centimetres.
_CANVAS_SIZE_CM: float = 50.0

#: Width/height of the inference frame in pixels that maps to _CANVAS_SIZE_CM.
_FRAME_SIZE_PX: float = 640.0

#: Centimetres represented by one pixel on the rigid-canvas scale.
_CM_PER_PIXEL: float = _CANVAS_SIZE_CM / _FRAME_SIZE_PX   # ≈ 0.078125 cm/px

#: Centimetres of point-of-impact shift produced by one scope click at 100 m
#: for a standard ¼-MOA tactical scope  (1 MOA ≈ 2.91 cm @ 100 m → ¼ MOA ≈ 0.7275 cm).
_CLICK_VALUE_CM_AT_100M: float = 0.7275


# ===========================================================================
# PRIMARY PUBLIC FUNCTION
# ===========================================================================

def calculate_zeroing_vectors(
    shot_group: list,
    target_center: dict,
    target_distance_meters: float = 100.0,
) -> dict:
    """
    Compute zeroing vectors from a list of detected bullet-hole coordinates.

    This is the canonical entry point for the ballistic engine.  It takes the
    raw detection dictionaries produced by ``vision.extract_bullet_coordinates``
    and returns a fully structured, self-describing correction payload.

    Algorithm
    ---------
    1. **MPI** — arithmetic mean of all ``x`` and ``y`` pixel values.
    2. **Pixel deviation** — ``target_center − MPI``.
    3. **Physical deviation (cm)** — pixel deviation × ``_CM_PER_PIXEL``
       (rigid canvas: 640 px ↔ 50 cm).
    4. **Y-axis inversion** — image coordinates increase downward; ballistic
       convention measures elevation upward.  The vertical deviation is
       negated so that a positive ``vertical_cm`` means the group is *high*.
    5. **Click conversion** — ``1 click = 0.7275 cm @ 100 m``, scaled linearly
       to ``target_distance_meters``.
    6. **Directions** — windage RIGHT/LEFT and elevation UP/DOWN reflect the
       turret adjustments a shooter must apply to move the group onto centre.

    Args:
        shot_group: List of detection dicts, each containing at minimum the
            keys ``"x"`` and ``"y"`` (pixel coordinates of the bullet-hole
            centre).  Typically the direct output of
            ``vision.extract_bullet_coordinates()``.
        target_center: Dict with keys ``"x"`` and ``"y"`` — pixel coordinates
            of the intended point of aim (bullseye centre).
        target_distance_meters: Distance from shooter to target in metres.
            Defaults to 100 m.  Must be positive.

    Returns:
        On success — a nested dict with two top-level keys:

        .. code-block:: python

            {
                "summary": {
                    "total_hits": int,               # number of detected holes
                    "mpi": {"x": float, "y": float}, # rounded pixel centroid
                    "deviation_cm": {
                        "horizontal": float,         # +ve = group is RIGHT of centre
                        "vertical":   float,         # +ve = group is HIGH (above centre)
                    },
                },
                "corrections": {
                    "elevation": {
                        "clicks":      int,          # absolute click count
                        "direction":   "UP"|"DOWN"|"NONE",
                        "instruction": str,          # human-readable turret command
                    },
                    "windage": {
                        "clicks":      int,
                        "direction":   "RIGHT"|"LEFT"|"NONE",
                        "instruction": str,
                    },
                    "combined_instruction": str,     # single-line summary for display
                },
            }

        On empty ``shot_group`` — an error dict:

        .. code-block:: python

            {"error": "No bullet holes detected — cannot compute zeroing vectors."}

    Raises:
        ValueError: If ``target_distance_meters`` is not positive, or if any
            detection dict is missing the required ``"x"`` / ``"y"`` keys.
    """
    # ------------------------------------------------------------------ guard
    if not shot_group:
        return {
            "error": (
                "No bullet holes detected — cannot compute zeroing vectors."
            )
        }

    if target_distance_meters <= 0:
        raise ValueError(
            f"target_distance_meters must be positive; got {target_distance_meters}"
        )

    for i, detection in enumerate(shot_group):
        if "x" not in detection or "y" not in detection:
            raise ValueError(
                f"Detection at index {i} is missing required keys 'x' and/or 'y'. "
                f"Got keys: {list(detection.keys())}"
            )

    # --------------------------------------------------------- Step 2 — MPI
    n: int = len(shot_group)
    mpi_x: float = sum(float(d["x"]) for d in shot_group) / n
    mpi_y: float = sum(float(d["y"]) for d in shot_group) / n

    # ----------------------------------------- Step 3 — pixel deviation
    #   deviation = target_center − MPI
    #   +horizontal_px  →  target is right of MPI  →  shots are LEFT  →  click RIGHT
    #   +vertical_px    →  target is below MPI      →  shots are HIGH  (image-space)
    cx: float = float(target_center["x"])
    cy: float = float(target_center["y"])

    dev_x_px: float = cx - mpi_x
    dev_y_px: float = cy - mpi_y

    # -------------------------------- Step 4 — convert pixels → centimetres
    dev_x_cm: float = dev_x_px * _CM_PER_PIXEL

    # ------------------------ Step 5 — invert Y for ballistic convention
    #   Image Y increases downward; ballistic elevation increases upward.
    #   Negation: positive dev_y_cm now means group is LOW (below centre in
    #   ballistic space) and the scope must be adjusted UP.
    #
    #   Derivation:
    #     dev_y_px > 0  →  target_y > mpi_y  →  MPI is ABOVE target (image)
    #                   →  shots are HIGH     →  scope DOWN
    #     After negation: dev_y_cm < 0  →  direction = DOWN  ✓
    #
    #     dev_y_px < 0  →  target_y < mpi_y  →  MPI is BELOW target (image)
    #                   →  shots are LOW      →  scope UP
    #     After negation: dev_y_cm > 0  →  direction = UP    ✓
    dev_y_cm: float = -dev_y_px * _CM_PER_PIXEL

    # -------------------- Step 6 — centimetres → scope clicks
    #   1 click = 0.7275 cm @ 100 m, scaled linearly to actual distance.
    click_value_cm: float = _CLICK_VALUE_CM_AT_100M * (target_distance_meters / 100.0)

    raw_clicks_x: float = dev_x_cm / click_value_cm
    raw_clicks_y: float = dev_y_cm / click_value_cm

    # Integer corrections (absolute counts)
    clicks_windage: int = abs(round(raw_clicks_x))
    clicks_elevation: int = abs(round(raw_clicks_y))

    # ----------- Step 7 — directional strings
    #   Windage:   positive dev_x_cm  →  shots LEFT of centre  →  click RIGHT
    #   Elevation: positive dev_y_cm  →  shots LOW             →  click UP
    if round(raw_clicks_x) > 0:
        windage_direction: str = "RIGHT"
    elif round(raw_clicks_x) < 0:
        windage_direction = "LEFT"
    else:
        windage_direction = "NONE"

    if round(raw_clicks_y) > 0:
        elevation_direction: str = "UP"
    elif round(raw_clicks_y) < 0:
        elevation_direction = "DOWN"
    else:
        elevation_direction = "NONE"

    # ------------------------------------ human-readable instruction strings
    if windage_direction == "NONE":
        windage_instruction = "Windage is on target — no adjustment required."
    else:
        windage_instruction = (
            f"Adjust windage {clicks_windage} click{'s' if clicks_windage != 1 else ''} "
            f"{windage_direction}."
        )

    if elevation_direction == "NONE":
        elevation_instruction = "Elevation is on target — no adjustment required."
    else:
        elevation_instruction = (
            f"Adjust elevation {clicks_elevation} "
            f"click{'s' if clicks_elevation != 1 else ''} {elevation_direction}."
        )

    if windage_direction == "NONE" and elevation_direction == "NONE":
        combined = "Group is centred — no scope adjustments required."
    elif windage_direction == "NONE":
        combined = elevation_instruction
    elif elevation_direction == "NONE":
        combined = windage_instruction
    else:
        combined = (
            f"Adjust {clicks_windage} click{'s' if clicks_windage != 1 else ''} "
            f"{windage_direction} and {clicks_elevation} "
            f"click{'s' if clicks_elevation != 1 else ''} {elevation_direction}."
        )

    # ----------------------------------- Step 8 — structured return payload
    return {
        "summary": {
            "total_hits": n,
            "mpi": {
                "x": round(mpi_x, 2),
                "y": round(mpi_y, 2),
            },
            "deviation_cm": {
                "horizontal": round(dev_x_cm, 2),
                "vertical":   round(dev_y_cm, 2),
            },
        },
        "corrections": {
            "elevation": {
                "clicks":      clicks_elevation,
                "direction":   elevation_direction,
                "instruction": elevation_instruction,
            },
            "windage": {
                "clicks":      clicks_windage,
                "direction":   windage_direction,
                "instruction": windage_instruction,
            },
            "combined_instruction": combined,
        },
    }


# ===========================================================================
# LEGACY HELPERS  (used by main.py's _run_pipeline; kept for backwards compat)
# ===========================================================================

def compute_mpi(shots: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """
    Return the arithmetic centroid of all detected bullet-hole pixel positions.

    Args:
        shots: Sequence of ``(x, y)`` pixel coordinate pairs.

    Returns:
        ``(x_mpi, y_mpi)`` — average pixel coordinates.
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

    ``SF = known_bullseye_diameter_inches / bullseye_pixel_diameter``

    Args:
        known_bullseye_diameter_inches: Real-world diameter of the detected
            bullseye ring in inches.
        bullseye_pixel_diameter: Pixel diameter of the same ring as found by
            the vision pipeline.

    Returns:
        Inches-per-pixel scaling factor.

    Raises:
        ValueError: If ``bullseye_pixel_diameter`` is zero or negative.
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
    Convert pixel offsets between MPI and target centre to physical inches.

    Y-axis convention:
        Pixel Y increases downward; target elevation increases upward.
        ``delta_y_inches = (y_center − y_mpi) × SF`` so that a shot placed
        above centre (``y_mpi < y_center`` in pixel space) yields a positive
        elevation reading.

    Args:
        x_mpi: MPI x-coordinate in pixels.
        y_mpi: MPI y-coordinate in pixels.
        x_center: Bullseye centre x-coordinate in pixels.
        y_center: Bullseye centre y-coordinate in pixels.
        scaling_factor: Inches per pixel.

    Returns:
        ``(delta_x_inches, delta_y_inches)`` where:
            ``delta_x_inches > 0`` — MPI is to the right of centre.
            ``delta_y_inches > 0`` — MPI is above centre.
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

    MOA reference: 1 MOA ≈ 1.047 inches at 100 yards.  This implementation
    uses the industry-standard simplification of exactly 1 inch per 100 yards
    so that click counts match the labels printed on most commercial turrets.

    Formula::

        MOA_factor       = distance_yards / 100.0
        required_moa_x   = delta_x_inches / MOA_factor
        required_moa_y   = delta_y_inches / MOA_factor
        clicks_windage   = round(required_moa_x / click_value_moa)
        clicks_elevation = round(required_moa_y / click_value_moa)

    Direction convention:
        Windage   — positive clicks → RIGHT,  negative → LEFT.
        Elevation — positive clicks → UP,     negative → DOWN.
        Zero clicks → "NONE".

    Args:
        delta_x_inches: Horizontal MPI deviation from centre (inches).
        delta_y_inches: Vertical MPI deviation from centre (inches).
        distance_yards: Shooting distance in yards.
        click_value_moa: MOA value per scope click (e.g. 0.25 for ¼-MOA).

    Returns:
        ``{"elevation": {"clicks": int, "direction": str},
           "windage":   {"clicks": int, "direction": str}}``

    Raises:
        ValueError: If ``distance_yards`` or ``click_value_moa`` is not positive.
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

    def _windage_dir(c: int) -> str:
        return "RIGHT" if c > 0 else "LEFT" if c < 0 else "NONE"

    def _elevation_dir(c: int) -> str:
        return "UP" if c > 0 else "DOWN" if c < 0 else "NONE"

    return {
        "elevation": {
            "clicks":    clicks_elevation,
            "direction": _elevation_dir(clicks_elevation),
        },
        "windage": {
            "clicks":    clicks_windage,
            "direction": _windage_dir(clicks_windage),
        },
    }
