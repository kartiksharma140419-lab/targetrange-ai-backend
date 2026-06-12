"""
Overlay rendering for TargetRange-AI.

Draws bullet holes, MPI crosshair, bullseye centre, and correction arrow
onto a copy of the input image, returning a PNG-encoded byte string.
"""

from __future__ import annotations

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Palette (BGR)
# ---------------------------------------------------------------------------
_COLOR_HOLE = (0, 220, 0)        # green   — detected bullet holes
_COLOR_MPI = (0, 60, 255)        # red     — Mean Point of Impact
_COLOR_CENTER = (255, 140, 0)    # blue    — bullseye centre
_COLOR_ARROW = (0, 200, 255)     # yellow  — correction arrow
_COLOR_TEXT_BG = (20, 20, 20)    # near-black — label background
_COLOR_TEXT_FG = (255, 255, 255) # white   — label text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_overlay(
    image_bgr: np.ndarray,
    shot_centroids: list[tuple[float, float]],
    mpi: tuple[float, float],
    target_center: tuple[float, float],
    correction_label: str,
    hole_radius: int = 14,
) -> bytes:
    """
    Draw analysis results onto a copy of *image_bgr* and return PNG bytes.

    Visual elements
    ---------------
    • Green circle + dot  — each detected bullet hole.
    • Red crosshair       — Mean Point of Impact (MPI).
    • Blue filled circle  — Target / bullseye centre.
    • Yellow arrow        — MPI → target centre (direction to walk the zero).
    • Text callout        — Click summary anchored near the arrowhead.

    Args:
        image_bgr:        Original BGR image array (not modified in-place).
        shot_centroids:   List of (x, y) pixel centres for each hole.
        mpi:              (x_mpi, y_mpi) pixel coordinates.
        target_center:    (x_center, y_center) pixel coordinates.
        correction_label: Human-readable summary string, e.g.
                          "↑ 4 UP  ← 2 LEFT".
        hole_radius:      Radius (px) of the circle drawn around each hole.

    Returns:
        PNG-encoded bytes ready to stream as an HTTP response.
    """
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # Dynamic stroke/font scaling so markings stay proportional on any
    # resolution between a phone snapshot and a high-res scan.
    scale = max(w, h) / 1000.0
    thick_thin = max(1, int(scale * 2))
    thick_heavy = max(2, int(scale * 3))
    font_scale = max(0.5, scale * 0.7)
    cross_arm = max(18, int(scale * 24))

    x_mpi, y_mpi = int(round(mpi[0])), int(round(mpi[1]))
    x_ctr, y_ctr = int(round(target_center[0])), int(round(target_center[1]))

    # ------------------------------------------------------------------ #
    # 1. Bullet holes                                                       #
    # ------------------------------------------------------------------ #
    for (hx, hy) in shot_centroids:
        cx, cy = int(round(hx)), int(round(hy))
        cv2.circle(canvas, (cx, cy), hole_radius, _COLOR_HOLE, thick_heavy)
        cv2.circle(canvas, (cx, cy), max(2, hole_radius // 5), _COLOR_HOLE, -1)

    # ------------------------------------------------------------------ #
    # 2. Bullseye centre                                                    #
    # ------------------------------------------------------------------ #
    cv2.circle(canvas, (x_ctr, y_ctr), max(6, int(scale * 8)), _COLOR_CENTER, -1)
    cv2.circle(canvas, (x_ctr, y_ctr), max(10, int(scale * 14)), _COLOR_CENTER, thick_thin)

    # ------------------------------------------------------------------ #
    # 3. MPI crosshair                                                      #
    # ------------------------------------------------------------------ #
    cv2.line(canvas, (x_mpi - cross_arm, y_mpi), (x_mpi + cross_arm, y_mpi), _COLOR_MPI, thick_heavy)
    cv2.line(canvas, (x_mpi, y_mpi - cross_arm), (x_mpi, y_mpi + cross_arm), _COLOR_MPI, thick_heavy)
    cv2.circle(canvas, (x_mpi, y_mpi), max(4, cross_arm // 4), _COLOR_MPI, -1)

    # ------------------------------------------------------------------ #
    # 4. Correction arrow  MPI → target centre                             #
    # ------------------------------------------------------------------ #
    dist_px = np.hypot(x_ctr - x_mpi, y_ctr - y_mpi)
    if dist_px > 5:
        tip_length = min(0.35, 20.0 / dist_px)
        cv2.arrowedLine(
            canvas,
            (x_mpi, y_mpi),
            (x_ctr, y_ctr),
            _COLOR_ARROW,
            thick_heavy,
            tipLength=tip_length,
        )

    # ------------------------------------------------------------------ #
    # 5. Text callout                                                       #
    # ------------------------------------------------------------------ #
    font = cv2.FONT_HERSHEY_SIMPLEX
    padding = max(6, int(scale * 8))

    # Position the label near the midpoint of the arrow, offset upward
    label_x = int((x_mpi + x_ctr) / 2)
    label_y = int((y_mpi + y_ctr) / 2) - int(scale * 20)

    (tw, th), baseline = cv2.getTextSize(correction_label, font, font_scale, thick_thin)
    bg_x1 = label_x - padding
    bg_y1 = label_y - th - padding
    bg_x2 = label_x + tw + padding
    bg_y2 = label_y + baseline + padding

    # Clamp label box inside image bounds
    bg_x1 = max(0, min(bg_x1, w - (bg_x2 - bg_x1)))
    bg_y1 = max(0, min(bg_y1, h - (bg_y2 - bg_y1)))
    bg_x2 = bg_x1 + tw + padding * 2
    bg_y2 = bg_y1 + th + baseline + padding * 2

    cv2.rectangle(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), _COLOR_TEXT_BG, -1)
    cv2.putText(
        canvas,
        correction_label,
        (bg_x1 + padding, bg_y1 + th + padding // 2),
        font,
        font_scale,
        _COLOR_TEXT_FG,
        thick_thin,
        cv2.LINE_AA,
    )

    # ------------------------------------------------------------------ #
    # 6. Legend (top-left corner)                                           #
    # ------------------------------------------------------------------ #
    _draw_legend(canvas, font, font_scale * 0.85, thick_thin, padding, scale)

    # ------------------------------------------------------------------ #
    # 7. Encode to PNG                                                      #
    # ------------------------------------------------------------------ #
    success, buffer = cv2.imencode(".png", canvas)
    if not success:
        raise RuntimeError("cv2.imencode failed to produce PNG output.")
    return buffer.tobytes()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_legend(
    canvas: np.ndarray,
    font: int,
    font_scale: float,
    thickness: int,
    padding: int,
    scale: float,
) -> None:
    """Draw a small colour-coded legend in the top-left corner."""
    entries = [
        (_COLOR_HOLE,   "Bullet hole"),
        (_COLOR_MPI,    "MPI (centroid)"),
        (_COLOR_CENTER, "Bullseye centre"),
        (_COLOR_ARROW,  "Correction direction"),
    ]
    swatch = max(10, int(scale * 14))
    line_h = swatch + padding
    x0 = padding * 2
    y0 = padding * 2

    for i, (color, label) in enumerate(entries):
        y = y0 + i * line_h
        cv2.rectangle(canvas, (x0, y), (x0 + swatch, y + swatch), color, -1)
        cv2.putText(
            canvas,
            label,
            (x0 + swatch + padding, y + swatch - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def build_correction_label(corrections: dict) -> str:
    """
    Format a compact human-readable correction summary from the corrections dict.

    Example output: "↑ 4 UP  → 2 RIGHT"
    """
    arrows = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→", "NONE": "•"}

    elev = corrections["elevation"]
    wind = corrections["windage"]

    e_sym = arrows[elev["direction"]]
    w_sym = arrows[wind["direction"]]

    e_part = f"{e_sym} {abs(elev['clicks'])} {elev['direction']}"
    w_part = f"{w_sym} {abs(wind['clicks'])} {wind['direction']}"

    if elev["direction"] == "NONE" and wind["direction"] == "NONE":
        return "• ON ZERO"
    if elev["direction"] == "NONE":
        return w_part
    if wind["direction"] == "NONE":
        return e_part
    return f"{e_part}   {w_part}"
