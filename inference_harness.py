"""
TARGETRANGE.AI — ASI LOCAL INFERENCE TESTING HARNESS
=====================================================
Single-file simulation script that mocks the trained YOLO model output,
runs the full ballistic computation pipeline, renders the visual target
overlay with OpenCV, and displays or exports the result.

Usage
-----
    python inference_harness.py
    python inference_harness.py --distance 200.0

Environment
-----------
Works headless (Replit / CI) and with a display.  Display priority:
  1. X11 popup via cv2.imshow  (if DISPLAY is set)
  2. matplotlib interactive window  (if a backend is available)
  3. Saved PNG file to disk  (always produced as fallback)
"""

from __future__ import annotations

import json
import math
import os
import sys
import argparse

import cv2
import numpy as np


# ===========================================================================
# Canvas / Physics Constants
# ===========================================================================

CANVAS_W: int   = 640        # pixels
CANVAS_H: int   = 640        # pixels
REF_X: float    = 320.0      # bullseye centre — X (pixels)
REF_Y: float    = 320.0      # bullseye centre — Y (pixels)

CM_PER_PIXEL_AT_100M: float = 0.05   # 1 px = 0.05 cm at 100 m
CM_PER_CLICK_AT_100M: float = 1.0    # 1 click = 1 cm at 100 m

# ---------------------------------------------------------------------------
# Target ring radii (px) and their BGRcolours
# ---------------------------------------------------------------------------
RINGS: list[tuple[int, tuple[int, int, int]]] = [
    (280, (40,  40,  40)),   # outer  — dark charcoal
    (220, (55,  55,  55)),
    (160, (70,  70,  70)),
    (110, (90,  90,  90)),
    (70,  (110, 110, 110)),
    (40,  (140, 140, 140)),
    (18,  (30,  30,  180)),  # inner bull — deep blue
    (8,   (0,   0,   220)),  # X-ring — bright blue
]

# OpenCV colour constants (BGR)
COL_HIT: tuple[int, int, int]       = (255, 255, 0)    # Cyan
COL_MPI: tuple[int, int, int]       = (0,   0,   255)  # Red
COL_CROSSHAIR: tuple[int, int, int] = (200, 200, 200)  # Light grey
COL_TEXT: tuple[int, int, int]      = (230, 230, 230)  # Near-white
COL_BANNER: tuple[int, int, int]    = (20,  20,  20)   # Very dark

# ---------------------------------------------------------------------------
# Hard-coded simulated shot group (pixel coordinates on 640x640 canvas).
# Cluster is intentionally offset slightly down-left to produce non-trivial
# correction output.
# ---------------------------------------------------------------------------
SIMULATED_SHOTS: list[tuple[int, int]] = [
    (295, 335),   # shot 1
    (302, 328),   # shot 2
    (288, 341),   # shot 3
    (310, 350),   # shot 4
    (298, 322),   # shot 5
]


# ===========================================================================
# Core inference simulator
# ===========================================================================

def simulate_target_inference(
    image_path: str | None,
    target_distance_meters: float = 100.0,
) -> dict:
    """
    Mock the trained model's coordinate output and compute full ballistic telemetry.

    Parameters
    ----------
    image_path:
        Optional path to a background target image.  If ``None`` or the file
        does not exist the function generates a synthetic charcoal canvas.
    target_distance_meters:
        Distance from shooter to target in metres.  All scaling factors are
        adjusted proportionally relative to the 100 m baseline.

    Returns
    -------
    dict
        Telemetry payload::

            {
                "image"            : np.ndarray  — annotated BGR canvas,
                "shots"            : list[tuple] — raw pixel coords,
                "mpi"              : {"x": float, "y": float},
                "deviation_px"     : {"x": float, "y": float},
                "deviation_cm"     : {"x": float, "y": float},
                "scope_clicks"     : {"horizontal": int, "vertical": int},
                "scope_directions" : {"horizontal": str, "vertical": str},
                "instruction"      : str,
                "distance_m"       : float,
            }
    """
    # ---------------------------------------------------------------- canvas
    canvas = _build_canvas(image_path)

    # ----------------------------------------------------------- draw rings
    cx, cy = int(REF_X), int(REF_Y)
    for radius, colour in RINGS:
        cv2.circle(canvas, (cx, cy), radius, colour, thickness=-1)
        cv2.circle(canvas, (cx, cy), radius, (80, 80, 80), thickness=1)

    # -------------------------------------------------------- draw crosshair
    cv2.line(canvas, (cx - 290, cy), (cx + 290, cy), COL_CROSSHAIR, 1)
    cv2.line(canvas, (cx, cy - 290), (cx, cy + 290), COL_CROSSHAIR, 1)

    # ------------------------------------------------------ shot coordinates
    shots = SIMULATED_SHOTS

    # ========================================================= STEP 1 — MPI
    n = len(shots)
    mpi_x: float = sum(s[0] for s in shots) / n
    mpi_y: float = sum(s[1] for s in shots) / n

    # ================================== STEP 2 — pixel deviation from centre
    # dev_x positive → MPI is RIGHT of centre
    # dev_y positive → MPI is ABOVE centre (Y axis inverted vs image space)
    dev_x_px: float = mpi_x - REF_X
    dev_y_px: float = REF_Y - mpi_y          # <-- Y inversion

    # ========================= STEP 3 — convert pixels → real-world cm
    # Scale factor is linear with distance: at 200 m each pixel covers 2×
    # as much physical space, so the cm deviation doubles.
    distance_scale: float   = target_distance_meters / 100.0
    px_to_cm: float         = CM_PER_PIXEL_AT_100M * distance_scale

    dev_x_cm: float = dev_x_px * px_to_cm
    dev_y_cm: float = dev_y_px * px_to_cm

    # ======================== STEP 4 — scope clicks (1 click = 1 cm @ 100 m)
    # At distance d m: 1 click = CM_PER_CLICK_AT_100M * (d / 100)
    cm_per_click: float = CM_PER_CLICK_AT_100M * distance_scale

    raw_clicks_h: float = dev_x_cm / cm_per_click   # + → MPI right → click LEFT
    raw_clicks_v: float = dev_y_cm / cm_per_click   # + → MPI high  → click DOWN

    clicks_h: int = abs(round(raw_clicks_h))
    clicks_v: int = abs(round(raw_clicks_v))

    dir_h: str = "LEFT"  if raw_clicks_h > 0 else "RIGHT" if raw_clicks_h < 0 else "ON TARGET"
    dir_v: str = "DOWN"  if raw_clicks_v > 0 else "UP"    if raw_clicks_v < 0 else "ON TARGET"

    # Human-readable correction instruction
    parts: list[str] = []
    if clicks_h:
        parts.append(f"{clicks_h} click{'s' if clicks_h != 1 else ''} {dir_h}")
    if clicks_v:
        parts.append(f"{clicks_v} click{'s' if clicks_v != 1 else ''} {dir_v}")
    instruction: str = (
        "Adjust turrets: " + " | ".join(parts) if parts else "Group is on target — no adjustment required."
    )

    # ======================================================= STEP 5 — draw
    # Individual bullet holes — Cyan filled circles
    for sx, sy in shots:
        cv2.circle(canvas, (sx, sy), 6, COL_HIT, thickness=-1)
        cv2.circle(canvas, (sx, sy), 7, (0, 0, 0), thickness=1)   # thin black edge

    # MPI — distinct Red circle + crosshair marker
    mx, my = int(round(mpi_x)), int(round(mpi_y))
    cv2.circle(canvas, (mx, my), 10, COL_MPI, thickness=2)
    cv2.line(canvas, (mx - 14, my), (mx + 14, my), COL_MPI, 1)
    cv2.line(canvas, (mx, my - 14), (mx, my + 14), COL_MPI, 1)

    # Arrow from MPI → bullseye (shows required physical correction direction)
    if dev_x_px != 0 or dev_y_px != 0:
        cv2.arrowedLine(
            canvas, (mx, my), (cx, cy),
            color=(0, 200, 200), thickness=1, tipLength=0.15,
        )

    # ============================================= STEP 6 — metadata banner
    canvas = _draw_banner(
        canvas,
        shots=shots,
        mpi_x=mpi_x, mpi_y=mpi_y,
        dev_x_cm=dev_x_cm, dev_y_cm=dev_y_cm,
        clicks_h=clicks_h, dir_h=dir_h,
        clicks_v=clicks_v, dir_v=dir_v,
        distance_m=target_distance_meters,
        instruction=instruction,
    )

    return {
        "image":            canvas,
        "shots":            shots,
        "mpi":              {"x": round(mpi_x, 2), "y": round(mpi_y, 2)},
        "deviation_px":     {"x": round(dev_x_px, 2), "y": round(dev_y_px, 2)},
        "deviation_cm":     {"x": round(dev_x_cm, 2), "y": round(dev_y_cm, 2)},
        "scope_clicks":     {"horizontal": clicks_h, "vertical": clicks_v},
        "scope_directions": {"horizontal": dir_h,    "vertical": dir_v},
        "instruction":      instruction,
        "distance_m":       target_distance_meters,
    }


# ===========================================================================
# Private drawing helpers
# ===========================================================================

def _build_canvas(image_path: str | None) -> np.ndarray:
    """Return a 640×640 BGR canvas — from file if available, else synthetic."""
    if image_path and os.path.isfile(image_path):
        img = cv2.imread(image_path)
        if img is not None:
            return cv2.resize(img, (CANVAS_W, CANVAS_H))

    # Synthetic charcoal dark canvas
    canvas = np.full((CANVAS_H, CANVAS_W, 3), fill_value=30, dtype=np.uint8)
    return canvas


def _draw_banner(
    canvas: np.ndarray,
    *,
    shots: list[tuple[int, int]],
    mpi_x: float,
    mpi_y: float,
    dev_x_cm: float,
    dev_y_cm: float,
    clicks_h: int,
    dir_h: str,
    clicks_v: int,
    dir_v: str,
    distance_m: float,
    instruction: str,
) -> np.ndarray:
    """
    Append a dark metadata banner below the target canvas with telemetry text.
    Returns the combined image (canvas + banner stacked vertically).
    """
    banner_h = 160
    banner = np.full((banner_h, CANVAS_W, 3), fill_value=15, dtype=np.uint8)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    small      = 0.44
    medium     = 0.52
    col_label  = (120, 200, 120)   # green-tinted labels
    col_value  = COL_TEXT

    lines = [
        ("TARGETRANGE.AI — INFERENCE RESULT",                        medium, (0, 180, 180)),
        (f"Distance: {distance_m:.0f} m  |  Total Hits: {len(shots)}",   small, col_value),
        (f"MPI: ({mpi_x:.1f} px, {mpi_y:.1f} px)",                       small, col_value),
        (f"Deviation: H={dev_x_cm:+.2f} cm  V={dev_y_cm:+.2f} cm",       small, col_label),
        (f"Windage : {clicks_h} click(s) {dir_h}",                        small, col_label),
        (f"Elevation: {clicks_v} click(s) {dir_v}",                       small, col_label),
        (instruction,                                                       small, (0, 220, 220)),
    ]

    y = 22
    for text, scale, colour in lines:
        cv2.putText(banner, text, (10, y), font, scale, colour, 1, cv2.LINE_AA)
        y += int(scale * 52) + 4

    return np.vstack([canvas, banner])


# ===========================================================================
# Display dispatcher
# ===========================================================================

def _display(image: np.ndarray, output_path: str = "inference_result.png") -> None:
    """
    Show the rendered canvas, trying display methods in order of preference:
      1. cv2.imshow  (X11 available)
      2. matplotlib  (interactive or inline notebook backend)
      3. Save to PNG  (headless fallback — always executed)
    """
    # Always save to disk so the result is accessible regardless of env
    cv2.imwrite(output_path, image)
    print(f"   [+] Image saved → {os.path.abspath(output_path)}")

    # Try X11 popup first
    if os.environ.get("DISPLAY"):
        try:
            cv2.imshow("TargetRange.ai - Local Inference Window", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        except cv2.error:
            pass

    # Matplotlib fallback (works in Replit desktop canvas and Jupyter)
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")                        # non-interactive safe backend
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8, 9), dpi=100)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title(
            "TargetRange.ai — Local Inference",
            fontsize=11, color="white", pad=6,
        )
        fig.patch.set_facecolor("#0d0d0d")
        plt.tight_layout(pad=0.5)
        plt.savefig(output_path.replace(".png", "_matplotlib.png"), dpi=120, bbox_inches="tight")
        print(f"   [+] Matplotlib render → {output_path.replace('.png', '_matplotlib.png')}")
        plt.show()
    except Exception as exc:
        print(f"   [!] Matplotlib unavailable: {exc}")
        print("   [+] Open the saved PNG from the Files panel instead.")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TargetRange.AI — local inference testing harness"
    )
    parser.add_argument(
        "--distance",
        type=float,
        default=100.0,
        help="Shooting distance in metres (default: 100.0)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a background target image (JPEG / PNG).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_result.png",
        help="File path for the saved output image (default: inference_result.png).",
    )
    args = parser.parse_args()

    print()
    print("=" * 64)
    print(">> INITIALIZING ARTIFICIAL SOLDIER INTELLIGENCE")
    print("   LOCAL INFERENCE HARNESS...")
    print("=" * 64)
    print(f"   Distance  : {args.distance:.1f} m")
    print(f"   Image src : {args.image or '(synthetic canvas)'}")
    print()

    result = simulate_target_inference(
        image_path=args.image,
        target_distance_meters=args.distance,
    )

    # Console telemetry dump
    print("── BALLISTIC TELEMETRY ─────────────────────────────────────")
    print(f"   Total hits    : {len(result['shots'])}")
    print(f"   MPI           : x={result['mpi']['x']} px  y={result['mpi']['y']} px")
    print(f"   Deviation     : H={result['deviation_cm']['x']:+.2f} cm  V={result['deviation_cm']['y']:+.2f} cm")
    print(f"   Windage       : {result['scope_clicks']['horizontal']} click(s) {result['scope_directions']['horizontal']}")
    print(f"   Elevation     : {result['scope_clicks']['vertical']} click(s) {result['scope_directions']['vertical']}")
    print(f"   Instruction   : {result['instruction']}")
    print("────────────────────────────────────────────────────────────")
    print()
    print("── TELEMETRY JSON ──────────────────────────────────────────")
    payload = {k: v for k, v in result.items() if k != "image"}
    print(json.dumps(payload, indent=2))
    print("────────────────────────────────────────────────────────────")
    print()

    _display(result["image"], output_path=args.output)

    print()
    print(">> INFERENCE HARNESS COMPLETE.")
    print()
