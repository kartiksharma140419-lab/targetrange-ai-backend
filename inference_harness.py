"""
TARGETRANGE.AI — ASI LOCAL INFERENCE TESTING HARNESS
=====================================================
Single-file inference script that dynamically detects bullet hole positions
from any input image, runs the full ballistic pipeline, renders an annotated
overlay, and saves the result to disk.

Detection pipeline (tried in order per image):
  1. YOLO weights  — best.pt in models/  (lazy import, skipped if missing)
  2. Yellow/green annotation boxes — HSV threshold on annotated training samples
  3. OpenCV blob detector — dark circular spots on light targets (raw images)
  4. SIMULATED_SHOTS fallback — only if all above methods return zero hits

Display pipeline (tried in order):
  1. cv2.imshow  (only when $DISPLAY is set)
  2. PNG saved to disk via cv2.imwrite  (always executed)
  3. Matplotlib Agg render  (headless-safe, no plt.show())

Usage
-----
    cd artifacts/targetrange-ai
    python inference_harness.py
    python inference_harness.py --image train_sample_1.png --distance 100.0
    python inference_harness.py --image train_sample_2.png --distance 200.0
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# matplotlib backend MUST be set before any pyplot import — do it here at the
# very top of the module so no import path can accidentally load a GUI backend.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ===========================================================================
# Canvas / Physics Constants
# ===========================================================================

CANVAS_W: int = 640
CANVAS_H: int = 640
REF_X: float  = 320.0      # bullseye centre X (pixels)
REF_Y: float  = 320.0      # bullseye centre Y (pixels)

CM_PER_PIXEL_AT_100M: float = 0.05   # 1 px = 0.05 cm at 100 m
CM_PER_CLICK_AT_100M: float = 1.0    # 1 click = 1 cm at 100 m

# ---------------------------------------------------------------------------
# Ring geometry (drawn on synthetic canvas only — real images keep their look)
# ---------------------------------------------------------------------------
RINGS: list[tuple[int, tuple[int, int, int]]] = [
    (280, (40,  40,  40)),
    (220, (55,  55,  55)),
    (160, (70,  70,  70)),
    (110, (90,  90,  90)),
    (70,  (110, 110, 110)),
    (40,  (140, 140, 140)),
    (18,  (30,  30,  180)),
    (8,   (0,   0,   220)),
]

# OpenCV colour constants (BGR)
COL_HIT:       tuple[int, int, int] = (255, 255,   0)   # Cyan
COL_MPI:       tuple[int, int, int] = (0,     0, 255)   # Red
COL_CROSSHAIR: tuple[int, int, int] = (200, 200, 200)   # Light grey
COL_TEXT:      tuple[int, int, int] = (230, 230, 230)   # Near-white

# ---------------------------------------------------------------------------
# Fallback shot group — only used when every detection method returns 0 hits
# ---------------------------------------------------------------------------
SIMULATED_SHOTS: list[tuple[int, int]] = [
    (295, 335),
    (302, 328),
    (288, 341),
    (310, 350),
    (298, 322),
]

# Minimum / maximum pixel area of a valid bullet hole contour
_HOLE_AREA_MIN: int = 30
_HOLE_AREA_MAX: int = 6_000

# Minimum / maximum pixel area of a yellow annotation bounding box
_BOX_AREA_MIN: int = 60
_BOX_AREA_MAX: int = 20_000


# ===========================================================================
# Image loading — always reads fresh from disk, never cached globally
# ===========================================================================

def _load_image_bgr(image_path: str | None) -> np.ndarray | None:
    """
    Load any image from disk as a BGR ndarray.

    Handles RGBA (4-channel) PNG by dropping the alpha channel.
    Returns None if the path is invalid or the file cannot be decoded.
    Each call opens the file fresh — no global caching.
    """
    if not image_path or not os.path.isfile(image_path):
        return None

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if img.ndim == 2:                           # greyscale → BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:                     # BGRA → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def _build_canvas(image_path: str | None) -> np.ndarray:
    """
    Build the 640×640 display canvas.
    Real image → resized BGR copy.
    No image  → synthetic dark charcoal background with rings drawn on it.
    """
    img = _load_image_bgr(image_path)
    if img is not None:
        return cv2.resize(img, (CANVAS_W, CANVAS_H))

    # Synthetic charcoal background
    canvas = np.full((CANVAS_H, CANVAS_W, 3), fill_value=30, dtype=np.uint8)
    cx, cy = int(REF_X), int(REF_Y)
    for radius, colour in RINGS:
        cv2.circle(canvas, (cx, cy), radius, colour, thickness=-1)
        cv2.circle(canvas, (cx, cy), radius, (80, 80, 80), thickness=1)
    cv2.line(canvas, (cx - 290, cy), (cx + 290, cy), COL_CROSSHAIR, 1)
    cv2.line(canvas, (cx, cy - 290), (cx, cy + 290), COL_CROSSHAIR, 1)
    return canvas


# ===========================================================================
# Detection layer — three methods, tried in order of reliability
# ===========================================================================

def _detect_yellow_annotations(
    img_bgr: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[int, int]]:
    """
    Detect bullet hole positions from yellow/lime-green bounding box annotations
    already drawn on training-sample images by a prior model run.

    Works by thresholding for bright yellow-green pixels in HSV space and
    computing the centroid of each qualifying contour.  Coordinates are scaled
    to the 640×640 canvas reference frame.

    Returns an empty list if no annotation boxes are found.
    """
    native_h, native_w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Yellow-green range — covers both pure yellow (H≈30) and lime (H≈60)
    # that appear in annotated target samples.
    lower = np.array([20,  120, 120], dtype=np.uint8)
    upper = np.array([90,  255, 255], dtype=np.uint8)
    mask  = cv2.inRange(hsv, lower, upper)

    # Small morphological close to bridge tiny gaps in the box edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shots: list[tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (_BOX_AREA_MIN <= area <= _BOX_AREA_MAX):
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        # Centroid in native resolution → scale to canvas resolution
        cx_native = M["m10"] / M["m00"]
        cy_native = M["m01"] / M["m00"]
        cx_canvas = int(cx_native * canvas_w / native_w)
        cy_canvas = int(cy_native * canvas_h / native_h)
        shots.append((cx_canvas, cy_canvas))

    return shots


def _detect_blob_holes(
    img_bgr: np.ndarray,
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[int, int]]:
    """
    Detect dark circular bullet holes on a lighter background using OpenCV's
    SimpleBlobDetector.  Intended for raw, unannotated target images.

    Returns an empty list if no qualifying blobs are found.
    """
    native_h, native_w = img_bgr.shape[:2]
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea      = True
    params.minArea           = _HOLE_AREA_MIN
    params.maxArea           = _HOLE_AREA_MAX
    params.filterByCircularity = True
    params.minCircularity    = 0.35
    params.filterByConvexity = True
    params.minConvexity      = 0.60
    params.filterByInertia   = True
    params.minInertiaRatio   = 0.20
    params.filterByColor     = True
    params.blobColor         = 0          # dark blobs on light background

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(grey)

    shots: list[tuple[int, int]] = []
    for kp in keypoints:
        cx_canvas = int(kp.pt[0] * canvas_w / native_w)
        cy_canvas = int(kp.pt[1] * canvas_h / native_h)
        shots.append((cx_canvas, cy_canvas))

    return shots


def _detect_shots_from_image(image_path: str | None) -> list[tuple[int, int]]:
    """
    Master detection dispatcher.

    Tries in priority order:
      1. YOLO model weights  (best.pt via ultralytics — skipped if unavailable)
      2. Yellow annotation box detection  (HSV threshold)
      3. OpenCV blob detector  (dark holes on light background)
      4. SIMULATED_SHOTS  (guaranteed non-empty fallback)

    Returns a fresh list every call — no global state is read or written.
    """
    shots: list[tuple[int, int]] = []

    # --- Build path for YOLO weights relative to this file
    _script_dir    = os.path.dirname(os.path.abspath(__file__))
    _weights_path  = os.path.join(_script_dir, "models", "best.pt")

    # ---- Method 1: YOLO inference (lazy import — won't crash if not installed)
    if image_path and os.path.isfile(image_path) and os.path.isfile(_weights_path):
        try:
            from ultralytics import YOLO as _YOLO
            _model = _YOLO(_weights_path)
            preds  = _model.predict(source=image_path, conf=0.25, imgsz=640,
                                    device="cpu", verbose=False)
            for r in preds:
                if r.boxes is None:
                    continue
                for box in r.boxes.xywh.cpu().numpy():
                    shots.append((int(box[0]), int(box[1])))
            if shots:
                print(f"   [YOLO]  {len(shots)} hit(s) detected via model weights.")
                return shots
        except Exception as exc:
            print(f"   [YOLO]  Skipped ({exc.__class__.__name__}: {exc})")

    # ---- Load the original image once for OpenCV methods
    img_bgr = _load_image_bgr(image_path)
    if img_bgr is None:
        print("   [INFO]  No image loaded — using SIMULATED_SHOTS fallback.")
        return list(SIMULATED_SHOTS)

    # ---- Method 2: yellow/green annotation box centroids
    shots = _detect_yellow_annotations(img_bgr, CANVAS_W, CANVAS_H)
    if shots:
        print(f"   [CV2-Yellow]  {len(shots)} hit(s) detected via annotation boxes.")
        return shots

    # ---- Method 3: blob detector for dark bullet holes on light background
    shots = _detect_blob_holes(img_bgr, CANVAS_W, CANVAS_H)
    if shots:
        print(f"   [CV2-Blob]    {len(shots)} hit(s) detected via blob detector.")
        return shots

    # ---- Method 4: guaranteed fallback
    print("   [INFO]  No hits auto-detected — using SIMULATED_SHOTS fallback.")
    return list(SIMULATED_SHOTS)


# ===========================================================================
# Core inference pipeline
# ===========================================================================

def simulate_target_inference(
    image_path: str | None,
    target_distance_meters: float = 100.0,
) -> dict:
    """
    Run the full detection → ballistics → rendering pipeline for one image.

    Every data structure (canvas, shots list, telemetry values) is initialised
    fresh inside this function.  No global mutable state is shared between calls.

    Parameters
    ----------
    image_path:
        Path to the target image.  Pass None for a synthetic canvas.
    target_distance_meters:
        Shooting distance in metres.  Scales both cm-per-pixel and cm-per-click.

    Returns
    -------
    dict with keys: image, shots, mpi, deviation_px, deviation_cm,
                    scope_clicks, scope_directions, instruction, distance_m,
                    detection_method
    """
    # ---- Fresh canvas for this call (no reuse of previous result)
    canvas: np.ndarray = _build_canvas(image_path)

    # ---- Fresh shot list detected from the current image
    shots: list[tuple[int, int]] = _detect_shots_from_image(image_path)

    # ============================ STEP 1 — MPI (Mean Point of Impact)
    n: int         = len(shots)
    mpi_x: float   = sum(float(s[0]) for s in shots) / n
    mpi_y: float   = sum(float(s[1]) for s in shots) / n

    # ============================ STEP 2 — pixel deviation from bullseye
    # dev_x > 0 → group is RIGHT of centre
    # dev_y > 0 → group is ABOVE centre (Y inverted: image Y increases downward)
    dev_x_px: float = mpi_x - REF_X
    dev_y_px: float = REF_Y - mpi_y

    # ============================ STEP 3 — pixels → centimetres
    distance_scale: float = target_distance_meters / 100.0
    px_to_cm: float       = CM_PER_PIXEL_AT_100M * distance_scale
    dev_x_cm: float       = dev_x_px * px_to_cm
    dev_y_cm: float       = dev_y_px * px_to_cm

    # ============================ STEP 4 — scope clicks
    cm_per_click: float = CM_PER_CLICK_AT_100M * distance_scale
    raw_clicks_h: float = dev_x_cm / cm_per_click   # +ve → MPI right → click LEFT
    raw_clicks_v: float = dev_y_cm / cm_per_click   # +ve → MPI high  → click DOWN

    clicks_h: int = abs(round(raw_clicks_h))
    clicks_v: int = abs(round(raw_clicks_v))

    dir_h: str = "LEFT"  if raw_clicks_h > 0 else ("RIGHT" if raw_clicks_h < 0 else "ON TARGET")
    dir_v: str = "DOWN"  if raw_clicks_v > 0 else ("UP"    if raw_clicks_v < 0 else "ON TARGET")

    parts: list[str] = []
    if clicks_h:
        parts.append(f"{clicks_h} click{'s' if clicks_h != 1 else ''} {dir_h}")
    if clicks_v:
        parts.append(f"{clicks_v} click{'s' if clicks_v != 1 else ''} {dir_v}")
    instruction: str = (
        "Adjust turrets: " + " | ".join(parts)
        if parts else
        "Group is on target — no adjustment required."
    )

    # ============================ STEP 5 — draw hits onto canvas
    cx, cy = int(REF_X), int(REF_Y)

    # Bullet hole markers — Cyan filled circles
    for sx, sy in shots:
        cv2.circle(canvas, (int(sx), int(sy)), 6, COL_HIT, thickness=-1)
        cv2.circle(canvas, (int(sx), int(sy)), 7, (0, 0, 0), thickness=1)

    # MPI — distinct Red ring + crosshair
    mx, my = int(round(mpi_x)), int(round(mpi_y))
    cv2.circle(canvas, (mx, my), 10, COL_MPI, thickness=2)
    cv2.line(canvas, (mx - 14, my), (mx + 14, my), COL_MPI, 1)
    cv2.line(canvas, (mx, my - 14), (mx, my + 14), COL_MPI, 1)

    # Correction arrow: MPI → bullseye
    if abs(dev_x_px) > 1 or abs(dev_y_px) > 1:
        cv2.arrowedLine(canvas, (mx, my), (cx, cy),
                        color=(0, 200, 200), thickness=1, tipLength=0.15)

    # ============================ STEP 6 — metadata banner
    canvas = _draw_banner(
        canvas,
        image_path=image_path,
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
# Drawing helpers
# ===========================================================================

def _draw_banner(
    canvas: np.ndarray,
    *,
    image_path: str | None,
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
    """Append a dark telemetry banner below the canvas and return the combined image."""
    banner_h = 165
    banner   = np.full((banner_h, CANVAS_W, 3), fill_value=15, dtype=np.uint8)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    small     = 0.44
    medium    = 0.52
    col_label = (120, 200, 120)
    col_val   = COL_TEXT

    src_label = os.path.basename(image_path) if image_path else "(synthetic canvas)"

    lines = [
        ("TARGETRANGE.AI — INFERENCE RESULT",                              medium, (0, 180, 180)),
        (f"Source: {src_label}  |  Distance: {distance_m:.0f} m",         small,  col_val),
        (f"Total Hits: {len(shots)}  |  MPI: ({mpi_x:.1f}, {mpi_y:.1f}) px", small, col_val),
        (f"Deviation: H={dev_x_cm:+.2f} cm  V={dev_y_cm:+.2f} cm",       small,  col_label),
        (f"Windage  : {clicks_h} click(s) {dir_h}",                       small,  col_label),
        (f"Elevation: {clicks_v} click(s) {dir_v}",                       small,  col_label),
        (instruction,                                                       small,  (0, 220, 220)),
    ]

    y = 22
    for text, scale, colour in lines:
        cv2.putText(banner, text, (10, y), font, scale, colour, 1, cv2.LINE_AA)
        y += int(scale * 52) + 4

    return np.vstack([canvas, banner])


# ===========================================================================
# Display — headless-safe, no blocking calls
# ===========================================================================

def _display(image: np.ndarray, output_path: str = "inference_result.png") -> None:
    """
    Save the annotated image to disk (always) and optionally open a GUI window.

    Matplotlib is configured for Agg at module import time, so plt.show() is
    never called — we save directly with plt.savefig() and call plt.close()
    to release the figure memory cleanly.
    """
    # Always write raw OpenCV PNG — available immediately in the Files panel
    cv2.imwrite(output_path, image)
    print(f"   [+] Raw PNG saved    → {os.path.abspath(output_path)}")

    # Try X11 GUI popup only if a real display is present
    if os.environ.get("DISPLAY"):
        try:
            cv2.imshow("TargetRange.ai - Local Inference Window", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"   [!] cv2.imshow failed: {exc}")

    # Matplotlib Agg render — no plt.show(), no blocking, no warning
    try:
        mpl_path = output_path.replace(".png", "_overlay.png")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8, 9), dpi=110)
        ax.imshow(rgb)
        ax.axis("off")
        ax.set_title("TargetRange.ai — Local Inference", fontsize=11,
                     color="white", pad=6)
        fig.patch.set_facecolor("#0d0d0d")
        plt.tight_layout(pad=0.4)
        plt.savefig(mpl_path, dpi=120, bbox_inches="tight")
        plt.close(fig)                           # release figure memory — no blocking
        print(f"   [+] Overlay PNG saved → {os.path.abspath(mpl_path)}")
    except Exception as exc:
        print(f"   [!] Matplotlib render failed: {exc}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TargetRange.AI — ASI local inference testing harness"
    )
    parser.add_argument("--image",    type=str,   default=None,
                        help="Path to target image (JPEG/PNG). Default: synthetic canvas.")
    parser.add_argument("--distance", type=float, default=100.0,
                        help="Shooting distance in metres (default: 100.0).")
    parser.add_argument("--output",   type=str,   default="inference_result.png",
                        help="Output PNG file name (default: inference_result.png).")
    args = parser.parse_args()

    print()
    print("=" * 64)
    print(">> INITIALIZING ARTIFICIAL SOLDIER INTELLIGENCE")
    print("   LOCAL INFERENCE HARNESS...")
    print("=" * 64)
    print(f"   Image src : {args.image or '(synthetic canvas)'}")
    print(f"   Distance  : {args.distance:.1f} m")
    print(f"   Output    : {args.output}")
    print()

    result = simulate_target_inference(
        image_path=args.image,
        target_distance_meters=args.distance,
    )

    print()
    print("── BALLISTIC TELEMETRY ─────────────────────────────────────")
    print(f"   Total hits    : {len(result['shots'])}")
    print(f"   Hit coords    : {result['shots']}")
    print(f"   MPI           : x={result['mpi']['x']} px  y={result['mpi']['y']} px")
    print(f"   Deviation     : H={result['deviation_cm']['x']:+.2f} cm  "
          f"V={result['deviation_cm']['y']:+.2f} cm")
    print(f"   Windage       : {result['scope_clicks']['horizontal']} click(s) "
          f"{result['scope_directions']['horizontal']}")
    print(f"   Elevation     : {result['scope_clicks']['vertical']} click(s) "
          f"{result['scope_directions']['vertical']}")
    print(f"   Instruction   : {result['instruction']}")
    print("────────────────────────────────────────────────────────────")
    print()
    print("── TELEMETRY JSON ──────────────────────────────────────────")
    print(json.dumps({k: v for k, v in result.items() if k != "image"}, indent=2))
    print("────────────────────────────────────────────────────────────")
    print()

    _display(result["image"], output_path=args.output)

    print()
    print(">> INFERENCE HARNESS COMPLETE.")
    print()
