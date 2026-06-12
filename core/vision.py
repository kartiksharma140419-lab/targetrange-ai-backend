"""
Vision Pipeline for TargetRange-AI.

Responsibilities
----------------
1. Load and cache a YOLOv8 object-detection model at startup.
2. Run inference to detect bullet holes and return their pixel centroids.
3. Use OpenCV to locate the bullseye centre and measure its pixel diameter
   (HoughCircles primary, contour bounding-box fallback).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model weights path
# ---------------------------------------------------------------------------
# Drop your custom-trained weights here after Google Colab training.
# Expected class index 0 → "hole".
MODEL_WEIGHTS_PATH = Path("models/best.pt")

# ---------------------------------------------------------------------------
# Module-level model cache — populated during FastAPI startup
# ---------------------------------------------------------------------------
_yolo_model = None


# ---------------------------------------------------------------------------
# Public: model lifecycle
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Load the YOLOv8 model from MODEL_WEIGHTS_PATH and cache it globally.

    Call this once inside FastAPI's lifespan startup handler so every request
    reuses the same in-memory model object without re-loading weights.

    Raises:
        FileNotFoundError: If the weights file is missing.
        RuntimeError: If ultralytics fails to load the model.
    """
    global _yolo_model

    if not MODEL_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"YOLOv8 weights not found at '{MODEL_WEIGHTS_PATH}'. "
            "Train the model in Google Colab and drop best.pt into the models/ directory."
        )

    try:
        from ultralytics import YOLO  # imported here to keep top-level import optional during tests
        _yolo_model = YOLO(str(MODEL_WEIGHTS_PATH))
        logger.info("YOLOv8 model loaded from '%s'.", MODEL_WEIGHTS_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load YOLOv8 model: {exc}") from exc


def get_model():
    """Return the cached YOLO model; raises if load_model() was never called."""
    if _yolo_model is None:
        raise RuntimeError(
            "YOLOv8 model is not loaded. Ensure load_model() ran at startup."
        )
    return _yolo_model


# ---------------------------------------------------------------------------
# Public: bullet-hole detection
# ---------------------------------------------------------------------------

def detect_bullet_holes(image_bgr: np.ndarray) -> list[tuple[float, float]]:
    """
    Run YOLOv8 inference and return the pixel centroid of every detected hole.

    The model is expected to produce detections of class 'hole' (class index 0).
    Bounding-box centres are used as the hole coordinate:
        X_i = (x_min + x_max) / 2
        Y_i = (y_min + y_max) / 2

    Args:
        image_bgr: BGR image array as returned by cv2.imdecode.

    Returns:
        List of (x, y) float tuples — one per detected hole, possibly empty.
    """
    model = get_model()
    results = model(image_bgr, verbose=False)

    centroids: list[tuple[float, float]] = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id != 0:
                continue
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            centroids.append((cx, cy))

    logger.debug("Detected %d bullet hole(s).", len(centroids))
    return centroids


# ---------------------------------------------------------------------------
# Public: bullseye geometry
# ---------------------------------------------------------------------------

def find_bullseye(
    image_bgr: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Locate the innermost concentric target ring using OpenCV.

    Strategy
    --------
    Primary  — Hough Circle Transform on a preprocessed grayscale image.
               The *smallest* detected circle is treated as the innermost ring.
    Fallback — Contour detection: find the smallest roughly-circular contour
               and return its bounding-box centre and diameter.

    Args:
        image_bgr: BGR image array.

    Returns:
        (x_center, y_center, pixel_diameter)
            All three are None when no ring can be found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # ------------------------------------------------------------------ #
    # PRIMARY: Hough Circle Transform                                      #
    # ------------------------------------------------------------------ #
    height, width = gray.shape
    min_radius = int(min(height, width) * 0.02)   # at least 2 % of shorter edge
    max_radius = int(min(height, width) * 0.40)   # at most 40 % — innermost ring

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2,
        param1=60,
        param2=35,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        # Select the smallest circle — the innermost ring
        smallest = min(circles, key=lambda c: c[2])
        cx, cy, radius = float(smallest[0]), float(smallest[1]), float(smallest[2])
        logger.debug(
            "Bullseye found via HoughCircles: centre=(%.1f, %.1f), diameter=%.1f px",
            cx, cy, radius * 2,
        )
        return cx, cy, radius * 2.0

    # ------------------------------------------------------------------ #
    # FALLBACK: Contour bounding-box geometry                              #
    # ------------------------------------------------------------------ #
    logger.warning("HoughCircles found no circles; falling back to contour detection.")
    return _bullseye_from_contours(gray)


def _bullseye_from_contours(
    gray: np.ndarray,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Contour-based fallback: find the smallest roughly-circular closed contour.

    A contour is considered circular when its circularity score
    (4π·area / perimeter²) exceeds 0.5.

    Returns (x_center, y_center, pixel_diameter) or (None, None, None).
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, float, float, float]] = []  # (area, cx, cy, diam)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.5:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w / 2.0
        cy = y + h / 2.0
        diameter = (w + h) / 2.0
        candidates.append((area, cx, cy, diameter))

    if not candidates:
        logger.error("Fallback contour detection also failed to locate a bullseye.")
        return None, None, None

    # Smallest circular contour → innermost ring
    _, cx, cy, diameter = min(candidates, key=lambda c: c[0])
    logger.debug(
        "Bullseye found via contour fallback: centre=(%.1f, %.1f), diameter=%.1f px",
        cx, cy, diameter,
    )
    return cx, cy, diameter
