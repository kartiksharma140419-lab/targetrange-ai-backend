"""
TargetRange-AI — FastAPI entry point.

Routes
------
POST /api/v1/process-target
    Accepts a multipart form containing the target image and shot parameters,
    runs the vision + ballistics pipeline, and returns structured correction data.

POST /api/v1/overlay
    Same inputs as /api/v1/process-target. Returns a PNG image with bullet
    holes, MPI crosshair, bullseye centre, and correction arrow annotated.

GET  /api/v1/healthz
    Liveness probe.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from core import ballistics, overlay, vision

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class Coordinates(BaseModel):
    x: float
    y: float


class DeviationsInches(BaseModel):
    horizontal: float = Field(..., description="Positive = MPI right of centre.")
    vertical: float = Field(..., description="Positive = MPI above centre.")


class AdjustmentAxis(BaseModel):
    clicks: int
    direction: Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


class Corrections(BaseModel):
    elevation: AdjustmentAxis
    windage: AdjustmentAxis


class ProcessTargetResponse(BaseModel):
    status: Literal["success", "error", "no_shots"]
    message: str
    total_shots_detected: int
    mpi_coordinates: Optional[Coordinates] = None
    target_center_coordinates: Optional[Coordinates] = None
    raw_deviations_inches: Optional[DeviationsInches] = None
    corrections: Optional[Corrections] = None


# ---------------------------------------------------------------------------
# Lifespan — model loaded once at startup, released at shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TargetRange-AI starting up — loading YOLOv8 model …")
    try:
        vision.load_model()
        logger.info("YOLOv8 model ready.")
    except FileNotFoundError as exc:
        # Non-fatal in development: the model file may not exist yet.
        # Requests will fail gracefully with a 503 until weights are present.
        logger.warning("Model not found at startup: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error loading model: %s", exc, exc_info=True)
    yield
    logger.info("TargetRange-AI shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TargetRange-AI",
    version="1.0.0",
    description="Automated Vision AI system for small-arms zeroing and ballistic grouping assessment.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/healthz", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "TargetRange-AI"}


@app.post(
    "/api/v1/process-target",
    response_model=ProcessTargetResponse,
    tags=["Analysis"],
    summary="Analyse a target photo and return scope correction clicks.",
)
async def process_target(
    image: UploadFile = File(..., description="Target photo — JPEG or PNG."),
    distance_yards: float = Form(default=100.0, ge=1.0, description="Shooting distance in yards."),
    click_value_moa: float = Form(default=0.25, gt=0.0, description="MOA value per scope click."),
    known_bullseye_diameter_inches: float = Form(
        default=4.0,
        gt=0.0,
        description="Real-world diameter of the innermost target ring (inches).",
    ),
):
    # ------------------------------------------------------------------ #
    # 1. Decode uploaded image                                             #
    # ------------------------------------------------------------------ #
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to decode image. Ensure the file is a valid JPEG or PNG.",
        )

    # ------------------------------------------------------------------ #
    # 2. YOLO — detect bullet holes                                        #
    # ------------------------------------------------------------------ #
    try:
        shot_centroids = vision.detect_bullet_holes(image_bgr)
    except RuntimeError as exc:
        # Model not loaded (weights missing) — return friendly error
        return ProcessTargetResponse(
            status="error",
            message=f"Vision model unavailable: {exc}",
            total_shots_detected=0,
        )

    if not shot_centroids:
        logger.info("No bullet holes detected in submitted image.")
        return ProcessTargetResponse(
            status="no_shots",
            message="No bullet holes were detected in the image. Verify image quality and model weights.",
            total_shots_detected=0,
        )

    # ------------------------------------------------------------------ #
    # 3. OpenCV — locate bullseye centre                                   #
    # ------------------------------------------------------------------ #
    x_center, y_center, pixel_diameter = vision.find_bullseye(image_bgr)

    if x_center is None:
        logger.error("Bullseye could not be located — aborting analysis.")
        return ProcessTargetResponse(
            status="error",
            message=(
                "Could not locate the bullseye ring in the image. "
                "Ensure the target is fully visible and well-lit."
            ),
            total_shots_detected=len(shot_centroids),
        )

    # ------------------------------------------------------------------ #
    # 4. Ballistic engine                                                  #
    # ------------------------------------------------------------------ #
    x_mpi, y_mpi = ballistics.compute_mpi(shot_centroids)

    scaling_factor = ballistics.compute_scaling_factor(
        known_bullseye_diameter_inches=known_bullseye_diameter_inches,
        bullseye_pixel_diameter=pixel_diameter,
    )

    delta_x_inches, delta_y_inches = ballistics.compute_linear_deviations(
        x_mpi=x_mpi,
        y_mpi=y_mpi,
        x_center=x_center,
        y_center=y_center,
        scaling_factor=scaling_factor,
    )

    correction_data = ballistics.compute_scope_corrections(
        delta_x_inches=delta_x_inches,
        delta_y_inches=delta_y_inches,
        distance_yards=distance_yards,
        click_value_moa=click_value_moa,
    )

    # ------------------------------------------------------------------ #
    # 5. Build and return structured response                              #
    # ------------------------------------------------------------------ #
    logger.info(
        "Analysis complete — %d shot(s), MPI=(%.1f, %.1f), "
        "windage %+d click(s) %s, elevation %+d click(s) %s.",
        len(shot_centroids),
        x_mpi, y_mpi,
        correction_data["windage"]["clicks"],
        correction_data["windage"]["direction"],
        correction_data["elevation"]["clicks"],
        correction_data["elevation"]["direction"],
    )

    return ProcessTargetResponse(
        status="success",
        message=f"Analysis complete. {len(shot_centroids)} shot(s) processed.",
        total_shots_detected=len(shot_centroids),
        mpi_coordinates=Coordinates(x=round(x_mpi, 2), y=round(y_mpi, 2)),
        target_center_coordinates=Coordinates(x=round(x_center, 2), y=round(y_center, 2)),
        raw_deviations_inches=DeviationsInches(
            horizontal=round(delta_x_inches, 4),
            vertical=round(delta_y_inches, 4),
        ),
        corrections=Corrections(
            elevation=AdjustmentAxis(**correction_data["elevation"]),
            windage=AdjustmentAxis(**correction_data["windage"]),
        ),
    )


@app.post(
    "/api/v1/overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    tags=["Analysis"],
    summary="Return an annotated PNG with holes, MPI crosshair, and correction arrow.",
)
async def overlay_target(
    image: UploadFile = File(..., description="Target photo — JPEG or PNG."),
    distance_yards: float = Form(default=100.0, ge=1.0, description="Shooting distance in yards."),
    click_value_moa: float = Form(default=0.25, gt=0.0, description="MOA value per scope click."),
    known_bullseye_diameter_inches: float = Form(
        default=4.0,
        gt=0.0,
        description="Real-world diameter of the innermost target ring (inches).",
    ),
):
    # ------------------------------------------------------------------ #
    # 1. Decode image                                                       #
    # ------------------------------------------------------------------ #
    raw_bytes = await image.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to decode image. Ensure the file is a valid JPEG or PNG.",
        )

    # ------------------------------------------------------------------ #
    # 2. Vision — holes + bullseye                                         #
    # ------------------------------------------------------------------ #
    try:
        shot_centroids = vision.detect_bullet_holes(image_bgr)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Vision model unavailable: {exc}")

    if not shot_centroids:
        raise HTTPException(
            status_code=422,
            detail="No bullet holes detected — cannot render overlay.",
        )

    x_center, y_center, pixel_diameter = vision.find_bullseye(image_bgr)
    if x_center is None:
        raise HTTPException(
            status_code=422,
            detail="Bullseye ring not found — cannot render overlay.",
        )

    # ------------------------------------------------------------------ #
    # 3. Ballistics                                                         #
    # ------------------------------------------------------------------ #
    x_mpi, y_mpi = ballistics.compute_mpi(shot_centroids)

    scaling_factor = ballistics.compute_scaling_factor(
        known_bullseye_diameter_inches=known_bullseye_diameter_inches,
        bullseye_pixel_diameter=pixel_diameter,
    )

    delta_x_inches, delta_y_inches = ballistics.compute_linear_deviations(
        x_mpi=x_mpi,
        y_mpi=y_mpi,
        x_center=x_center,
        y_center=y_center,
        scaling_factor=scaling_factor,
    )

    correction_data = ballistics.compute_scope_corrections(
        delta_x_inches=delta_x_inches,
        delta_y_inches=delta_y_inches,
        distance_yards=distance_yards,
        click_value_moa=click_value_moa,
    )

    # ------------------------------------------------------------------ #
    # 4. Render and return PNG                                              #
    # ------------------------------------------------------------------ #
    label = overlay.build_correction_label(correction_data)

    png_bytes = overlay.render_overlay(
        image_bgr=image_bgr,
        shot_centroids=shot_centroids,
        mpi=(x_mpi, y_mpi),
        target_center=(x_center, y_center),
        correction_label=label,
    )

    logger.info(
        "Overlay rendered — %d shot(s), label='%s'.",
        len(shot_centroids),
        label,
    )

    return Response(content=png_bytes, media_type="image/png")
