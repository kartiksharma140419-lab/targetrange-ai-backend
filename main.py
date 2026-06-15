"""
TargetRange-AI — FastAPI entry point.

Routes
------
POST /api/v1/process-target
    Accepts a multipart form containing the target image and distance, runs
    the full vision + ballistics pipeline, and returns a consolidated JSON
    payload that includes both the legacy MOA-based corrections and the new
    structured zeroing-vector output from ``calculate_zeroing_vectors``.

POST /api/v1/overlay
    Same inputs as /api/v1/process-target. Returns a PNG image with bullet
    holes, MPI crosshair, bullseye centre, and correction arrow annotated.

GET  /api/v1/healthz
    Liveness probe.
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

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
# Metres → yards conversion constant (legacy MOA path expects yards)
# ---------------------------------------------------------------------------
_M_TO_YD: float = 1.09361


# ---------------------------------------------------------------------------
# Baseline bullseye reference point for calculate_zeroing_vectors
# Corresponds to the exact pixel centre of a 640 × 640 inference frame.
# ---------------------------------------------------------------------------
_BULLSEYE_CENTRE: dict = {"x": 320.0, "y": 320.0}


# ===========================================================================
# Pydantic response models
# ===========================================================================

class Coordinates(BaseModel):
    x: float
    y: float


class DeviationsInches(BaseModel):
    horizontal: float = Field(..., description="Positive = MPI right of centre.")
    vertical: float   = Field(..., description="Positive = MPI above centre.")


class AdjustmentAxis(BaseModel):
    clicks:    int
    direction: Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]


class Corrections(BaseModel):
    elevation: AdjustmentAxis
    windage:   AdjustmentAxis


# --- New structured zeroing-vector sub-models --------------------------------

class ZeroingMPI(BaseModel):
    x: float
    y: float


class ZeroingDeviationCm(BaseModel):
    horizontal: float = Field(..., description="Positive = group is RIGHT of target centre.")
    vertical:   float = Field(..., description="Positive = group is HIGH (ballistic convention).")


class ZeroingSummary(BaseModel):
    total_hits:   int
    mpi:          ZeroingMPI
    deviation_cm: ZeroingDeviationCm


class ZeroingCorrectionAxis(BaseModel):
    clicks:      int
    direction:   Literal["UP", "DOWN", "LEFT", "RIGHT", "NONE"]
    instruction: str


class ZeroingCorrections(BaseModel):
    elevation:            ZeroingCorrectionAxis
    windage:              ZeroingCorrectionAxis
    combined_instruction: str


# --- Top-level response model -----------------------------------------------

class ProcessTargetResponse(BaseModel):
    # Core status
    status:  Literal["success", "error", "no_shots"]
    message: str

    # Detection metadata
    total_shots_detected: int

    # Legacy MOA-based output (preserved for backward compatibility)
    mpi_coordinates:            Optional[Coordinates]     = None
    target_center_coordinates:  Optional[Coordinates]     = None
    raw_deviations_inches:      Optional[DeviationsInches] = None
    corrections:                Optional[Corrections]      = None

    # New structured zeroing-vector output (from calculate_zeroing_vectors)
    zeroing_summary:     Optional[ZeroingSummary]     = None
    zeroing_corrections: Optional[ZeroingCorrections] = None


# ===========================================================================
# Internal dataclass — full pipeline result carrier
# ===========================================================================

@dataclass
class PipelineResult:
    """Carries every computed value from the shared vision + ballistics pipeline."""
    detections:       list[dict]
    shot_centroids:   list[tuple[float, float]]
    x_mpi:            float
    y_mpi:            float
    x_center:         float
    y_center:         float
    pixel_diameter:   float
    delta_x_inches:   float
    delta_y_inches:   float
    correction_data:  dict
    zeroing_result:   dict = field(default_factory=dict)


# ===========================================================================
# Lifespan — model loaded once at startup, released at shutdown
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TargetRange-AI starting up — loading YOLOv8 model …")
    try:
        vision.load_model()
        logger.info("YOLOv8 model ready.")
    except FileNotFoundError as exc:
        # Non-fatal during development: weights may not exist yet.
        # Requests fail gracefully until best.pt is placed in models/.
        logger.warning("Model not found at startup: %s", exc)
    except Exception as exc:
        logger.error("Unexpected error loading model: %s", exc, exc_info=True)
    yield
    logger.info("TargetRange-AI shutting down.")


# ===========================================================================
# App
# ===========================================================================

app = FastAPI(
    title="TargetRange-AI",
    version="1.0.0",
    description=(
        "Automated Vision AI system for small-arms zeroing "
        "and ballistic grouping assessment."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Internal helpers
# ===========================================================================

def _decode_and_persist(raw_bytes: bytes) -> tuple[str, np.ndarray]:
    """
    Validate, decode, and persist uploaded image bytes to a named temporary
    file on disk.

    ``extract_bullet_coordinates`` requires a file-system path; this helper
    bridges the gap between FastAPI's in-memory upload and the YOLO predictor.

    The caller **must** delete the temp file after use — always wrap calls to
    this function in a try/finally block and call ``os.unlink(temp_path)``.

    Args:
        raw_bytes: Raw bytes read from the FastAPI ``UploadFile`` object.

    Returns:
        ``(temp_path, image_bgr)`` — absolute path to the persisted JPEG and
        the decoded BGR ndarray for OpenCV operations.

    Raises:
        HTTPException 400: If ``raw_bytes`` is empty.
        HTTPException 422: If the bytes cannot be decoded as a valid image.
        HTTPException 500: If writing the temp file fails.
    """
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image_bgr   = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(
            status_code=422,
            detail="Unable to decode image. Ensure the file is a valid JPEG or PNG.",
        )

    fd, temp_path = tempfile.mkstemp(suffix=".jpg")
    try:
        os.close(fd)
        cv2.imwrite(temp_path, image_bgr)
    except Exception as exc:
        os.unlink(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist image for inference: {exc}",
        )

    return temp_path, image_bgr


def _run_pipeline(
    temp_path: str,
    image_bgr: np.ndarray,
    target_distance_meters: float,
    click_value_moa: float,
    known_bullseye_diameter_inches: float,
) -> PipelineResult:
    """
    Execute the full vision + ballistics pipeline and return all computed values.

    Centralises the shared logic between /api/v1/process-target and
    /api/v1/overlay so each route only handles its own response format.

    Steps
    -----
    1. YOLO inference — ``vision.extract_bullet_coordinates``
    2. OpenCV bullseye detection — ``vision.find_bullseye``
    3. Legacy MOA path — ``ballistics.compute_mpi``, ``compute_scaling_factor``,
       ``compute_linear_deviations``, ``compute_scope_corrections``
    4. Zeroing-vector path — ``ballistics.calculate_zeroing_vectors`` with the
       fixed 640 × 640 frame centre as the baseline target reference.

    Args:
        temp_path:                      Absolute path to the persisted image for
                                        YOLO inference.
        image_bgr:                      Decoded BGR array for OpenCV operations.
        target_distance_meters:         Shooting distance in metres (used by both
                                        paths; converted to yards for the MOA path).
        click_value_moa:                MOA value per scope click (legacy path).
        known_bullseye_diameter_inches: Real-world bullseye diameter in inches
                                        (legacy path).

    Returns:
        A fully populated ``PipelineResult`` dataclass.

    Raises:
        HTTPException 503: If the YOLO model is not loaded.
        HTTPException 422: If no holes or no bullseye are detected.
    """
    # ------------------------------------------------------------------ YOLO
    try:
        detections = vision.extract_bullet_coordinates(temp_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Vision model unavailable: {exc}")

    if not detections:
        raise HTTPException(
            status_code=422,
            detail=(
                "No bullet holes were detected. "
                "Verify image quality and model weights."
            ),
        )

    # Convert detection dicts to (x, y) tuples for the legacy ballistic helpers.
    shot_centroids: list[tuple[float, float]] = [
        (d["x"], d["y"]) for d in detections
    ]

    # ------------------------------------------------- OpenCV bullseye ring
    x_center, y_center, pixel_diameter = vision.find_bullseye(image_bgr)
    if x_center is None:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not locate the bullseye ring. "
                "Ensure the target is fully visible and well-lit."
            ),
        )

    # ----------------------------------------------- Legacy MOA-based path
    distance_yards = target_distance_meters * _M_TO_YD

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

    # ----------------------------------------- New zeroing-vector path
    # Uses the fixed frame-centre reference point (320, 320) as the baseline
    # bullseye position so that calculate_zeroing_vectors operates on pure
    # pixel coordinates without requiring OpenCV ring detection.
    zeroing_result = ballistics.calculate_zeroing_vectors(
        shot_group=detections,
        target_center=_BULLSEYE_CENTRE,
        target_distance_meters=target_distance_meters,
    )

    return PipelineResult(
        detections=detections,
        shot_centroids=shot_centroids,
        x_mpi=x_mpi,
        y_mpi=y_mpi,
        x_center=x_center,
        y_center=y_center,
        pixel_diameter=pixel_diameter,
        delta_x_inches=delta_x_inches,
        delta_y_inches=delta_y_inches,
        correction_data=correction_data,
        zeroing_result=zeroing_result,
    )


def _build_zeroing_models(
    zeroing_result: dict,
) -> tuple[Optional[ZeroingSummary], Optional[ZeroingCorrections]]:
    """
    Convert the raw ``calculate_zeroing_vectors`` dict into validated Pydantic
    models, returning ``(None, None)`` on error or missing data.

    Args:
        zeroing_result: Direct return value from ``calculate_zeroing_vectors``.

    Returns:
        ``(ZeroingSummary, ZeroingCorrections)`` on success;
        ``(None, None)`` if the payload contains an error or is malformed.
    """
    if "error" in zeroing_result or "summary" not in zeroing_result:
        return None, None

    try:
        s = zeroing_result["summary"]
        c = zeroing_result["corrections"]

        summary = ZeroingSummary(
            total_hits=s["total_hits"],
            mpi=ZeroingMPI(x=s["mpi"]["x"], y=s["mpi"]["y"]),
            deviation_cm=ZeroingDeviationCm(
                horizontal=s["deviation_cm"]["horizontal"],
                vertical=s["deviation_cm"]["vertical"],
            ),
        )

        corrections = ZeroingCorrections(
            elevation=ZeroingCorrectionAxis(**c["elevation"]),
            windage=ZeroingCorrectionAxis(**c["windage"]),
            combined_instruction=c["combined_instruction"],
        )

        return summary, corrections

    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not build zeroing models from payload: %s", exc)
        return None, None


# ===========================================================================
# Routes
# ===========================================================================

@app.get("/api/v1/healthz", tags=["Health"])
async def health_check():
    return {"status": "ok", "service": "TargetRange-AI"}


@app.post(
    "/api/v1/process-target",
    response_model=ProcessTargetResponse,
    tags=["Analysis"],
    summary=(
        "Analyse a target photo and return consolidated scope correction data "
        "including MOA-based corrections and structured zeroing vectors."
    ),
)
async def process_target(
    image: UploadFile = File(
        ...,
        description="Target photo — JPEG or PNG.",
    ),
    target_distance_meters: float = Form(
        default=100.0,
        ge=1.0,
        description="Shooting distance in metres. Used by both the MOA path and the zeroing-vector engine.",
    ),
    click_value_moa: float = Form(
        default=0.25,
        gt=0.0,
        description="MOA value per scope click (legacy MOA path only).",
    ),
    known_bullseye_diameter_inches: float = Form(
        default=4.0,
        gt=0.0,
        description="Real-world diameter of the innermost target ring in inches (legacy MOA path only).",
    ),
):
    """
    Full tactical pipeline: image → YOLO detection → ballistics → consolidated JSON.

    The response payload contains two correction sections:

    * **corrections** — legacy MOA-based output using the physical bullseye ring
      as the centre reference (requires OpenCV ring detection to succeed).
    * **zeroing_summary / zeroing_corrections** — new structured output from
      ``calculate_zeroing_vectors``, always using pixel (320, 320) as the
      baseline bullseye centre for pure pixel-space calculations.
    """
    raw_bytes = await image.read()

    # Persist to temp file; MUST be cleaned up in finally block.
    temp_path, image_bgr = _decode_and_persist(raw_bytes)

    try:
        result = _run_pipeline(
            temp_path=temp_path,
            image_bgr=image_bgr,
            target_distance_meters=target_distance_meters,
            click_value_moa=click_value_moa,
            known_bullseye_diameter_inches=known_bullseye_diameter_inches,
        )

    except HTTPException as exc:
        # Return 503 (model missing) and 422 (no holes) as structured JSON
        # payloads so callers always receive a consistent response envelope.
        if exc.status_code == 503:
            return ProcessTargetResponse(
                status="error",
                message=exc.detail,
                total_shots_detected=0,
            )
        if exc.status_code == 422 and "No bullet holes" in exc.detail:
            return ProcessTargetResponse(
                status="no_shots",
                message=exc.detail,
                total_shots_detected=0,
            )
        raise

    finally:
        # Guarantee temp file removal regardless of success or failure.
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as exc:
                logger.warning("Could not remove temp file '%s': %s", temp_path, exc)

    # Build validated Pydantic models from the zeroing-vector payload.
    zeroing_summary, zeroing_corrections = _build_zeroing_models(result.zeroing_result)

    logger.info(
        "Analysis complete — %d shot(s) | MPI=(%.1f, %.1f) | "
        "windage %+d click(s) %s | elevation %+d click(s) %s | "
        "zeroing: %s",
        len(result.shot_centroids),
        result.x_mpi, result.y_mpi,
        result.correction_data["windage"]["clicks"],
        result.correction_data["windage"]["direction"],
        result.correction_data["elevation"]["clicks"],
        result.correction_data["elevation"]["direction"],
        result.zeroing_result.get("corrections", {}).get("combined_instruction", "n/a"),
    )

    return ProcessTargetResponse(
        status="success",
        message=(
            f"Analysis complete. {len(result.shot_centroids)} shot(s) processed "
            f"at {target_distance_meters:.0f} m."
        ),
        total_shots_detected=len(result.shot_centroids),
        # Legacy MOA output
        mpi_coordinates=Coordinates(
            x=round(result.x_mpi, 2),
            y=round(result.y_mpi, 2),
        ),
        target_center_coordinates=Coordinates(
            x=round(result.x_center, 2),
            y=round(result.y_center, 2),
        ),
        raw_deviations_inches=DeviationsInches(
            horizontal=round(result.delta_x_inches, 4),
            vertical=round(result.delta_y_inches, 4),
        ),
        corrections=Corrections(
            elevation=AdjustmentAxis(**result.correction_data["elevation"]),
            windage=AdjustmentAxis(**result.correction_data["windage"]),
        ),
        # New structured zeroing-vector output
        zeroing_summary=zeroing_summary,
        zeroing_corrections=zeroing_corrections,
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
    target_distance_meters: float = Form(
        default=100.0,
        ge=1.0,
        description="Shooting distance in metres.",
    ),
    click_value_moa: float = Form(
        default=0.25,
        gt=0.0,
        description="MOA value per scope click.",
    ),
    known_bullseye_diameter_inches: float = Form(
        default=4.0,
        gt=0.0,
        description="Real-world diameter of the innermost target ring in inches.",
    ),
):
    raw_bytes = await image.read()
    temp_path, image_bgr = _decode_and_persist(raw_bytes)

    try:
        result = _run_pipeline(
            temp_path=temp_path,
            image_bgr=image_bgr,
            target_distance_meters=target_distance_meters,
            click_value_moa=click_value_moa,
            known_bullseye_diameter_inches=known_bullseye_diameter_inches,
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError as exc:
                logger.warning("Could not remove temp file '%s': %s", temp_path, exc)

    label = overlay.build_correction_label(result.correction_data)

    png_bytes = overlay.render_overlay(
        image_bgr=image_bgr,
        shot_centroids=result.shot_centroids,
        mpi=(result.x_mpi, result.y_mpi),
        target_center=(result.x_center, result.y_center),
        correction_label=label,
    )

    logger.info(
        "Overlay rendered — %d shot(s), label='%s'.",
        len(result.shot_centroids),
        label,
    )

    return Response(content=png_bytes, media_type="image/png")
