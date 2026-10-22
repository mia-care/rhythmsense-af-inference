"""HTTP API of the AF Inference Engine.

Traces to: SRS-API-010 (classify endpoint), SRS-API-011 (model version
disclosure), SRS-SEC-004 (no PHI in logs).
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier import MODEL_VERSION, classify_window
from .features import compute_features, detect_beats, inter_beat_intervals

logger = logging.getLogger("rhythmsense.inference")

app = FastAPI(title="RhythmSense AF Inference Engine", version=MODEL_VERSION)


class ClassifyRequest(BaseModel):
    window_id: str = Field(min_length=1, max_length=64)
    sampling_rate_hz: float = Field(gt=10, le=1000)
    samples: list[float] = Field(min_length=100)
    signal_quality: float = Field(ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    window_id: str
    result: str
    probability: float | None
    model_version: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    signal = np.asarray(req.samples, dtype=float)
    peaks = detect_beats(signal, req.sampling_rate_hz)
    ibi = inter_beat_intervals(peaks, req.sampling_rate_hz)
    try:
        features = compute_features(ibi)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result = classify_window(features, req.signal_quality)
    logger.info("window classified", extra={"window_id": req.window_id, "result": result.result.value})
    return ClassifyResponse(
        window_id=req.window_id,
        result=result.result.value,
        probability=result.probability,
        model_version=result.model_version,
    )
