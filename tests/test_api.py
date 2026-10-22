"""Unit tests for SRS-API-010 and SRS-API-011."""

from fastapi.testclient import TestClient

from rhythmsense_inference.api import app

from .conftest import FS, synth_ppg

client = TestClient(app)


def test_health_reports_model_version():
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_version": "1.3.0"}


def test_classify_returns_result_for_valid_window(irregular_ibi):
    payload = {
        "window_id": "w-001",
        "sampling_rate_hz": FS,
        "samples": synth_ppg(irregular_ibi).tolist(),
        "signal_quality": 0.92,
    }
    response = client.post("/v1/classify", json=payload)
    assert response.status_code == 200
    assert response.json()["result"] == "irregular"


def test_classify_rejects_out_of_range_signal_quality():
    payload = {"window_id": "w", "sampling_rate_hz": FS, "samples": [0.0] * 200, "signal_quality": 1.4}
    assert client.post("/v1/classify", json=payload).status_code == 422


def test_classify_rejects_flat_signal():
    payload = {"window_id": "w", "sampling_rate_hz": FS, "samples": [0.0] * 2000, "signal_quality": 0.9}
    assert client.post("/v1/classify", json=payload).status_code == 422
