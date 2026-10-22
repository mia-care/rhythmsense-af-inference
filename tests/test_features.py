"""Unit tests for SRS-ALG-001 and SRS-ALG-002."""

import numpy as np
import pytest

from rhythmsense_inference.features import (
    compute_features,
    detect_beats,
    inter_beat_intervals,
    shannon_entropy,
)

from .conftest import FS, synth_ppg


def test_detects_every_beat_in_clean_regular_signal(regular_ibi):
    signal = synth_ppg(regular_ibi)
    peaks = detect_beats(signal, FS)
    assert abs(peaks.size - (regular_ibi.size + 1)) <= 1


def test_ibi_matches_ground_truth_within_one_sample(regular_ibi):
    peaks = detect_beats(synth_ppg(regular_ibi), FS)
    ibi = inter_beat_intervals(peaks, FS)
    assert np.median(np.abs(ibi[: regular_ibi.size] - regular_ibi[: ibi.size])) < 1000 / FS


def test_non_physiological_intervals_are_discarded():
    peaks = np.array([0, 5, 70, 140])  # 5 samples apart = 78 ms -> > 220 bpm
    ibi = inter_beat_intervals(peaks, FS)
    assert ibi.size == 2


def test_rmssd_is_higher_for_irregular_rhythm(regular_ibi, irregular_ibi):
    assert compute_features(irregular_ibi).rmssd_ms > 5 * compute_features(regular_ibi).rmssd_ms


def test_entropy_is_zero_for_empty_tachogram():
    assert shannon_entropy(np.empty(0)) == 0.0


def test_features_reject_too_short_tachogram():
    with pytest.raises(ValueError):
        compute_features(np.array([800.0, 810.0]))


def test_mean_heart_rate_is_computed_from_intervals():
    features = compute_features(np.full(10, 1000.0))
    assert features.mean_hr_bpm == pytest.approx(60.0)
