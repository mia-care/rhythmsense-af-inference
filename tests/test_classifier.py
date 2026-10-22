"""Unit tests for SRS-ALG-003, SRS-ALG-004 and SRS-ALG-005."""

from rhythmsense_inference.classifier import (
    Classification,
    NotificationConfirmer,
    WindowResult,
    classify_window,
)
from rhythmsense_inference.features import compute_features


def test_regular_rhythm_is_classified_regular(regular_ibi):
    result = classify_window(compute_features(regular_ibi), signal_quality=0.95)
    assert result.result is WindowResult.REGULAR


def test_irregular_rhythm_is_classified_irregular(irregular_ibi):
    result = classify_window(compute_features(irregular_ibi), signal_quality=0.95)
    assert result.result is WindowResult.IRREGULAR
    assert result.probability is not None and result.probability >= 0.62


def test_low_signal_quality_window_is_not_classified(irregular_ibi):
    result = classify_window(compute_features(irregular_ibi), signal_quality=0.4)
    assert result.result is WindowResult.UNCLASSIFIABLE
    assert result.probability is None


def test_out_of_range_heart_rate_is_not_classified():
    import numpy as np

    result = classify_window(compute_features(np.full(20, 350.0)), signal_quality=0.99)
    assert result.result is WindowResult.UNCLASSIFIABLE


def test_classification_discloses_model_version(regular_ibi):
    assert classify_window(compute_features(regular_ibi), 0.9).model_version == "1.3.0"


IRR = Classification(WindowResult.IRREGULAR, 0.9)
REG = Classification(WindowResult.REGULAR, 0.1)
UNC = Classification(WindowResult.UNCLASSIFIABLE, None)


def test_notification_requires_five_of_six_irregular_windows():
    confirmer = NotificationConfirmer()
    outcomes = [confirmer.push(c) for c in [IRR, IRR, REG, IRR, IRR, IRR]]
    assert outcomes[-1] is True
    assert not any(outcomes[:-1])


def test_no_notification_with_four_of_six_irregular_windows():
    confirmer = NotificationConfirmer()
    outcomes = [confirmer.push(c) for c in [IRR, REG, IRR, REG, IRR, IRR]]
    assert not any(outcomes)


def test_unclassifiable_windows_are_skipped_by_confirmation():
    """SRS-ALG-004: only classifiable windows count towards confirmation.

    Fails because of known anomaly RS-ANOM-014.
    """
    confirmer = NotificationConfirmer()
    outcomes = [confirmer.push(c) for c in [IRR, IRR, UNC, IRR, IRR, IRR, REG]]
    assert outcomes[-1] is True


def test_reset_clears_history():
    confirmer = NotificationConfirmer()
    for _ in range(5):
        confirmer.push(IRR)
    confirmer.reset()
    assert confirmer.push(IRR) is False
