"""Irregular-rhythm classifier and notification confirmation logic.

Software item: AF Inference Engine (IEC 62304 class B).
Traces to: SRS-ALG-003 (classification), SRS-ALG-004 (confirmation),
SRS-ALG-005 (abstention on low signal quality).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

from .features import RhythmFeatures

MODEL_VERSION = "1.3.0"

# Locked logistic-regression weights (model card MC-RS-001, trained on the
# 2025 retrospective dataset; see the Algorithm Change Protocol before
# changing any value).
_WEIGHTS = {
    "rmssd_ms": 0.031,
    "pnn50": 4.2,
    "cv": 18.5,
    "shannon_entropy": 0.95,
}
_BIAS = -9.8

DECISION_THRESHOLD = 0.62
MIN_SIGNAL_QUALITY = 0.70
CONFIRMATION_WINDOWS = 6
CONFIRMATION_POSITIVES = 5


class WindowResult(str, Enum):
    IRREGULAR = "irregular"
    REGULAR = "regular"
    UNCLASSIFIABLE = "unclassifiable"


@dataclass(frozen=True)
class Classification:
    result: WindowResult
    probability: float | None
    model_version: str = MODEL_VERSION


def irregularity_probability(features: RhythmFeatures) -> float:
    z = _BIAS + sum(w * getattr(features, name) for name, w in _WEIGHTS.items())
    return 1.0 / (1.0 + math.exp(-z))


def classify_window(features: RhythmFeatures, signal_quality: float) -> Classification:
    """Classify one tachogram window.

    Windows whose signal-quality index is below MIN_SIGNAL_QUALITY must not be
    classified (SRS-ALG-005).
    """
    if signal_quality < MIN_SIGNAL_QUALITY:
        return Classification(WindowResult.UNCLASSIFIABLE, None)
    if not 40 <= features.mean_hr_bpm <= 150:
        return Classification(WindowResult.UNCLASSIFIABLE, None)
    p = irregularity_probability(features)
    result = WindowResult.IRREGULAR if p >= DECISION_THRESHOLD else WindowResult.REGULAR
    return Classification(result, p)


class NotificationConfirmer:
    """Raise an irregular-rhythm notification only when at least 5 of the last
    6 classifiable windows are irregular (SRS-ALG-004).

    KNOWN ANOMALY RS-ANOM-014: unclassifiable windows are appended to the
    history as non-irregular instead of being skipped, which delays (but never
    falsely raises) a notification. Tracked in the problem-resolution log.
    """

    def __init__(self) -> None:
        self._history: deque[bool] = deque(maxlen=CONFIRMATION_WINDOWS)

    def push(self, classification: Classification) -> bool:
        self._history.append(classification.result is WindowResult.IRREGULAR)
        return (
            len(self._history) == CONFIRMATION_WINDOWS
            and sum(self._history) >= CONFIRMATION_POSITIVES
        )

    def reset(self) -> None:
        self._history.clear()
