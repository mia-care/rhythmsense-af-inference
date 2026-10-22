"""Inter-beat interval extraction and heart-rhythm irregularity features.

Software item: AF Inference Engine (IEC 62304 class B).
Traces to: SRS-ALG-001 (peak detection), SRS-ALG-002 (IBI features).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# Physiological bounds for an inter-beat interval: 30-220 bpm.
MIN_IBI_MS = 60_000 / 220
MAX_IBI_MS = 60_000 / 30


@dataclass(frozen=True)
class RhythmFeatures:
    """Irregularity features computed over one tachogram window."""

    mean_hr_bpm: float
    rmssd_ms: float
    pnn50: float
    cv: float
    shannon_entropy: float
    n_beats: int


def bandpass(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 4.0) -> np.ndarray:
    """Zero-phase band-pass filter tuned to the cardiac band of a PPG signal."""
    if signal.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    nyq = fs / 2.0
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_beats(signal: np.ndarray, fs: float) -> np.ndarray:
    """Return the sample indices of systolic peaks in a PPG window."""
    filtered = bandpass(signal, fs)
    min_distance = int(fs * MIN_IBI_MS / 1000)
    prominence = 0.3 * np.std(filtered)
    peaks, _ = find_peaks(filtered, distance=max(min_distance, 1), prominence=prominence)
    return peaks


def inter_beat_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    """Convert peak indices to inter-beat intervals in ms, dropping non-physiological ones."""
    if peaks.size < 2:
        return np.empty(0)
    ibi = np.diff(peaks) / fs * 1000.0
    return ibi[(ibi >= MIN_IBI_MS) & (ibi <= MAX_IBI_MS)]


def shannon_entropy(ibi: np.ndarray, bins: int = 16) -> float:
    """Shannon entropy (bits) of the IBI histogram; high for chaotic rhythms."""
    if ibi.size == 0:
        return 0.0
    hist, _ = np.histogram(ibi, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def compute_features(ibi: np.ndarray) -> RhythmFeatures:
    """Compute the irregularity feature vector for a tachogram."""
    if ibi.size < 3:
        raise ValueError("at least 3 inter-beat intervals are required")
    diffs = np.diff(ibi)
    return RhythmFeatures(
        mean_hr_bpm=float(60_000 / ibi.mean()),
        rmssd_ms=float(np.sqrt(np.mean(diffs**2))),
        pnn50=float(np.mean(np.abs(diffs) > 50.0)),
        cv=float(ibi.std() / ibi.mean()),
        shannon_entropy=shannon_entropy(ibi),
        n_beats=int(ibi.size + 1),
    )
