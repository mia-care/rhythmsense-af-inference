import numpy as np
import pytest

FS = 64.0


def synth_ppg(ibi_ms: np.ndarray, fs: float = FS, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """Synthetic PPG: one Gaussian-shaped pulse per beat at the given intervals."""
    rng = np.random.default_rng(seed)
    beat_times = np.concatenate([[0.5], 0.5 + np.cumsum(ibi_ms) / 1000.0])
    duration = beat_times[-1] + 1.0
    t = np.arange(0, duration, 1 / fs)
    signal = np.zeros_like(t)
    for bt in beat_times:
        signal += np.exp(-((t - bt) ** 2) / (2 * 0.06**2))
    return signal + rng.normal(0, noise, t.size)


@pytest.fixture
def regular_ibi() -> np.ndarray:
    rng = np.random.default_rng(1)
    return 850 + rng.normal(0, 12, 60)


@pytest.fixture
def irregular_ibi() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.uniform(420, 1150, 60)
