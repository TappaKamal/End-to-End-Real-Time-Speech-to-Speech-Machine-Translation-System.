"""Noise suppression: bandpass + optional noisereduce on utterance segments."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from s2st.config import NoiseSuppressionConfig

try:
    import noisereduce as nr
except ImportError:
    nr = None  # type: ignore[assignment]


def _butter_bandpass(low_hz: float, high_hz: float, sr: int, order: int = 4):
    nyq = 0.5 * sr
    low = max(low_hz / nyq, 1e-5)
    high = min(high_hz / nyq, 0.999)
    return butter(order, [low, high], btype="band", output="sos")


class NoiseSuppressor:
    """
    Real-time path: bandpass filtering per frame (fast).
    After an utterance is collected, optional noisereduce improves ASR robustness.
    """

    def __init__(self, sr: int, cfg: NoiseSuppressionConfig) -> None:
        self.sr = sr
        self.cfg = cfg
        self._sos = _butter_bandpass(cfg.bandpass_low_hz, cfg.bandpass_high_hz, sr)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """frame: float32 mono, shape (n,)"""
        x = frame.astype(np.float32)
        if x.size < 15:
            return x
        try:
            y = sosfiltfilt(self._sos, x)
        except ValueError:
            y = x
        return np.clip(y, -1.0, 1.0).astype(np.float32)

    def process_utterance(self, utterance: np.ndarray) -> np.ndarray:
        """Full segment before ASR: bandpass + optional noisereduce."""
        x = utterance.astype(np.float32).reshape(-1)
        if x.size == 0:
            return x
        y = self.process_frame(x)
        if float(np.std(y)) < 1e-8:
            return y
        if self.cfg.use_noisereduce_on_utterance and nr is not None:
            y = nr.reduce_noise(
                y=y,
                sr=self.sr,
                stationary=self.cfg.noisereduce_stationary,
                prop_decrease=0.85,
            )
            y = np.asarray(y, dtype=np.float32)
        return np.clip(y, -1.0, 1.0)
