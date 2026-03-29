"""Shared helpers for converting browser/file audio to model input."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample as scipy_resample


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.clip(audio.astype(np.float32), -1.0, 1.0)


def resample_to_sr(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return to_mono_float32(audio)
    n = int(len(audio) * dst_sr / src_sr)
    if n < 1:
        return np.zeros(0, dtype=np.float32)
    out = scipy_resample(to_mono_float32(audio), n).astype(np.float32)
    return np.clip(out, -1.0, 1.0)


def gradio_audio_to_16k_mono(audio: tuple | None, target_sr: int = 16000) -> np.ndarray | None:
    """
    Gradio Audio returns (sample_rate, numpy_array) or None.
    """
    if audio is None:
        return None
    sr, data = audio
    if data is None or len(data) == 0:
        return None
    data = np.asarray(data)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype in (np.float32, np.float64):
        data = np.clip(data.astype(np.float32), -1.0, 1.0)
    else:
        data = data.astype(np.float32)
    return resample_to_sr(data, int(sr), target_sr)
