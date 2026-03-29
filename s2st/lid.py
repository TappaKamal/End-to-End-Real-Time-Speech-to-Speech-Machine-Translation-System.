"""Language identification on speech segments (Whisper-based, with confidence)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LIDResult:
    language: str
    confidence: float


def lid_from_asr_result(asr: Any) -> LIDResult:
    """Use ASR output (same Whisper pass) for LID to avoid duplicate inference."""
    return LIDResult(language=asr.language, confidence=asr.language_probability)


class LanguageIdentifier:
    """
    Standalone LID using the same Whisper / faster-whisper backend as ASR.
    Prefer `lid_from_asr_result` when ASR already ran on the segment.
    """

    def __init__(self, whisper_model) -> None:
        self._model = whisper_model

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> LIDResult:
        if audio.size == 0:
            return LIDResult(language="en", confidence=0.0)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        segments, info = self._model.transcribe(
            audio,
            language=None,
            task="transcribe",
            beam_size=1,
            vad_filter=False,
        )
        _ = list(segments)
        lang = info.language or "en"
        conf = float(getattr(info, "language_probability", 1.0) or 1.0)
        return LIDResult(language=lang, confidence=conf)
