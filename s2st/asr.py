"""Automatic speech recognition with faster-whisper (timestamps + sentence boundaries)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from s2st.config import ASRConfig


@dataclass
class WordSpan:
    start: float
    end: float
    word: str


@dataclass
class SegmentSpan:
    start: float
    end: float
    text: str
    words: List[WordSpan] = field(default_factory=list)


@dataclass
class ASRResult:
    text: str
    language: str
    language_probability: float
    segments: List[SegmentSpan]


class StreamingASR:
    def __init__(self, cfg: ASRConfig) -> None:
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return ASRResult(
                text="",
                language=language or "en",
                language_probability=1.0,
                segments=[],
            )
        segments_iter, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self.cfg.beam_size,
            vad_filter=False,
            word_timestamps=True,
        )
        segs: List[SegmentSpan] = []
        texts: List[str] = []
        for s in segments_iter:
            words: List[WordSpan] = []
            if s.words:
                for w in s.words:
                    words.append(
                        WordSpan(start=w.start, end=w.end, word=w.word)
                    )
            segs.append(
                SegmentSpan(start=s.start, end=s.end, text=s.text.strip(), words=words)
            )
            texts.append(s.text)
        full = "".join(texts).strip()
        lang = info.language or (language or "en")
        prob = float(getattr(info, "language_probability", 1.0) or 1.0)
        return ASRResult(
            text=full,
            language=lang,
            language_probability=prob,
            segments=segs,
        )
