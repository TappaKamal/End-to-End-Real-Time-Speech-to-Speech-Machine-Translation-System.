"""Structured output from the S2ST pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class S2STResult:
    ok: bool
    message: str = ""
    language: str = ""
    language_confidence: float = 0.0
    text_source: str = ""
    text_target: str = ""
    tts_audio_path: Optional[str] = None
    voice: str = ""
    segments: List[Any] = field(default_factory=list)
