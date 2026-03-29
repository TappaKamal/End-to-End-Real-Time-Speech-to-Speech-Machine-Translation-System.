"""Default configuration for capture, VAD, ASR, translation, and TTS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    # Silero VAD expects 512 samples @ 16 kHz per step
    frame_samples: int = 512
    channels: int = 1
    dtype: str = "float32"


@dataclass
class NoiseSuppressionConfig:
    bandpass_low_hz: float = 80.0
    bandpass_high_hz: float = 7600.0
    use_noisereduce_on_utterance: bool = True
    noisereduce_stationary: bool = False


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_duration_ms: int = 400
    speech_pad_ms: int = 400


@dataclass
class ASRConfig:
    model_size: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 5


@dataclass
class TranslationConfig:
    model_name: str = "facebook/m2m100_418M"
    device: str = "cpu"


@dataclass
class TTSConfig:
    voice: Optional[str] = None  # edge-tts voice; None = pick by target lang


@dataclass
class PipelineConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    noise: NoiseSuppressionConfig = field(default_factory=NoiseSuppressionConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    target_lang: str = "en"
    source_lang: Optional[str] = None  # None = auto (Whisper LID)
