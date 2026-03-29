"""Text-to-speech via edge-tts (multilingual neural voices, low latency)."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional

import edge_tts

# Shortlist of Edge voices by BCP-47-ish locale (extend as needed)
_DEFAULT_VOICES = {
    "en": "en-US-JennyNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "pl": "pl-PL-ZofiaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "ar": "ar-SA-ZariyahNeural",
    "hi": "hi-IN-SwaraNeural",
    "tr": "tr-TR-EmelNeural",
    "nl": "nl-NL-ColetteNeural",
    "sv": "sv-SE-SofieNeural",
    "da": "da-DK-ChristelNeural",
    "fi": "fi-FI-NooraNeural",
    "cs": "cs-CZ-VlastaNeural",
    "uk": "uk-UA-PolinaNeural",
    "vi": "vi-VN-HoaiMyNeural",
    "id": "id-ID-GadisNeural",
    "th": "th-TH-PremwadeeNeural",
}


def pick_voice(target_lang_iso: str, override: Optional[str] = None) -> str:
    if override:
        return override
    code = (target_lang_iso or "en").strip().lower()
    if "-" in code:
        code = code.split("-", 1)[0]
    return _DEFAULT_VOICES.get(code, "en-US-JennyNeural")


async def _synthesize_file_async(text: str, voice: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(path)


def synthesize_to_mp3(text: str, voice: str, out_path: Optional[str] = None) -> str:
    """Write MP3 to path (or temp file) and return path."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
    asyncio.run(_synthesize_file_async(text, voice, out_path))
    return out_path


def play_mp3(path: str) -> None:
    """Play MP3 using system default (Windows-friendly)."""
    try:
        import playsound
    except ImportError as e:
        raise RuntimeError("Install playsound for audio playback: pip install playsound==1.2.2") from e
    playsound.playsound(path, block=True)
