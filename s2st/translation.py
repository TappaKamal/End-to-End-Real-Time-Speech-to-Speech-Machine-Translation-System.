"""Multilingual translation (M2M100) with Whisper → M2M100 language code mapping."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Whisper ISO 639-1 codes → M2M100 codes (subset; fallback to English)
_WHISPER_TO_M2M100 = {
    "en": "en",
    "zh": "zh",
    "de": "de",
    "es": "es",
    "ru": "ru",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
    "pt": "pt",
    "pl": "pl",
    "nl": "nl",
    "ar": "ar",
    "tr": "tr",
    "sv": "sv",
    "da": "da",
    "fi": "fi",
    "no": "no",
    "cs": "cs",
    "hu": "hu",
    "ro": "ro",
    "bg": "bg",
    "el": "el",
    "uk": "uk",
    "hi": "hi",
    "vi": "vi",
    "id": "id",
    "ms": "ms",
    "th": "th",
    "he": "he",
    "ko": "ko",
}


def _normalize_lang(code: str) -> str:
    c = (code or "en").strip().lower()
    if "-" in c:
        c = c.split("-", 1)[0]
    return c


def to_m2m100_lang(whisper_lang: str) -> str:
    w = _normalize_lang(whisper_lang)
    return _WHISPER_TO_M2M100.get(w, "en")


class M2M100Translator:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.device = device
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    @torch.inference_mode()
    def translate(
        self,
        text: str,
        source_lang_whisper: str,
        target_lang_iso: str,
        max_length: int = 512,
    ) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        src = to_m2m100_lang(source_lang_whisper)
        tgt = to_m2m100_lang(target_lang_iso)
        if src == tgt:
            return text
        self.tokenizer.src_lang = src
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        tgt_id = self.tokenizer.get_lang_id(tgt)
        generated = self.model.generate(
            **encoded,
            forced_bos_token_id=int(tgt_id),
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
        )
        out = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out.strip()


@lru_cache(maxsize=1)
def get_translator(model_name: str, device: str) -> M2M100Translator:
    return M2M100Translator(model_name, device=device)
