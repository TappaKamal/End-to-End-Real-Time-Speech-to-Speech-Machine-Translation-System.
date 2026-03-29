"""
Gradio web UI for Speech-to-Speech Translation (connects to the same backend as main.py).

Run:
  .venv\\Scripts\\python ui_app.py
Then open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
from typing import Any, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from s2st.audio_utils import gradio_audio_to_16k_mono
from s2st.config import PipelineConfig
from s2st.pipeline import RealtimeS2STPipeline

# --- Target languages (ISO 639-1) for translation + TTS voice mapping ---
TARGET_CHOICES: list[tuple[str, str]] = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese (Mandarin)", "zh"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Turkish", "tr"),
    ("Dutch", "nl"),
    ("Polish", "pl"),
    ("Vietnamese", "vi"),
    ("Indonesian", "id"),
    ("Thai", "th"),
    ("Ukrainian", "uk"),
]

SOURCE_CHOICES: list[tuple[str, Optional[str]]] = [
    ("Auto-detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese", "zh"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
]

WHISPER_MODELS = ["tiny", "base", "small", "medium"]


class _PipelineHolder:
    """Lazy-load ASR/M2M100; rebuild when model or device changes."""

    def __init__(self) -> None:
        self._pipeline: Optional[RealtimeS2STPipeline] = None
        self._key: Optional[tuple[str, str, str]] = None

    def get(
        self,
        whisper_model: str,
        device: str,
        compute_type: str,
    ) -> RealtimeS2STPipeline:
        key = (whisper_model, device, compute_type)
        if self._pipeline is None or self._key != key:
            cfg = PipelineConfig()
            cfg.asr.model_size = whisper_model
            cfg.asr.device = device
            cfg.asr.compute_type = compute_type
            cfg.translation.device = device
            self._pipeline = RealtimeS2STPipeline(cfg)
            self._key = key
        return self._pipeline


_holder = _PipelineHolder()


def _resolve_device(choice: str) -> str:
    if choice == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return "cpu"


def _compute_type_for_device(device: str) -> str:
    return "float16" if device == "cuda" else "int8"


def translate_audio(
    audio: Any,
    target_label: str,
    source_label: str,
    whisper_model: str,
    device_choice: str,
) -> Tuple[str, str, str, str, Optional[str], Optional[str]]:
    """
    Gradio returns: status, lang line, source text, translated text, path for audio.
    """
    if audio is None:
        return (
            "❌ No audio. Use the microphone or upload a file.",
            "",
            "",
            "",
            None,
            "",
        )

    audio_16k = gradio_audio_to_16k_mono(audio, target_sr=16000)
    if audio_16k is None or len(audio_16k) < 1600:
        return (
            "❌ Audio too short. Record at least ~0.1 s of speech.",
            "",
            "",
            "",
            None,
            "",
        )

    device = _resolve_device(device_choice)
    compute_type = _compute_type_for_device(device)

    tgt = next((c for c in TARGET_CHOICES if c[0] == target_label), TARGET_CHOICES[0])[1]
    src_map = {s[0]: s[1] for s in SOURCE_CHOICES}
    src = src_map.get(source_label)

    pipeline = _holder.get(whisper_model, device, compute_type)
    pipeline.cfg.target_lang = tgt
    pipeline.cfg.source_lang = src

    try:
        result = pipeline.process_utterance(np.asarray(audio_16k, dtype=np.float32), play_audio=False)
    except Exception as exc:  # noqa: BLE001
        return (
            f"❌ Error: {exc}",
            "",
            "",
            "",
            None,
            "",
        )

    if not result.ok:
        return (
            f"⚠️ {result.message}",
            f"**Detected language:** {result.language} (confidence: {result.language_confidence:.2f})"
            if result.language
            else "",
            "",
            "",
            None,
            "",
        )

    status = (
        f"✅ Done (Whisper **{whisper_model}**, device **{device}**). "
        f"TTS voice: `{result.voice}`"
    )
    lang_line = (
        f"**Detected language:** `{result.language}` — "
        f"**confidence:** {result.language_confidence:.3f}"
    )
    return (
        status,
        lang_line,
        result.text_source,
        result.text_target,
        result.tts_audio_path,
        result.voice,
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="S2ST — Speech to Speech Translation") as demo:
        gr.Markdown(
            """
### Real-time Speech → Speech Translation
Record audio or upload **WAV/FLAC**. The pipeline runs **noise suppression → VAD (via Silero in live mode) → ASR (Whisper) → LID → M2M100 translation → TTS (Edge)**.

**First run** downloads models (Whisper, M2M100, Silero); allow several minutes on first use.
"""
        )
        with gr.Row():
            with gr.Column(scale=1):
                target_dd = gr.Dropdown(
                    [x[0] for x in TARGET_CHOICES],
                    value="English",
                    label="Translate to (target language)",
                )
                source_dd = gr.Dropdown(
                    [x[0] for x in SOURCE_CHOICES],
                    value="Auto-detect",
                    label="Source language (ASR)",
                )
                whisper_dd = gr.Dropdown(
                    WHISPER_MODELS,
                    value="small",
                    label="Whisper model (smaller = faster, lower quality)",
                )
                device_dd = gr.Radio(
                    ["cpu", "cuda"],
                    value="cpu",
                    label="Device (CUDA only if GPU + drivers are installed)",
                )
            with gr.Column(scale=2):
                audio_in = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Microphone or file upload",
                )
                run_btn = gr.Button("Translate speech", variant="primary")

        status_md = gr.Markdown("")
        lang_md = gr.Markdown("")
        src_txt = gr.Textbox(label="Transcription (source language)", lines=3)
        tgt_txt = gr.Textbox(label="Translation (target language)", lines=3)
        voice_txt = gr.Textbox(label="TTS voice used", lines=1)
        audio_out = gr.Audio(
            label="Synthesized speech (translated text)",
            type="filepath",
        )

        run_btn.click(
            fn=translate_audio,
            inputs=[audio_in, target_dd, source_dd, whisper_dd, device_dd],
            outputs=[status_md, lang_md, src_txt, tgt_txt, audio_out, voice_txt],
        )

        gr.Markdown(
            """
---
**CLI:** `python main.py --target-lang fr` (live mic) or `python main.py --wav clip.wav --target-lang en`.  
**Repo:** ensure `README.md` setup (venv, `pip install -r requirements.txt`).
"""
        )
    return demo


def main() -> None:
    p = argparse.ArgumentParser(description="Gradio UI for S2ST")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (use 0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary public Gradio link (requires internet)",
    )
    args = p.parse_args()

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
