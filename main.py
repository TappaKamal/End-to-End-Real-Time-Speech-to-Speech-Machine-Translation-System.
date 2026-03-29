"""
Real-time Speech-to-Speech Translation CLI.

Pipeline: Mic → stream → noise suppression → VAD → ASR → LID → M2M100 → edge-tts → playback.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample as scipy_resample

from s2st.config import PipelineConfig
from s2st.pipeline import RealtimeS2STPipeline


def _load_wav_mono(path: str, target_sr: int) -> np.ndarray:
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        n = int(len(data) * target_sr / sr)
        data = scipy_resample(data, n).astype(np.float32)
    return np.clip(data, -1.0, 1.0)


def main() -> None:
    p = argparse.ArgumentParser(description="Real-time Speech-to-Speech Translation")
    p.add_argument(
        "--target-lang",
        default="en",
        help="Target language ISO 639-1 code for translation/TTS (e.g. en, fr, es)",
    )
    p.add_argument(
        "--source-lang",
        default=None,
        help="Optional fixed source language (skips auto LID for ASR)",
    )
    p.add_argument("--whisper-model", default="small", help="faster-whisper model size name")
    p.add_argument(
        "--device",
        default="cpu",
        help="torch/faster-whisper device: cpu or cuda",
    )
    p.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (e.g. int8, float16 on GPU)",
    )
    p.add_argument(
        "--wav",
        default=None,
        metavar="PATH",
        help="Process one WAV/FLAC file instead of live mic (16 kHz mono recommended)",
    )
    args = p.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; using CPU.", file=sys.stderr)
        dev = "cpu"

    cfg = PipelineConfig()
    cfg.target_lang = args.target_lang
    cfg.source_lang = args.source_lang
    cfg.asr.model_size = args.whisper_model
    cfg.asr.device = dev
    cfg.asr.compute_type = args.compute_type
    cfg.translation.device = dev

    pipeline = RealtimeS2STPipeline(cfg)

    def on_event(ev: dict) -> None:
        t = ev.get("type")
        if t == "asr":
            lid = ev["lid"]
            print(
                json.dumps(
                    {
                        "event": "asr+lid",
                        "language": lid.language,
                        "confidence": round(lid.confidence, 4),
                        "text_source": ev["text_source"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        elif t == "translation":
            print(
                json.dumps(
                    {"event": "translation", "text_target": ev["text_target"]},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        elif t == "tts":
            print(
                json.dumps({"event": "tts", "voice": ev["voice"], "path": ev["path"]}),
                flush=True,
            )
        elif t == "empty":
            print(json.dumps({"event": "empty_utterance"}), flush=True)

    if args.wav:
        audio = _load_wav_mono(args.wav, pipeline.cfg.audio.sample_rate)
        pipeline.process_utterance(audio, on_event=on_event)
        return

    print(
        "Listening. Speak clearly; pause ~0.5s after each phrase. Ctrl+C to exit.",
        file=sys.stderr,
    )
    try:
        pipeline.run_forever(on_event=on_event)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
