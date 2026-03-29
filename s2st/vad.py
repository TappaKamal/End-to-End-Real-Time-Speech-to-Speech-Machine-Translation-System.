"""Voice activity detection using Silero VAD (streaming, probability + state machine)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch


def load_silero_vad(device: Optional[torch.device] = None) -> Tuple[Any, Any]:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, utils


class StreamingVAD:
    """
    Feed fixed-size float32 chunks (512 samples @ 16 kHz ≈ 32 ms).
    Returns a completed utterance when speech ends after trailing silence.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        threshold: float = 0.5,
        min_silence_duration_ms: int = 400,
        min_speech_duration_ms: int = 200,
        speech_pad_ms: int = 200,
        device: Optional[str] = None,
    ) -> None:
        self._dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _ = load_silero_vad(self._dev)
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.speech_pad_ms = speech_pad_ms

        self._frame_samples = 512 if sampling_rate == 16000 else 256
        self._silence_frames_needed = max(
            1, int(min_silence_duration_ms / (1000.0 * self._frame_samples / sampling_rate))
        )
        self._min_speech_frames = max(
            1, int(min_speech_duration_ms / (1000.0 * self._frame_samples / sampling_rate))
        )
        self._pad_frames = max(0, int(speech_pad_ms / (1000.0 * self._frame_samples / sampling_rate)))

        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buf: list[np.ndarray] = []
        self._pre_buf: list[np.ndarray] = []  # ring of last N frames for padding before speech

    def reset(self) -> None:
        self._in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._buf.clear()
        self._pre_buf.clear()

    def _prob(self, chunk: np.ndarray) -> float:
        x = torch.from_numpy(chunk.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(self._dev)
        with torch.no_grad():
            p = self.model(x, self.sampling_rate)
        if isinstance(p, torch.Tensor):
            p = float(p.detach().cpu().reshape(-1)[0].item())
        return float(p)

    def process_chunk(self, chunk_float32: np.ndarray) -> Optional[np.ndarray]:
        if chunk_float32.size != self._frame_samples:
            raise ValueError(
                f"Expected {self._frame_samples} samples per chunk, got {chunk_float32.size}"
            )
        chunk_float32 = np.clip(chunk_float32.astype(np.float32), -1.0, 1.0)
        p = self._prob(chunk_float32)

        # keep short pre-roll for natural starts
        self._pre_buf.append(chunk_float32.copy())
        max_pre = self._pad_frames + 2
        if len(self._pre_buf) > max_pre:
            self._pre_buf.pop(0)

        if not self._in_speech:
            if p >= self.threshold:
                self._in_speech = True
                self._speech_frames = 1
                self._silence_frames = 0
                self._buf = list(self._pre_buf[:-1]) if len(self._pre_buf) > 1 else []
                self._buf.append(chunk_float32.copy())
            return None

        # in speech
        self._buf.append(chunk_float32.copy())
        if p >= self.threshold:
            self._speech_frames += 1
            self._silence_frames = 0
        else:
            self._silence_frames += 1

        if self._silence_frames >= self._silence_frames_needed:
            if self._speech_frames >= self._min_speech_frames:
                out = np.concatenate(self._buf, axis=0)
            else:
                out = None
            self._in_speech = False
            self._speech_frames = 0
            self._silence_frames = 0
            self._buf.clear()
            return out
        return None
