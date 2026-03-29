"""Continuous microphone capture with chunked streaming and ring buffering."""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Callable, Generator, Optional

import numpy as np
import sounddevice as sd


class RingBuffer:
    """Fixed-capacity float32 ring buffer for streaming audio."""

    def __init__(self, capacity_samples: int, channels: int = 1) -> None:
        self._capacity = capacity_samples
        self._channels = channels
        self._buf = np.zeros((capacity_samples, channels), dtype=np.float32)
        self._write = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    def write(self, chunk: np.ndarray) -> None:
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        n = chunk.shape[0]
        for i in range(n):
            self._buf[self._write] = chunk[i]
            self._write = (self._write + 1) % self._capacity
            if self._size < self._capacity:
                self._size += 1
            else:
                self._size = self._capacity

    def read_all(self) -> np.ndarray:
        if self._size == 0:
            return np.zeros((0, self._channels), dtype=np.float32)
        if self._size < self._capacity:
            start = 0
            out = self._buf[: self._write].copy()
            return out
        idx = (self._write - self._size) % self._capacity
        if idx + self._size <= self._capacity:
            return self._buf[idx : idx + self._size].copy()
        part1 = self._buf[idx:]
        part2 = self._buf[: self._write]
        return np.vstack([part1, part2])


class AudioStreamer:
    """
    Low-latency capture: delivers fixed-size frames (e.g. 512 samples ≈ 32 ms @ 16 kHz).
    Uses an internal queue so processing can lag briefly without dropping input.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_samples: int = 512,
        channels: int = 1,
        queue_max_frames: int = 256,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self.channels = channels
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_max_frames)
        self._stream: Optional[sd.InputStream] = None
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        if status:
            pass
        mono = indata[:, 0].astype(np.float32).copy()
        # Split into frame_samples chunks
        offset = 0
        while offset + self.frame_samples <= len(mono):
            chunk = mono[offset : offset + self.frame_samples]
            offset += self.frame_samples
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(chunk)
                except queue.Full:
                    pass

    def start(self) -> None:
        self._stop.clear()
        self._stream = sd.InputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.frame_samples,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Blocking generator yielding float32 mono frames of length frame_samples."""
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=0.25)
            except queue.Empty:
                continue


def int16_to_float32(pcm: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1]."""
    if pcm.dtype == np.float32:
        return np.clip(pcm, -1.0, 1.0)
    return np.clip(pcm.astype(np.float32) / 32768.0, -1.0, 1.0)


def resample_simple(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear interpolation resampling (lightweight; use torchaudio for quality)."""
    if src_sr == dst_sr:
        return x.astype(np.float32)
    duration = len(x) / src_sr
    new_len = int(duration * dst_sr)
    if new_len < 2:
        return x.astype(np.float32)
    t_old = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    t_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)
