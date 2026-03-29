"""End-to-end real-time S2ST: capture → NS → VAD → ASR → LID → translate → TTS."""

from __future__ import annotations

import os
import tempfile
from typing import Callable, Optional

import numpy as np

from s2st.asr import StreamingASR
from s2st.audio_stream import AudioStreamer
from s2st.config import PipelineConfig
from s2st.lid import LIDResult, lid_from_asr_result
from s2st.noise_suppression import NoiseSuppressor
from s2st.translation import get_translator
from s2st.results import S2STResult
from s2st.tts import pick_voice, play_mp3, synthesize_to_mp3
from s2st.vad import StreamingVAD


class RealtimeS2STPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        a = self.cfg.audio
        self._ns = NoiseSuppressor(a.sample_rate, self.cfg.noise)
        self._vad = StreamingVAD(
            sampling_rate=a.sample_rate,
            threshold=self.cfg.vad.threshold,
            min_silence_duration_ms=self.cfg.vad.min_silence_duration_ms,
            speech_pad_ms=self.cfg.vad.speech_pad_ms,
            device=self.cfg.asr.device,
        )
        self._asr = StreamingASR(self.cfg.asr)
        self._translator = get_translator(
            self.cfg.translation.model_name,
            self.cfg.translation.device,
        )

    def process_utterance(
        self,
        utterance_float32: np.ndarray,
        on_event: Optional[Callable[[dict], None]] = None,
        play_audio: bool = True,
    ) -> S2STResult:
        """Run ASR → LID → translation → TTS for one speech segment."""
        cleaned = self._ns.process_utterance(utterance_float32)
        lang_hint = self.cfg.source_lang
        asr = self._asr.transcribe(cleaned, sample_rate=self.cfg.audio.sample_rate, language=lang_hint)
        lid: LIDResult = lid_from_asr_result(asr)
        src = lid.language
        tgt = self.cfg.target_lang
        text_src = asr.text.strip()
        if not text_src:
            if on_event:
                on_event({"type": "empty", "lid": lid})
            return S2STResult(
                ok=False,
                message="No speech detected or empty transcription. Try speaking louder or longer.",
                language=lid.language,
                language_confidence=lid.confidence,
            )

        if on_event:
            on_event(
                {
                    "type": "asr",
                    "lid": lid,
                    "text_source": text_src,
                    "segments": asr.segments,
                }
            )

        text_tgt = self._translator.translate(text_src, src, tgt)
        if on_event:
            on_event({"type": "translation", "text_target": text_tgt, "target_lang": tgt})

        voice = pick_voice(tgt, self.cfg.tts.voice)
        path = os.path.join(tempfile.gettempdir(), "s2st_last.mp3")
        synthesize_to_mp3(text_tgt, voice, out_path=path)
        if on_event:
            on_event({"type": "tts", "path": path, "voice": voice})
        if play_audio:
            play_mp3(path)
        return S2STResult(
            ok=True,
            message="OK",
            language=lid.language,
            language_confidence=lid.confidence,
            text_source=text_src,
            text_target=text_tgt,
            tts_audio_path=path,
            voice=voice,
            segments=list(asr.segments),
        )

    def run_forever(self, on_event: Optional[Callable[[dict], None]] = None) -> None:
        """Blocking loop: microphone → NS → VAD → full pipeline."""
        a = self.cfg.audio
        streamer = AudioStreamer(
            sample_rate=a.sample_rate,
            frame_samples=a.frame_samples,
            channels=a.channels,
        )
        streamer.start()
        try:
            for frame in streamer.frames():
                denoised = self._ns.process_frame(frame)
                utt = self._vad.process_chunk(denoised)
                if utt is None:
                    continue
                self.process_utterance(utt, on_event=on_event)
        finally:
            streamer.stop()
