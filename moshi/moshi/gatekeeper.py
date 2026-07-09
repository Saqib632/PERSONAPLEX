# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""Best-effort language/topic gatekeeper for a PersonaFlex-style session.

PersonaPlex has no built-in ASR of the user's speech and no VAD/turn
boundary detection (the model only ever produces a text stream for its own
generated speech). This module bolts on a side pipeline: VAD to find
utterance boundaries in the user's raw mic audio, local ASR to transcribe
each utterance, and an external text classifier to flag disallowed
language/topic. Because generation is causal and already streaming by the
time a verdict comes back, enforcement is best-effort: on a FAIL verdict we
splice a fixed rejection clip into the *output* stream going forward, but
any of the model's real response already sent for that turn cannot be
recalled. The model's own generation state is never reset, so the
conversation continues normally afterwards.
"""

from __future__ import annotations

import asyncio
import os
import wave
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL_LANGUAGE = "FAIL_LANGUAGE"
    FAIL_TOPIC = "FAIL_TOPIC"


REJECTION_TEXT: dict[Verdict, str] = {
    Verdict.FAIL_LANGUAGE: (
        "Sorry, I don't understand your language. "
        "Please write your question fully in English."
    ),
    Verdict.FAIL_TOPIC: "Sorry, I only provide information related to the U.S. trucking industry.",
}

GATE_SYSTEM_PROMPT = """You are a strict two-gate validator for PersonaFlex, a voice assistant \
that ONLY serves the U.S. trucking industry.

You will be given a single transcribed user utterance. Reply with EXACTLY one \
of these tokens and nothing else: PASS, FAIL_LANGUAGE, FAIL_TOPIC.

- FAIL_LANGUAGE if the utterance is not entirely in English (any non-English \
words, code-switching, or non-Latin script all fail). Trucking abbreviations \
like FMCSA, ELD, HOS, CDL do not count as foreign.
- Otherwise FAIL_TOPIC if it is not about the U.S. trucking industry (FMCSA/DOT \
regulation, ELD/HOS, dispatching, freight/logistics, fleet maintenance, IFTA, \
insurance, CDL licensing, CSA scores, inspections, owner-operator business, or \
similar). Trucking in other countries only passes if tied to U.S. cross-border \
operations.
- Otherwise PASS.
"""


def _resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Cheap linear-interpolation resampler. Quality is not critical here:
    it's used for VAD framing and for short, fixed TTS rejection clips."""
    pcm = np.asarray(pcm, dtype=np.float32)
    if src_sr == dst_sr or pcm.size == 0:
        return pcm
    duration = pcm.size / src_sr
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=pcm.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, pcm).astype(np.float32)


class Vad:
    """Loads the shared silero-vad model once; safe to reuse (read-only
    inference) across many per-connection `_UtteranceTracker` instances."""

    SR = 16000
    WINDOW = 512  # required fixed window size for silero-vad at 16kHz

    def __init__(self) -> None:
        model, _utils = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
        model.eval()
        self._model = model

    def speech_prob(self, window: np.ndarray) -> float:
        with torch.no_grad():
            return float(self._model(torch.from_numpy(window), self.SR).item())


class _UtteranceTracker:
    """Per-connection streaming utterance-boundary state machine, built on
    top of a shared `Vad` model instance."""

    def __init__(
        self,
        vad: Vad,
        source_sample_rate: int,
        speech_threshold: float = 0.5,
        end_silence_ms: float = 600.0,
        min_speech_ms: float = 200.0,
    ) -> None:
        self._vad = vad
        self.source_sample_rate = source_sample_rate
        self._threshold = speech_threshold
        window_ms = Vad.WINDOW / Vad.SR * 1000
        self._end_silence_windows = max(1, round(end_silence_ms / window_ms))
        self._min_speech_windows = max(1, round(min_speech_ms / window_ms))
        self._resample_buf = np.zeros(0, dtype=np.float32)
        self._speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._utterance_chunks: list[np.ndarray] = []

    def push(self, pcm: np.ndarray) -> Optional[np.ndarray]:
        """Feed one chunk of raw source-rate mono PCM. Returns the
        accumulated utterance PCM when an end-of-speech boundary is
        crossed, else None."""
        resampled = _resample_linear(pcm, self.source_sample_rate, Vad.SR)
        self._resample_buf = np.concatenate([self._resample_buf, resampled])

        started = False
        ended_utterance: Optional[np.ndarray] = None
        while len(self._resample_buf) >= Vad.WINDOW:
            window = self._resample_buf[: Vad.WINDOW]
            self._resample_buf = self._resample_buf[Vad.WINDOW:]
            is_speech = self._vad.speech_prob(window) >= self._threshold
            if is_speech:
                self._speech_run += 1
                self._silence_run = 0
                if not self._speaking and self._speech_run >= self._min_speech_windows:
                    self._speaking = True
                    started = True
            else:
                self._silence_run += 1
                self._speech_run = 0
                if self._speaking and self._silence_run >= self._end_silence_windows:
                    self._speaking = False
                    if self._utterance_chunks:
                        ended_utterance = np.concatenate(self._utterance_chunks)
                    self._utterance_chunks = []

        if self._speaking or started:
            self._utterance_chunks.append(np.asarray(pcm, dtype=np.float32))

        return ended_utterance


class Asr:
    """Local speech-to-text used to transcribe a completed user utterance
    before it's handed to the classifier."""

    def __init__(self, device: str = "cpu", model_size: str = "small") -> None:
        from faster_whisper import WhisperModel

        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, pcm: np.ndarray, source_sample_rate: int) -> str:
        audio = _resample_linear(pcm, source_sample_rate, 16000)
        segments, _info = self._model.transcribe(audio, beam_size=1)
        return " ".join(segment.text for segment in segments).strip()


class Classifier:
    """Fast external text classifier for Gate 1 (language) + Gate 2
    (U.S. trucking topic). Only ever returns a terse enum verdict."""

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; required when --gatekeeper is enabled."
            )
        self._client = anthropic.AsyncAnthropic()
        self._model = model

    async def classify(self, utterance_text: str) -> Verdict:
        if not utterance_text.strip():
            # Nothing intelligible was transcribed (e.g. a VAD false
            # positive on noise) -- don't flag a turn we can't evaluate.
            return Verdict.PASS
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=8,
            system=GATE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": utterance_text}],
        )
        raw = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        for verdict in Verdict:
            if verdict.value in raw:
                return verdict
        return Verdict.PASS


def _load_wav_mono_float32(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
    if sample_width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM WAV, got {sample_width * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


@dataclass
class RejectionClip:
    pcm: np.ndarray  # float32 mono, already resampled to the target rate
    text: str


class RejectionClips:
    """Loads and caches the two fixed rejection audio clips, resampled once
    to the model's output sample rate."""

    def __init__(self, lang_clip_path: Path, topic_clip_path: Path, target_sample_rate: int) -> None:
        lang_pcm, lang_sr = _load_wav_mono_float32(lang_clip_path)
        topic_pcm, topic_sr = _load_wav_mono_float32(topic_clip_path)
        self._clips = {
            Verdict.FAIL_LANGUAGE: RejectionClip(
                pcm=_resample_linear(lang_pcm, lang_sr, target_sample_rate),
                text=REJECTION_TEXT[Verdict.FAIL_LANGUAGE],
            ),
            Verdict.FAIL_TOPIC: RejectionClip(
                pcm=_resample_linear(topic_pcm, topic_sr, target_sample_rate),
                text=REJECTION_TEXT[Verdict.FAIL_TOPIC],
            ),
        }

    def get(self, verdict: Verdict) -> RejectionClip:
        return self._clips[verdict]


class GatekeeperSession:
    """Per-connection gate: taps the input PCM stream for VAD/ASR/classify,
    and filters the output PCM/text stream to splice in a rejection clip
    when a turn is flagged."""

    def __init__(self, resources: "GatekeeperResources", source_sample_rate: int, log: Callable) -> None:
        self._resources = resources
        self._tracker = _UtteranceTracker(resources.vad, source_sample_rate)
        self._log = log
        self._tasks: set[asyncio.Task] = set()
        self._mute_clip: Optional[np.ndarray] = None
        self._mute_cursor = 0
        self._mute_text: Optional[str] = None

    def on_input_chunk(self, pcm: np.ndarray) -> None:
        utterance = self._tracker.push(pcm)
        if utterance is not None and utterance.size > 0:
            task = asyncio.create_task(self._run_gate(utterance))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run_gate(self, utterance_pcm: np.ndarray) -> None:
        try:
            text = await asyncio.to_thread(
                self._resources.asr.transcribe, utterance_pcm, self._tracker.source_sample_rate
            )
            verdict = await self._resources.classifier.classify(text)
        except Exception as exc:  # gatekeeper failures must never break the call
            self._log("warning", f"gatekeeper check failed, passing turn through: {exc}")
            return
        if verdict is not Verdict.PASS:
            clip = self._resources.clips.get(verdict)
            self._mute_clip = clip.pcm
            self._mute_cursor = 0
            self._mute_text = clip.text

    def is_muting(self) -> bool:
        return self._mute_clip is not None

    def pending_rejection_text(self) -> Optional[str]:
        """Returns the rejection text exactly once, the first time it's
        polled after a mute begins."""
        text, self._mute_text = self._mute_text, None
        return text

    def filter_output(self, main_pcm: np.ndarray, frame_size: int) -> np.ndarray:
        """Called once per decoded output audio frame. Returns the model's
        own PCM unchanged, or a same-sized slice of the active rejection
        clip while muting."""
        if self._mute_clip is None:
            return main_pcm
        end = self._mute_cursor + frame_size
        out = self._mute_clip[self._mute_cursor:end]
        self._mute_cursor = end
        if self._mute_cursor >= len(self._mute_clip):
            self._mute_clip = None
        if len(out) < frame_size:
            out = np.pad(out, (0, frame_size - len(out)))
        return out.astype(np.float32)


@dataclass
class GatekeeperResources:
    """Shared, process-wide gatekeeper resources loaded once at server
    startup and reused (read-only inference) across all connections."""

    vad: Vad
    asr: Asr
    classifier: Classifier
    clips: RejectionClips

    @classmethod
    def load(
        cls,
        asr_device: str,
        lang_clip_path: Path,
        topic_clip_path: Path,
        target_sample_rate: int,
    ) -> "GatekeeperResources":
        return cls(
            vad=Vad(),
            asr=Asr(device=asr_device),
            classifier=Classifier(),
            clips=RejectionClips(lang_clip_path, topic_clip_path, target_sample_rate),
        )

    def new_session(self, source_sample_rate: int, log: Callable) -> GatekeeperSession:
        return GatekeeperSession(self, source_sample_rate, log)
