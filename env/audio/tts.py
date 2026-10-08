"""The caller's voice (roadmap P2-01): local, free TTS with a roster of distinct voices, chosen deterministically per
task so the same seed always speaks with the same voice.

Backends: `KokoroTTS` (open weights, 82M, runs on the Mac; `pip install -e tests/behavioral[voice]`) and `ToneTTS`,
a dependency-free stand-in for tests that encodes the text's length into a tone burst. Both return float32 mono at
their native rate; `env/audio/degrade.py` takes it from there.

    tts = load_tts()                       # Kokoro when installed, else the tone backend
    samples, sr = tts.synthesize("Hi, I'd like to book a follow-up visit.", voice=voice_for(seed))
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

# Kokoro's English voices — a spread of gender and accent (a = American, b = British), which is what a clinic's
# phone line hears. Twelve is the plan's floor; the roster is the thing a seed indexes, so order matters: keep
# appending, never reorder.
KOKORO_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "am_eric", "am_liam",
    "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
)


@dataclass(frozen=True)
class Voice:
    id: str
    gender: str  # f | m
    accent: str  # american | british

    @classmethod
    def parse(cls, voice_id: str) -> "Voice":
        return cls(id=voice_id, gender=voice_id[1], accent={"a": "american", "b": "british"}.get(voice_id[0], "other"))


def voice_for(seed: int, roster: Tuple[str, ...] = KOKORO_VOICES) -> str:
    """The voice a task's caller speaks with — a deterministic draw per seed, spread over the roster."""
    return random.Random(f"ehr-env-voice:{seed}").choice(roster)


class TTS(Protocol):
    name: str
    sample_rate: int

    def synthesize(self, text: str, voice: str = KOKORO_VOICES[0], speed: float = 1.0) -> Tuple[np.ndarray, int]: ...


class ToneTTS:
    """Dependency-free stand-in: a 440 Hz burst whose length tracks the text (≈ 60 ms per word), so the degradation
    chain and the audio plumbing can be tested without a model. Not speech; the ASR round trip skips it."""

    name = "tone"
    sample_rate = 16000

    def synthesize(self, text: str, voice: str = KOKORO_VOICES[0], speed: float = 1.0) -> Tuple[np.ndarray, int]:
        words = max(1, len(text.split()))
        seconds = max(0.3, 0.06 * words / max(speed, 0.1))
        t = np.arange(int(self.sample_rate * seconds)) / self.sample_rate
        pitch = 180.0 if Voice.parse(voice).gender == "m" else 260.0
        env = np.minimum(1.0, np.minimum(t / 0.02, (seconds - t) / 0.02))
        return (0.3 * env * np.sin(2 * np.pi * pitch * t)).astype(np.float32), self.sample_rate


class KokoroTTS:
    """Kokoro-82M through its KPipeline (24 kHz). The pipeline loads lazily on the first call."""

    name = "kokoro"
    sample_rate = 24000

    def __init__(self, lang_code: str = "a") -> None:
        self._lang = lang_code
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self._lang, repo_id="hexgrad/Kokoro-82M")
        return self._pipeline

    def synthesize(self, text: str, voice: str = KOKORO_VOICES[0], speed: float = 1.0) -> Tuple[np.ndarray, int]:
        pipeline = self._load()
        chunks = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=voice, speed=speed):
            if audio is None:
                continue
            arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
            chunks.append(arr.astype(np.float32))
        if not chunks:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        return np.concatenate(chunks), self.sample_rate


def kokoro_available() -> bool:
    try:
        import kokoro  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — any import failure (missing deps, no torch) means "not here"
        return False


def load_tts(prefer: Optional[str] = None) -> TTS:
    """Kokoro when it is installed (or when `prefer="kokoro"` insists), the tone backend otherwise."""
    if prefer == "tone":
        return ToneTTS()
    if prefer == "kokoro" or (prefer is None and kokoro_available()):
        return KokoroTTS()
    return ToneTTS()


__all__ = ["KOKORO_VOICES", "KokoroTTS", "TTS", "ToneTTS", "Voice", "kokoro_available", "load_tts", "voice_for"]
