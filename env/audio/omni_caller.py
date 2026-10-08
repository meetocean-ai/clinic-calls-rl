"""Audio-native caller (roadmap P3-02): a frozen Qwen3-Omni (or any OpenAI-compatible chat model with speech out)
voices the caller's line itself — persona and goal in the system prompt, no TTS — then the line goes down the P2-02
phone line like every other clip. It is a `TTS` for `VoiceChannel`: `synthesize(text, voice)` returns samples, so the
env, the disfluencies and the degradation are untouched; the fallback when the server is unreachable is the P2-01
TTS caller (Kokoro / tone).

    channel = VoiceChannel.for_task(seed, tier, tts=OmniCallerTTS(endpoint, model, persona=task.persona, goal=task.goal))

Verified against a live vLLM-Omni at P2-11; offline test with a scripted client in `env/audio/test_omni_caller.py`.
"""
from __future__ import annotations

import base64
from typing import Any, Optional, Tuple

import numpy as np

from .tts import TTS, load_tts

CALLER_SYSTEM = ("You are a patient calling a chiropractic clinic's front desk. Persona: {persona}. Your goal on this call: {goal}. "
                 "Say EXACTLY the line you are given, as this person would say it on the phone — same words, natural delivery — "
                 "and nothing else.")


class OmniCallerTTS:
    """`/v1/chat/completions` with `modalities=["text","audio"]` and `audio={voice, format: "wav"}` (the OpenAI audio-chat
    shape vLLM-Omni serves): the reply's `message.audio.data` is base64 WAV."""

    name = "omni-caller"

    def __init__(self, endpoint: str, model: str, *, persona: str = "polite", goal: str = "", api_key: str = "", voice: str = "Chelsie",
                 fallback: Optional[TTS] = None, client=None, timeout_s: float = 120.0) -> None:
        self.endpoint, self.model, self.persona, self.goal, self.voice = endpoint.rstrip("/"), model, persona, goal, voice
        self.fallback = fallback
        self.fallbacks = 0
        if client is None:
            from openai import OpenAI

            client = OpenAI(base_url=self.endpoint, api_key=api_key or "none", timeout=timeout_s)
        self.client = client

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str, voice: Optional[str] = None, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        from .channel import wav_to_samples

        try:
            resp = self.client.chat.completions.create(
                model=self.model, modalities=["text", "audio"], audio={"voice": voice or self.voice, "format": "wav"},
                messages=[{"role": "system", "content": CALLER_SYSTEM.format(persona=self.persona, goal=self.goal or "as given")},
                          {"role": "user", "content": f"Say this line: {text}"}], temperature=0.7)
            data = getattr(getattr(resp.choices[0].message, "audio", None), "data", None)
            if not data:
                raise RuntimeError("no audio in the reply")
            samples, sr = wav_to_samples(base64.b64decode(data))
            return np.asarray(samples, dtype=np.float32), sr
        except Exception:  # noqa: BLE001 — the caller must always have a voice: fall back to the TTS caller
            self.fallbacks += 1
            if self.fallback is None:
                self.fallback = load_tts()
            return self.fallback.synthesize(text, voice=voice or "af_heart")


def caller_tts_for(task, endpoint: Optional[str], model: Optional[str], **kw: Any) -> TTS:
    """The audio-native caller when a server is given, else the TTS caller."""
    if endpoint and model:
        return OmniCallerTTS(endpoint, model, persona=task.persona, goal=task.goal, **kw)
    return load_tts()


__all__ = ["CALLER_SYSTEM", "OmniCallerTTS", "caller_tts_for"]
