"""The audio-native caller (roadmap P3-02) offline: a scripted audio-chat client stands in for vLLM-Omni; the clip it
returns goes through the normal channel, and a server failure falls back to the TTS caller.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_omni_caller.py -v
"""
from __future__ import annotations

import base64
from types import SimpleNamespace

import numpy as np
import pytest

from env.audio.channel import VoiceChannel, wav_bytes
from env.audio.omni_caller import CALLER_SYSTEM, OmniCallerTTS, caller_tts_for
from env.audio.tts import ToneTTS
from env.tasks import generate

pytestmark = [pytest.mark.covers("rl-env-voice")]


class FakeAudioChat:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise ConnectionError("server down")
        t = np.arange(24000) / 24000
        wav = wav_bytes((0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32), 24000)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="", audio=SimpleNamespace(data=base64.b64encode(wav).decode())))])


def test_omni_caller_speaks_the_line_in_persona_and_rides_the_channel():
    task = generate("booking", 5, tier=3)
    client = FakeAudioChat()
    tts = OmniCallerTTS("http://fake/v1", "Qwen/Qwen3-Omni-30B-A3B-Instruct", persona=task.persona, goal=task.goal, client=client)
    samples, sr = tts.synthesize("It's 1964-04-11.")
    assert sr == 24000 and len(samples) == 24000 and tts.fallbacks == 0
    req = client.calls[0]
    assert req["modalities"] == ["text", "audio"] and req["audio"]["format"] == "wav"
    assert req["messages"][0]["content"] == CALLER_SYSTEM.format(persona=task.persona, goal=task.goal) and "It's 1964-04-11." in req["messages"][1]["content"]
    clip = VoiceChannel.for_task(task.seed, task.tier, tts=tts).render("It's 1964-04-11.")
    assert clip["tts"] == "omni-caller" and clip["sample_rate"] == 8000 and clip["seconds"] > 0.9  # down the phone line like any clip


def test_omni_caller_falls_back_to_the_tts_caller_when_the_server_fails():
    tts = OmniCallerTTS("http://fake/v1", "m", client=FakeAudioChat(fail=True), fallback=ToneTTS())
    samples, sr = tts.synthesize("Yes, that works.", voice="af_heart")
    assert sr == 16000 and len(samples) > 0 and tts.fallbacks == 1
    assert isinstance(caller_tts_for(generate("cancel", 10), None, None), ToneTTS) or caller_tts_for(generate("cancel", 10), None, None).name in ("tone", "kokoro")
    assert isinstance(caller_tts_for(generate("cancel", 10), "http://fake/v1", "m", client=FakeAudioChat()), OmniCallerTTS)
