"""Audio mode of the environment (roadmap P2-03): caller lines arrive as telephone audio next to the text, the mode is
in method_version, the voice/line specs in the manifest. Uses the dependency-free tone backend so no model is needed.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_channel.py -v
"""
from __future__ import annotations

import base64
import os

import numpy as np
import pytest

from env.audio.channel import VoiceChannel, wav_bytes, wav_to_samples
from env.audio.tts import ToneTTS
from env.versioning import method_version

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def test_channel_renders_decodable_telephony_audio_with_the_spoken_text():
    ch = VoiceChannel.for_task(seed=5, tier=4, tts=ToneTTS())
    clip = ch.render("Hi, I need to move my appointment on Tuesday, January 15 at 11 AM to another day.")
    samples, sr = wav_to_samples(base64.b64decode(clip["wav_b64"]))
    assert sr == clip["sample_rate"] == 8000 and abs(len(samples) / sr - clip["seconds"]) < 0.01
    assert clip["voice"] == ch.voice and clip["tts"] == "tone"
    assert "Tuesday, January 15 at 11 AM" in clip["spoken"]  # the facts survive the disfluencies
    assert ch.disfluency.fillers >= 2 and clip["spoken"] != "Hi, I need to move my appointment on Tuesday, January 15 at 11 AM to another day."
    spec = ch.spec()
    assert spec["degradation"]["snr_db"] <= 15.0 and spec["disfluency"]["fillers"] >= 2
    again = VoiceChannel.for_task(seed=5, tier=4, tts=ToneTTS()).render("Hi, I need to move my appointment on Tuesday, January 15 at 11 AM to another day.")
    assert again["wav_b64"] == clip["wav_b64"], "same seed, same tier, same text → the same bytes"


def test_wav_roundtrip_keeps_samples():
    x = (0.4 * np.sin(np.linspace(0, 200, 8000))).astype(np.float32)
    y, sr = wav_to_samples(wav_bytes(x, 8000))
    assert sr == 8000 and np.max(np.abs(y - x)) < 1e-3


def test_method_version_marks_audio_mode():
    assert method_version(audio=True) == method_version() + "+audio"
    assert method_version(judge_model="j", audio=True).endswith("+judge-j+audio")


@local_only
async def test_env_audio_mode_attaches_audio_to_every_caller_line_and_stamps_the_manifest():
    from adapters.medplum import MedplumAdapter
    from env.env import EhrSchedulingEnv, SchedulingAction

    medplum = await MedplumAdapter.from_env()
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    try:
        obs = await env.reset(seed=5, family="booking", tier=3)
        assert obs["patient_text"] and obs["audio"]["sample_rate"] == 8000 and obs["audio"]["seconds"] > 0
        heard = await env.step_async(SchedulingAction(kind="say", text="Could I have your date of birth, please?"))
        assert heard["patient_text"] and heard["audio"]["spoken"] and heard["audio"]["wav_b64"]
        tool = await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": env.task.patient["dob"]}))
        assert "audio" not in tool  # tool results are not spoken
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["done"]
        m = env.manifest()
        assert m["method_version"].endswith("+audio") and m["voice"]["voice"] == env.channel.voice and m["voice"]["tts"] == "tone"
        assert m["voice"]["degradation"]["seed"] == 5
        text_only = EhrSchedulingEnv(medplum, scripted_caller=True)
        await text_only.reset(seed=5, family="booking", tier=3)
        try:
            assert "audio" not in text_only.initial_observation and text_only.manifest()["voice"] is None
            assert not text_only.manifest()["method_version"].endswith("+audio")
        finally:
            await text_only.close()
    finally:
        await env.close()
        await medplum.close()


@local_only
async def test_server_serves_audio_when_asked(monkeypatch):
    from env.client import EhrAction
    from env.server import EhrOpenEnv

    monkeypatch.setenv("EHR_ENV_AUDIO", "1")
    monkeypatch.setenv("EHR_ENV_TTS", "tone")
    env = EhrOpenEnv(scripted_caller=True)
    try:
        first = await env.reset_async(seed=10, family="cancel", tier=1)
        assert first.audio and first.audio["sample_rate"] == 8000 and first.patient_text
        heard = await env.step_async(EhrAction(kind="say", text="Your date of birth, please?"))
        assert heard.audio and heard.audio["spoken"]
        assert env.state.method_version.endswith("+audio")
    finally:
        await env.aclose()
