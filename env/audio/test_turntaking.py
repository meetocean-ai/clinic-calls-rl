"""Agent-side speech hooks (roadmap P2-05): a policy that answers with audio is transcribed for the judges, its clips
land in the episode record as Opus (pruned to the newest runs), and turn-taking is read off the two audio streams.
No model: tone audio and an ear that reads the clip's text stand in for Kokoro + Whisper.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_turntaking.py -v
"""
from __future__ import annotations

import base64
import os
from pathlib import Path

import numpy as np
import pytest

from env.audio.channel import wav_bytes
from env.audio.tts import ToneTTS
from env.audio.turntaking import classify_onset, decode_audio, encode_opus, pauses, turn_taking
from env.env import EhrSchedulingEnv, SchedulingAction, prune_agent_audio

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def _tone(seconds: float, sr: int = 8000, hz: float = 220.0, level: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _with_pause(sr: int = 8000) -> np.ndarray:
    """1.0 s of speech, a 0.5 s pause, 1.0 s of speech."""
    return np.concatenate([_tone(1.0, sr), np.zeros(int(0.5 * sr), dtype=np.float32), _tone(1.0, sr)])


def test_pauses_are_found_inside_the_line_not_at_its_edges():
    x = _with_pause()
    assert pauses(x, 8000) == [(1.0, 1.5)]
    assert pauses(np.concatenate([np.zeros(4000, dtype=np.float32), _tone(1.0)]), 8000) == []  # leading silence is not a pause
    assert pauses(_tone(2.0), 8000) == [] and pauses(np.zeros(10, dtype=np.float32), 8000) == []


def test_onset_classification():
    p = [(1.0, 1.5)]
    assert classify_onset(None, 2.5, p, 3.0) == "after" and classify_onset(0.4, 2.5, p, 3.0) == "after"
    assert classify_onset(-1.3, 2.5, p, 3.0) == "takeover"  # began at 1.2 s, inside the caller's pause
    assert classify_onset(-2.0, 2.5, p, 3.0) == "interruption"  # began at 0.5 s, over the caller's speech
    assert classify_onset(-2.0, 2.5, p, 0.6) == "backchannel"  # short overlapping clip


def test_opus_roundtrip_keeps_the_clip():
    x = _tone(1.5)
    data = encode_opus(x, 8000)
    y, sr = decode_audio(data)
    assert data[:4] == b"OggS" and sr == 8000 and abs(len(y) / sr - 1.5) < 0.1 and len(data) < len(wav_bytes(x, 8000)) / 3


def test_turn_taking_summary_from_a_recorded_episode():
    caller = {"audio": {"seconds": 2.5, "pauses": [(1.0, 1.5)]}}
    steps = [
        {"action": {"kind": "say", "agent_audio": {"turn": 1, "seconds": 2.0, "onset_s": None, "wall_latency_s": 0.8}}, "observation": {"turn": 1, **caller}},
        {"action": {"kind": "tool", "tool_name": "verify_patient"}, "observation": {"turn": 1, "audio": None}},
        {"action": {"kind": "say", "agent_audio": {"turn": 2, "seconds": 3.0, "onset_s": -1.3, "wall_latency_s": 0.1}}, "observation": {"turn": 2, **caller}},
        {"action": {"kind": "say", "agent_audio": {"turn": 3, "seconds": 0.5, "onset_s": -2.0, "wall_latency_s": 0.1}}, "observation": {"turn": 3, **caller}},
        {"action": {"kind": "say", "agent_audio": {"turn": 4, "seconds": 4.0, "onset_s": -2.0, "wall_latency_s": 0.1}}, "observation": {"turn": 4, "audio": {"seconds": 1.0, "pauses": []}}},
        {"action": {"kind": "say", "text": "text only"}, "observation": {"turn": 5}},
    ]
    s = turn_taking(steps)
    assert s["agent_turns_with_audio"] == 4 and s["waited"] == 1 and s["takeovers_in_pause"] == 1 and s["backchannels"] == 1 and s["interruptions"] == 1
    assert s["barge_ins"] == 3 and s["response_latency_s"]["p50"] == -1.3 and s["response_latency_s"]["max"] == 0.8
    assert s["overlap_s"] == round(1.3 + 0.5 + 2.0, 3) and s["agent_talk_s"] == 9.5 and s["caller_talk_s"] == 8.5 and s["talk_ratio"] == round(9.5 / 8.5, 3)
    assert [r["kind"] for r in s["turns"]] == ["after", "takeover", "backchannel", "interruption"]
    assert turn_taking([{"action": {"kind": "say", "text": "hi"}, "observation": {}}]) is None


def test_prune_keeps_the_newest_runs(tmp_path):
    for run in ("20261001-000000", "20261002-000000", "20261003-000000"):
        d = tmp_path / run / "p" / "booking-00005"
        d.mkdir(parents=True)
        (d / "0.agent-01.opus").write_bytes(b"x")
        (d / "0.manifest.json").write_text("{}")
    removed = prune_agent_audio(tmp_path, keep_runs=2)
    assert [p.parts[-4] for p in removed] == ["20261001-000000"]
    assert not (tmp_path / "20261001-000000" / "p" / "booking-00005" / "0.agent-01.opus").exists()
    assert (tmp_path / "20261001-000000" / "p" / "booking-00005" / "0.manifest.json").exists()  # records stay
    assert (tmp_path / "20261003-000000" / "p" / "booking-00005" / "0.agent-01.opus").exists()
    assert prune_agent_audio(tmp_path / "missing") == []


class SpeakingPolicy:
    """Answers with audio only (no text): the env must hear it. The second turn cuts the caller off inside a pause."""

    id = "speaking-fake"

    def __init__(self, lines):
        self.lines, self.tts = lines, ToneTTS()

    async def run(self, env):
        onsets = [None, -0.2, None, None]
        for i, line in enumerate(self.lines):
            samples, sr = self.tts.synthesize(line, voice="am_adam")
            obs = await env.step_async(SchedulingAction(kind="say", audio_b64=base64.b64encode(wav_bytes(samples, sr)).decode(), sample_rate=sr,
                                                        onset_s=onsets[i] if i < len(onsets) else None))
            if obs.get("done"):
                return obs
            if i == 0:
                dob = env.task.patient["dob"]
                await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": dob}))
        return await env.step_async(SchedulingAction(kind="end_call"))


class TextEar:
    """Pretends to transcribe: returns the lines it was told, in order (tone audio carries no words)."""

    name = "text-ear"

    def __init__(self, lines):
        self.lines, self.i = list(lines), 0

    def transcribe_samples(self, samples, sample_rate):
        self.i += 1
        return self.lines[self.i - 1]


@local_only
async def test_env_hears_a_speaking_policy_records_opus_and_measures_turn_taking(medplum, tmp_path):
    from env.policies import run_episode

    lines = ["Could I have your date of birth, please?", "Thanks. Anything else?", "Goodbye."]
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone", ears=TextEar(lines))
    r = await run_episode(env, SpeakingPolicy(lines), seed=5, family="cancel", out=tmp_path, run_id="20261008-000000")
    assert r.error == "" and r.turns == 3
    # the transcript stood in for the text: the safety gate and the judges saw the heard lines
    agent_turns = env.last_world_extra["agent_turns"]
    assert [t["text"] for t in agent_turns] == lines
    says = [s for s in env.steps if s["action"]["kind"] == "say"]
    assert all(s["action"]["has_audio"] and s["action"]["agent_audio"]["heard"] == lines[i] for i, s in enumerate(says))
    assert says[1]["action"]["onset_s"] == -0.2 and says[1]["action"]["agent_audio"]["onset_s"] == -0.2
    assert says[0]["action"]["agent_audio"]["wall_latency_s"] is not None and "audio_b64" not in says[0]["action"]
    assert "pauses" in env.initial_observation["audio"] and all("pauses" in s["observation"]["audio"] for s in says if s["observation"].get("audio"))
    m = env.manifest()
    tt = m["turn_taking"]
    assert tt["agent_turns_with_audio"] == 3 and tt["ears"] == "text-ear" and tt["barge_ins"] == 1 and tt["waited"] == 2
    assert tt["response_latency_s"]["p50"] is not None and tt["agent_talk_s"] > 0 and tt["caller_talk_s"] > 0
    folder = tmp_path / "20261008-000000" / "speaking-fake" / f"{env.task.id}-t{env.task.tier}"  # episode folders carry the tier
    opus = sorted(p.name for p in folder.glob("0.agent-*.opus"))
    assert opus == ["0.agent-01.opus", "0.agent-02.opus", "0.agent-03.opus"]
    y, sr = decode_audio((folder / "0.agent-01.opus").read_bytes())
    assert sr == 16000 and len(y) > 0.3 * sr  # the tone backend speaks at 16 kHz, ~0.5 s for the question
    assert m["method_version"].endswith("+audio")
    # a text-only episode has no turn-taking block
    text_env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await text_env.reset(seed=5, family="cancel")
    try:
        assert text_env.manifest()["turn_taking"] is None
    finally:
        await text_env.close()


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()
