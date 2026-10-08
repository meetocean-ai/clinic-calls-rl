"""S2S policy adapters offline (roadmap P2-06) and the audio judges (P2-05b): scripted servers stand in for vLLM-Omni,
the VoiceChat NIM and a Moshi server; the env is real (local Medplum, tone caller audio). Each adapter must book the
tier-1 held-out booking task through the env's tools and leave a record with its own audio.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/s2s -v
"""
from __future__ import annotations

import base64
import json
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

from env.audio.audio_judges import LATENCY_BUDGET_S, deterministic_scores, parse_rubric
from env.audio.channel import wav_bytes, wav_to_samples
from env.audio.tts import ToneTTS
from env.env import EhrSchedulingEnv
from env.s2s.omni import OmniPolicy
from env.s2s.personaplex import DELEGATE_TOKEN, PersonaPlexPolicy
from env.s2s.stubs import PICK, pick_from_messages, pick_slot, slots_in, substitute
from env.s2s.transport import concat_audio, onset_s, text_of
from env.s2s.voicechat import VoiceChatPolicy

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")

TTS = ToneTTS()
ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def _clip_b64(text: str, sr_out=None) -> str:
    samples, sr = TTS.synthesize(text, voice="am_adam")
    if sr_out and sr_out != sr:
        from env.audio.degrade import resample

        samples, sr = resample(samples, sr, sr_out), sr_out
    return base64.b64encode(wav_bytes(samples, sr)).decode()


def _audio_event(t: float, text: str, sr=None):
    return {"type": "audio", "t": t, "wav_b64": _clip_b64(text, sr), "sample_rate": sr or 16000}


# ── protocol helpers and judges, no env ────────────────────────────────────────────────────────────────────────


def test_concat_audio_text_and_onset():
    events = [{"type": "text", "t": 1.0, "text": "Sure — "}, _audio_event(1.0, "Sure, one moment."), _audio_event(1.6, "Your date of birth?", sr=8000),
              {"type": "text", "t": 1.6, "text": "your date of birth?"}, {"type": "end_turn", "t": 3.0}]
    clip = concat_audio(events)
    y, sr = wav_to_samples(base64.b64decode(clip["wav_b64"]))
    assert sr == 16000 and clip["sample_rate"] == 16000 and clip["onset_t"] == 1.0 and abs(len(y) / sr - clip["seconds"]) < 0.01
    assert text_of(events) == "Sure — your date of birth?" and onset_s(events, 2.5) == -1.5 and onset_s([{"type": "end_turn", "t": 1}], 2.5) is None
    assert concat_audio([{"type": "text", "t": 0, "text": "x"}]) is None


def test_parse_rubric_and_deterministic_scores():
    r = parse_rubric('Here you go:\n```json\n{"tone": 4, "empathy": 5.4, "mispronunciation": 0, "artifacts": 3, "reasoning": "warm but clipped"}\n```')
    assert r == {"tone": 4, "empathy": 5, "mispronunciation": 1, "artifacts": 3, "reasoning": "warm but clipped"}
    assert "error" in parse_rubric("no json here") and "error" in parse_rubric('{"tone": 3}')
    tt = {"turns": [{"response_latency_s": 0.4}, {"response_latency_s": 2.0}, {"response_latency_s": None}], "interruptions": 1,
          "takeovers_in_pause": 0, "overlap_s": 0.7, "backchannels": 1, "response_latency_s": {"p50": 0.4, "p95": 2.0}}
    from env.audio.degrade import DegradationSpec

    spec = DegradationSpec.from_seed(5, 4).to_dict()  # the real key names, not a hand-typed shape
    d = deterministic_scores(tt, {"degradation": spec})
    assert d["latency_within_budget"] == {"budget_s": LATENCY_BUDGET_S, "rate": 0.5, "p50_s": 0.4, "p95_s": 2.0}
    assert d["no_interruptions"]["rate"] == round(1 - 1 / 3, 3) and d["line"]["frame_drop_rate"] == spec["frame_drop_p"] and d["line"]["snr_db"] == spec["snr_db"] and d["talk_over"]["overlap_s"] == 0.7
    assert deterministic_scores(None, None) is None


# ── scripted servers ────────────────────────────────────────────────────────────────────────────────────────────


class FakeOmniClient:
    """vLLM-Omni stand-in: chat.completions scripted per step (callables see the live message list), audio.speech = tone TTS."""

    def __init__(self, script, surname=None):
        self.script, self.calls, self.speech_calls, self.surname = list(script), [], [], surname
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.audio = SimpleNamespace(speech=SimpleNamespace(create=self._speech))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if callable(item):
            item = item(kwargs["messages"])
        item = substitute(item, pick_from_messages(kwargs["messages"], self.surname))  # PICK → the named provider's slot
        if isinstance(item, str):
            msg = SimpleNamespace(content=item, tool_calls=None)
        else:
            msg = SimpleNamespace(content="", tool_calls=[SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a)))
                                                          for i, (n, a) in enumerate(item)])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    async def _speech(self, **kwargs):
        self.speech_calls.append(kwargs)
        samples, sr = TTS.synthesize(kwargs["input"], voice="am_adam")
        return SimpleNamespace(content=wav_bytes(samples, sr))


def _dob_from_messages(messages):
    """The adapter hands the caller's audio to the model; the fake 'hears' the DOB from the env's text, like a real ear would."""
    raise NotImplementedError  # replaced per test with a closure over the env


class ScriptedDuplex:
    """A duplex server scripted per caller turn. Each turn item is a list of events; a `("tool", name, args)` tuple inside
    becomes a toolcall event whose result triggers the next item in `after_tool`. Tracks what it was sent."""

    name = "scripted-duplex"

    def __init__(self, turns, after_tool=None, prefills=None, surname=None):
        self.turns, self.after_tool, self.prefills, self.surname = list(turns), list(after_tool or []), list(prefills or []), surname
        self.sent, self.results, self.prefilled, self.opened, self.closed = [], [], [], None, False

    async def open(self, system_prompt):
        self.opened = system_prompt

    async def turn(self, wav_b64, sample_rate):
        self.sent.append((len(wav_b64), sample_rate))
        return substitute(self.turns.pop(0) if self.turns else [{"type": "hangup", "t": 0.0}], getattr(self, "slot", 0))

    async def tool_result(self, call_id, result):
        self.results.append((call_id, result))
        if slots_in(result):
            self.slot = pick_slot(result, self.surname)
        nxt = self.after_tool.pop(0) if self.after_tool else [{"type": "end_turn", "t": 1.0}]
        return substitute(nxt, getattr(self, "slot", 0))

    async def prefill(self, text):
        self.prefilled.append(text)
        return self.prefills.pop(0) if self.prefills else [{"type": "text", "t": 2.0, "text": text}, _audio_event(2.0, text), {"type": "end_turn", "t": 3.0}]

    async def close(self):
        self.closed = True


class FakeAuditor:
    name = "fake-auditor"

    def __init__(self):
        self.seen = []

    def audit(self, wav_b64, sample_rate, transcript):
        self.seen.append((sample_rate, transcript))
        return {"tone": 4, "empathy": 4, "mispronunciation": 5, "artifacts": 3, "reasoning": "fine"}


async def _booked(env) -> bool:
    if not env.done:
        await env.finish("policy_returned")
    return env.result.reward == 1


@local_only
async def test_omni_policy_books_with_audio_in_and_speech_out(medplum):
    auditor = FakeAuditor()
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone", auditor=auditor)
    await env.reset(seed=5, family="booking")
    dob = env.task.patient["dob"]
    client = FakeOmniClient(["Hi! Could I have your date of birth, please?", [("verify_patient", {"dob": dob})], [("look_up_availability", {})],
                            "I have an opening tomorrow morning. Does that work?", [("confirm_appointment", {"slot_index": PICK})],
                            "You're all set. Anything else?", [("end_call", {})]])
    client.surname = env.task.provider["family"]
    policy = OmniPolicy("http://fake/v1", "Qwen/Qwen3-Omni-30B-A3B-Instruct", speak=True, client=client)
    assert policy.id == "s2s-omni-qwen3-omni-30b-a3b-instruct"
    try:
        await policy.run(env)
        assert await _booked(env), env.result.diffs
        # the caller reached the model as audio parts, never as text
        first_user = client.calls[0]["messages"][1]["content"]
        assert isinstance(first_user, list) and first_user[-1]["type"] == "audio_url" and first_user[-1]["audio_url"]["url"].startswith("data:audio/wav;base64,")
        assert all(isinstance(m["content"], list) for c in client.calls for m in c["messages"] if m["role"] == "user")
        assert len(client.speech_calls) == 3 and client.speech_calls[0]["response_format"] == "wav"
        says = [s for s in env.steps if s["action"]["kind"] == "say"]
        assert len(says) == 3 and all(s["action"]["has_audio"] and s["action"]["agent_audio"]["heard"] == s["action"]["text"] for s in says)
        m = env.manifest()
        assert m["turn_taking"]["agent_turns_with_audio"] == 3 and m["turn_taking"]["waited"] == 3
        assert m["audio_scores"]["rewarded"] is False and m["audio_scores"]["deterministic"]["turns"] == 3
        assert m["audio_scores"]["auditor"]["auditor"] == "fake-auditor" and m["audio_scores"]["auditor"]["means"] == {"tone": 4.0, "empathy": 4.0, "mispronunciation": 5.0, "artifacts": 3.0}
        assert [t for _, t in auditor.seen] == [s["action"]["text"] for s in says]
        assert m["method_version"].endswith("+audio")
    finally:
        await env.close()


@local_only
async def test_voicechat_policy_routes_the_toolcall_channel_and_speaks_on_hold(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    await env.reset(seed=5, family="booking")
    dob = env.task.patient["dob"]
    turns = [
        [{"type": "text", "t": 1.2, "text": "Hi! Your date of birth, please?"}, _audio_event(1.2, "Hi! Your date of birth, please?"), {"type": "end_turn", "t": 3.0}],
        [{"type": "onhold", "t": 0.9, "text": "One moment while I check that."}, _audio_event(0.9, "One moment."),
         {"type": "toolcall", "t": 0.9, "id": "t1", "name": "verify_patient", "arguments": {"dob": dob}}],
        [{"type": "toolcall", "t": 0.5, "id": "t3", "name": "confirm_appointment", "arguments": json.dumps({"slot_index": PICK})}],
        [{"type": "toolcall", "t": 0.3, "id": "t4", "name": "end_call", "arguments": {}}],
    ]
    after_tool = [
        [{"type": "toolcall", "t": 1.4, "id": "t2", "name": "look_up_availability", "arguments": {}}],  # after verify
        [{"type": "text", "t": 2.0, "text": "I have an opening tomorrow morning. Does that work?"}, _audio_event(2.0, "Does that work?"), {"type": "end_turn", "t": 4.0}],
        [{"type": "text", "t": 1.0, "text": "You're all set. Anything else?"}, _audio_event(1.0, "All set."), {"type": "end_turn", "t": 2.0}],
    ]
    transport = ScriptedDuplex(turns, after_tool, surname=env.task.provider["family"])
    policy = VoiceChatPolicy(transport)
    try:
        await policy.run(env)
        assert await _booked(env), env.result.diffs
        assert transport.opened and transport.closed and len(transport.sent) == 4 and all(sr == 8000 for _, sr in transport.sent)
        assert [cid for cid, _ in transport.results] == ["t1", "t2", "t3"] and transport.results[0][1]["verified"] is True
        says = [s for s in env.steps if s["action"]["kind"] == "say"]
        assert len(says) == 3 and says[1]["action"]["text"].startswith("One moment while I check that. I have an opening")
        assert all(s["action"]["has_audio"] for s in says)
        tools = [s["action"]["tool_name"] for s in env.steps if s["action"]["kind"] == "tool"]
        assert tools == ["verify_patient", "look_up_availability", "confirm_appointment", "end_call"]
        tt = env.manifest()["turn_taking"]
        assert tt["agent_turns_with_audio"] == 3 and all(r["onset_s"] is not None for r in tt["turns"])
    finally:
        await env.close()


@local_only
async def test_personaplex_policy_delegates_tool_work_to_the_text_backend_and_prefills_the_answer(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    await env.reset(seed=5, family="booking")

    def verify_from_transcript(messages):
        m = ISO.findall(messages[-1]["content"] if messages[-1]["role"] == "user" else "")
        dobs = [d for d in m if d != env.task.today]
        return [("verify_patient", {"dob": dobs[-1]})]

    backend = FakeOmniClient([
        verify_from_transcript, [("look_up_availability", {})], "I have an opening tomorrow morning — does that work?",  # delegation 1
        [("confirm_appointment", {"slot_index": PICK})], "You're all set.",  # delegation 2
        [("end_call", {})],  # delegation 3
    ])
    turns = [
        [{"type": "text", "t": 1.0, "text": "Hi there! Could I get your date of birth?"}, _audio_event(1.0, "Could I get your date of birth?"), {"type": "end_turn", "t": 3.0}],
        [{"type": "text", "t": 0.8, "text": f"Thanks. {DELEGATE_TOKEN}"}, {"type": "delegate", "t": 0.8}, {"type": "end_turn", "t": 1.0}],
        [{"type": "delegate", "t": 0.5}, {"type": "end_turn", "t": 0.6}],
        [{"type": "delegate", "t": 0.5}, {"type": "end_turn", "t": 0.6}],
    ]
    backend.surname = env.task.provider["family"]
    transport = ScriptedDuplex(turns)
    policy = PersonaPlexPolicy(transport, backend, "qwen3:8b")
    try:
        await policy.run(env)
        assert await _booked(env), env.result.diffs
        assert transport.prefilled == ["I have an opening tomorrow morning — does that work?", "You're all set."]
        says = [s["action"]["text"] for s in env.steps if s["action"]["kind"] == "say"]
        assert says == ["Hi there! Could I get your date of birth?", "Thanks. I have an opening tomorrow morning — does that work?", "You're all set."]
        tools = [s["action"]["tool_name"] for s in env.steps if s["action"]["kind"] == "tool"]
        assert tools == ["verify_patient", "look_up_availability", "confirm_appointment", "end_call"]
        # the backend saw the transcript, including what the caller said (so it could verify the spoken DOB)
        assert "Transcript so far" in backend.calls[0]["messages"][1]["content"] and env.task.patient["dob"] in backend.calls[0]["messages"][1]["content"]
    finally:
        await env.close()


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()
