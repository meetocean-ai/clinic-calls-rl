"""Closed reference adapters offline (roadmap P2-07): scripted OpenAI Realtime and Gemini Live sessions drive the live
env with the real event / message shapes; no key, no minutes billed. The real sessions are `connect_realtime` /
`connect_live`, used only under a ledger line.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/s2s/test_closed_adapters.py -v
"""
from __future__ import annotations

import base64
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from env.audio.channel import wav_to_samples
from env.env import EhrSchedulingEnv
from env.s2s.gemini_live import GeminiLivePolicy, clip_from_pcm16, gemini_tools, pcm16
from env.s2s.openai_realtime import PCM_RATE, OpenAIRealtimePolicy, pcm16_b64, wav_b64_from_pcm16
from env.s2s.stubs import PICK, pick_slot, slots_in, substitute

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def _tone_pcm(seconds=0.5, rate=24000):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2").tobytes()


def test_pcm_helpers_round_trip():
    x = 0.3 * np.sin(np.linspace(0, 100, 8000)).astype(np.float32)
    b64 = pcm16_b64(x, 8000)
    clip = wav_b64_from_pcm16([b64])
    y, sr = wav_to_samples(base64.b64decode(clip["wav_b64"]))
    assert sr == PCM_RATE and abs(len(y) / sr - 1.0) < 0.01 and wav_b64_from_pcm16([]) is None
    raw = pcm16(x, 8000)
    assert len(raw) == 2 * 16000  # 1 s at 16 kHz, 16-bit
    assert clip_from_pcm16([_tone_pcm(0.5)])["seconds"] == 0.5 and clip_from_pcm16([]) is None
    decls = gemini_tools()[0]["function_declarations"]
    assert {d["name"] for d in decls} >= {"verify_patient", "look_up_availability", "confirm_appointment", "end_call"}


class ScriptedRealtime:
    """Server events scripted per `response.create`; function-call items are answered by the policy and trigger the next script item."""

    def __init__(self, responses, surname=None):
        self.responses, self.sent, self.closed, self.surname, self.slot = list(responses), [], False, surname, 0
        self._queue = []

    async def send(self, event):
        self.sent.append(event)
        if event["type"] == "conversation.item.create" and event["item"].get("type") == "function_call_output" and slots_in(event["item"]["output"]):
            self.slot = pick_slot(json.loads(event["item"]["output"]), self.surname)
        if event["type"] == "response.create":
            self._queue.extend(substitute(self.responses.pop(0), self.slot))

    async def recv(self):
        return self._queue.pop(0)

    async def close(self):
        self.closed = True


def _speech(text):
    return [{"type": "response.output_audio.delta", "delta": base64.b64encode(_tone_pcm(0.4)).decode()},
            {"type": "response.output_audio_transcript.delta", "delta": text}, {"type": "response.done"}]


def _call(name, args, call_id):
    return [{"type": "response.function_call_arguments.done", "name": name, "arguments": json.dumps(args), "call_id": call_id}, {"type": "response.done"}]


@local_only
async def test_openai_realtime_policy_books_over_the_event_protocol(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    await env.reset(seed=5, family="booking")
    dob = env.task.patient["dob"]
    session = ScriptedRealtime(surname=env.task.provider["family"], responses=[
        _speech("Hi! Could I have your date of birth, please?"),
        _call("verify_patient", {"dob": dob}, "c1"), _call("look_up_availability", {}, "c2"), _speech("I have an opening tomorrow morning. Does that work?"),
        _call("confirm_appointment", {"slot_index": PICK}, "c3"), _speech("You're all set. Anything else?"),
        _call("end_call", {}, "c4"),
    ])

    async def factory():
        return session

    policy = OpenAIRealtimePolicy(factory, model="gpt-realtime")
    assert policy.id == "closed-openai-gpt-realtime"
    try:
        await policy.run(env)
        if not env.done:
            await env.finish("policy_returned")
        assert env.result.reward == 1, env.result.diffs
        kinds = [e["type"] for e in session.sent]
        assert kinds[0] == "session.update" and session.sent[0]["session"]["tools"][0]["name"] == "verify_patient"
        assert kinds.count("input_audio_buffer.append") == 4 and kinds.count("input_audio_buffer.commit") == 4
        outputs = [e for e in session.sent if e["type"] == "conversation.item.create" and e["item"]["type"] == "function_call_output"]
        assert [o["item"]["call_id"] for o in outputs] == ["c1", "c2", "c3"] and '"verified": true' in outputs[0]["item"]["output"].lower()
        says = [s for s in env.steps if s["action"]["kind"] == "say"]
        assert len(says) == 3 and all(s["action"]["has_audio"] and s["action"]["agent_audio"]["sample_rate"] == PCM_RATE for s in says)
        assert session.closed and env.manifest()["turn_taking"]["agent_turns_with_audio"] == 3
    finally:
        await env.close()


class ScriptedGemini:
    """`receive()` yields the scripted messages for the current caller turn; tool responses trigger the next item."""

    def __init__(self, turns, surname=None):
        self.turns, self.realtime, self.tool_responses, self.client_content, self.closed = list(turns), [], [], [], False
        self.surname, self.slot, self._pending = surname, 0, []

    async def send_client_content(self, **kw):
        self.client_content.append(kw)

    async def send_realtime_input(self, **kw):
        self.realtime.append(kw)
        if "audio" in kw:
            self._pending = list(self.turns.pop(0))

    async def send_tool_response(self, function_responses):
        self.tool_responses.append(function_responses)
        for fr in function_responses:
            if slots_in(fr["response"].get("result")):
                self.slot = pick_slot(fr["response"]["result"], self.surname)
        self._pending.extend(self.turns.pop(0))

    async def receive(self):
        while self._pending:
            msg = self._pending.pop(0)
            tc = getattr(msg, "tool_call", None)
            if tc is not None:
                for fc in tc.function_calls:
                    fc.args = substitute(fc.args, self.slot)
            yield msg

    async def close(self):
        self.closed = True


def _g_speech(text, complete=True):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=_tone_pcm(0.4)), text=text)
    return [SimpleNamespace(server_content=SimpleNamespace(model_turn=SimpleNamespace(parts=[part]), turn_complete=False), tool_call=None),
            SimpleNamespace(server_content=SimpleNamespace(model_turn=None, turn_complete=complete), tool_call=None)]


def _g_call(name, args, fid):
    return [SimpleNamespace(server_content=None, tool_call=SimpleNamespace(function_calls=[SimpleNamespace(id=fid, name=name, args=args)]))]


@local_only
async def test_gemini_live_policy_books_over_the_session_api(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    await env.reset(seed=5, family="booking")
    dob = env.task.patient["dob"]
    session = ScriptedGemini(surname=env.task.provider["family"], turns=[
        _g_speech("Hi! Could I have your date of birth, please?"),
        _g_call("verify_patient", {"dob": dob}, "f1"), _g_call("look_up_availability", {}, "f2"), _g_speech("I have an opening tomorrow morning. Does that work?"),
        _g_call("confirm_appointment", {"slot_index": PICK}, "f3"), _g_speech("You're all set. Anything else?"),
        _g_call("end_call", {}, "f4"),
    ])

    async def factory():
        return session

    policy = GeminiLivePolicy(factory)
    assert policy.id.startswith("closed-gemini-")
    try:
        await policy.run(env)
        if not env.done:
            await env.finish("policy_returned")
        assert env.result.reward == 1, env.result.diffs
        audio_sends = [r for r in session.realtime if "audio" in r]
        assert len(audio_sends) == 4 and audio_sends[0]["audio"]["mime_type"] == "audio/pcm;rate=16000" and len(session.client_content) == 1
        assert [r[0]["id"] for r in session.tool_responses] == ["f1", "f2", "f3"] and session.tool_responses[0][0]["response"]["result"]["verified"] is True
        says = [s for s in env.steps if s["action"]["kind"] == "say"]
        assert len(says) == 3 and all(s["action"]["has_audio"] for s in says) and session.closed
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
