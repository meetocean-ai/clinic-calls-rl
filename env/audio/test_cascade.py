"""The cascaded voice policy (roadmap P2-04): the policy hears the caller through an ASR, the env keeps what the
caller really said. No model needed — the tone TTS and an "ear" that reads the clip's spoken text (optionally garbling
it) stand in for Kokoro + Whisper; the real pair runs through `python -m env.policies --voice`.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_cascade.py -v
"""
from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest

from env.audio.cascade import CascadedVoicePolicy, HeardEnv, Transcriber, VoicedCaller, asr_summary
from env.audio.channel import VoiceChannel
from env.audio.disfluencies import verbalize
from env.audio.tts import ToneTTS
from env.policies import OpenAICompatiblePolicy, ProductionPolicy, make_policy, run_episode
from env.s2s.stubs import named_surname, pick_from_messages
from env.voice_gap import voice_gap_rows, voice_gap_table

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


class Ear(Transcriber):
    """An ASR that hears exactly what was voiced (the clip's `spoken` text), through `garble` when given."""

    def __init__(self, garble=None):
        self.name, self.garble, self.heard = "echo-ear", garble, []

    def hear(self, clip):
        text = clip["spoken"]
        if self.garble:
            text = self.garble(text)
        self.heard.append(text)
        return text


def _shift_day(text: str) -> str:
    """Mishear the day of the month by one — the ASR error a date cannot survive ("April 11" → "April 12")."""
    return re.sub(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})\b",
                  lambda m: f"{m.group(1)} {int(m.group(2)) + 1}", text)


class _FakeSim:
    tenant_id = "tenant-1"

    def __init__(self):
        self.lines = ["Hi, I'd like to book a follow-up with Dr. Eddins.", "It's 1964-04-11.", "Yes, that works.", "Thank you, goodbye."]
        self.agent_said = []

    async def first_utterance(self):
        return self.lines[0]

    async def reply_to(self, agent_message):
        self.agent_said.append(agent_message)
        return self.lines[len(self.agent_said)]


async def test_voiced_caller_hands_the_agent_the_transcript_and_keeps_what_was_said():
    sim, ear = _FakeSim(), Ear(_shift_day)
    caller = VoicedCaller(sim, VoiceChannel.for_task(seed=5, tier=4, tts=ToneTTS()), ear)
    opening = await caller.first_utterance()
    dob = await caller.reply_to("Your date of birth, please?")
    assert caller.tenant_id == "tenant-1" and sim.agent_said == ["Your date of birth, please?"]
    assert caller.said == sim.lines[:2]  # the clean lines, ISO date intact
    assert "April 12" in dob and "1964" in dob, dob  # the agent heard the day wrong
    assert "Dr. Eddins" in opening  # tier-4 disfluencies never touch the protected tokens
    assert len(caller.rows) == 2 and caller.rows[1]["said"] == "It's 1964-04-11." and caller.rows[1]["heard"] == dob
    assert caller.rows[1]["spoken"].startswith("It's") and "April 11, 1964" in caller.rows[1]["spoken"]  # verbalized before TTS
    assert 0 < caller.rows[1]["similarity"] < 1.0
    s = asr_summary(caller.rows)
    assert s["lines"] == 2 and s["min_similarity"] <= s["mean_similarity"] <= 1.0 and asr_summary([]) is None


def test_spoken_dates_as_an_asr_writes_them_are_dates_the_env_understands():
    """The oracle (and verify_patient's anti-cheat) read dates out of text; a transcript says "April 11th, 1964"."""
    from env.env import _dates_spoken

    assert _dates_spoken(["It's April 11th, 1964."]) == {"1964-04-11"}
    assert _dates_spoken(["My date of birth is August 25, 1971"]) == {"1971-08-25"}
    assert _dates_spoken(["It's 1971-08-25.", "born 3rd of nothing"]) == {"1971-08-25"}


async def test_cascaded_policy_refuses_an_env_without_audio():
    policy = CascadedVoicePolicy(SimpleNamespace(id="inner"), transcriber=Ear())
    assert policy.id == "inner+voice"
    with pytest.raises(RuntimeError, match="audio mode"):
        await policy.run(SimpleNamespace(channel=None, world=SimpleNamespace(extra={})))


def test_make_policy_voice_wraps_any_policy_and_tags_the_id():
    p = make_policy("oracle", voice=True, transcriber=Ear())
    assert isinstance(p, CascadedVoicePolicy) and p.id == "oracle+voice" and p.inner.id == "oracle"
    q = make_policy("openai-compatible", voice=True, transcriber=Ear(), endpoint="http://localhost:11434/v1", model="qwen3:8b", think=False)
    assert q.id == "openai-compatible-qwen3-8b+voice" and q.inner.extra == {"reasoning_effort": "none"}
    assert make_policy("openai-compatible", endpoint="http://x/v1", model="m").extra == {}


def test_voice_gap_table_pairs_tasks_and_reports_the_ear():
    def runs(policy, rewards, asr=None, err=""):
        return [{"policy_id": policy, "family": "booking", "task_id": tid, "reward": rw, "trial": i, "error": err,
                 "diffs": [] if rw else ["booking_matches: none"], "asr": asr}
                for tid, rws in rewards.items() for i, rw in enumerate(rws)]

    text = runs("m", {"booking-5": [1, 1, 1], "booking-10": [1, 0, 1], "booking-15": [1, 1, 1]})
    voice = runs("m+voice", {"booking-5": [1, 0, 0], "booking-10": [1, 1, 1], "booking-20": [0, 0, 0]},
                 asr={"lines": 4, "mean_similarity": 0.9, "min_similarity": 0.7, "below_floor": 1})
    rows = voice_gap_rows(text, voice)
    b, a = rows["booking"], rows["all"]
    assert b["tasks"] == 2 and b["k"] == 3 and sorted(b["unpaired"]) == ["booking-15:text-only", "booking-20:voice-only"]
    assert b["text_pass"] == round(5 / 6, 3) and b["voice_pass"] == round(4 / 6, 3) and b["gap"] == round(1 / 6, 3)
    assert b["lost"] == 1 and b["gained"] == 1 and b["asr_lines"] == 24 and b["asr_below_floor"] == 6 and b["asr_mean_similarity"] == 0.9
    assert a["gap"] == b["gap"] and a["text_ci"] and a["voice_ci"][0] < a["voice_pass"] < a["voice_ci"][1]
    table = voice_gap_table(rows, text_id="m", voice_id="m+voice")
    assert "| m pass@1 |" in table and "| **all** | 2 | 3 |" in table and "booking-20:voice-only" in table
    assert "+0.167" in table and "6/24" in table


class _FakeChat:
    """Scripted model turns; an item may be a callable over the live message list (to read what the caller said)."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if callable(item):
            item = item(kwargs["messages"])
        if isinstance(item, str):
            msg = SimpleNamespace(content=item, tool_calls=None)
        else:
            msg = SimpleNamespace(content="", tool_calls=[
                SimpleNamespace(id=f"c{i}", function=SimpleNamespace(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(item)])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


_MONTHS = {m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august", "september",
                                       "october", "november", "december"), 1)}


def _verify_with_heard_dob(messages):
    """What a rule-following model does: read the DOB the caller said (as heard) and verify with it in ISO form."""
    last = [m for m in messages if m["role"] == "user"][-1]["content"]
    m = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})", last)
    assert m, last
    return [("verify_patient", {"dob": f"{m.group(3)}-{_MONTHS[m.group(1).lower()]:02d}-{int(m.group(2)):02d}"})]


def _confirm_named_providers_slot(messages):
    """What a rule-following model does: book the slot of the doctor the caller asked for (named in the opening line)."""
    opening = next(m["content"] for m in messages if m["role"] == "user")
    return [("confirm_appointment", {"slot_index": pick_from_messages(messages, named_surname(opening))})]


def _model_script():
    return ["Hi! Could I have your date of birth, please?", _verify_with_heard_dob, [("look_up_availability", {})],
            "I have an opening tomorrow morning. Does that work?", _confirm_named_providers_slot,
            "You're all set. Anything else?", [("end_call", {})]]


def _scripted_voice_policy(ear):
    inner = OpenAICompatiblePolicy("http://fake/v1", "fake-model")
    inner.client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeChat(_model_script())))
    return CascadedVoicePolicy(inner, transcriber=ear)


@local_only
async def test_cascaded_openai_compatible_policy_books_from_the_transcript(medplum):
    from env.env import EhrSchedulingEnv

    ear = Ear()
    policy = _scripted_voice_policy(ear)
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    r = await run_episode(env, policy, seed=5, family="booking")  # tier 1: the plain booking the fixed script fits
    assert r.reward == 1, (r.diffs, r.error)
    assert r.policy_id == "openai-compatible-fake-model+voice" and r.method_version.endswith("+audio")
    dob, spoken_dob = env.task.patient["dob"], verbalize(env.task.patient["dob"])  # "1971-08-25" / "August 25, 1971"
    # the model only ever saw transcripts: the DOB reached it spoken, never in ISO form (the caller record aside)
    users = [m["content"] for m in policy.inner.client.chat.completions.calls[-1]["messages"] if m["role"] == "user"][1:]
    assert any(spoken_dob in u for u in users) and not any(dob in u for u in users), users
    assert len(ear.heard) >= 3
    # the env kept the clean lines (ISO date) and logged every heard line next to them
    assert any(dob in line for line in env.caller_lines)
    m = env.manifest()
    assert m["asr"]["model"] == "echo-ear" and m["asr"]["summary"]["lines"] == len(ear.heard) and m["asr"]["lines"][1]["said"] == f"It's {dob}."
    assert m["asr"]["lines"][1]["heard"] == f"It's {spoken_dob}."
    # the env's trajectory keeps the clean line as patient_text and what the ear made of it as patient_text_heard
    heard_steps = [s["observation"] for s in env.steps if s["action"]["kind"] == "say" and s["observation"]["patient_text"]]
    assert heard_steps and all(o["patient_text_heard"] == verbalize(o["patient_text"]) for o in heard_steps)
    assert env.initial_observation["patient_text_heard"] == ear.heard[0]


@local_only
async def test_cascaded_policy_loses_the_booking_when_the_ear_shifts_the_dob(medplum):
    """The voice gap, mechanically: a day misheard → verify_patient refuses (the caller never said that date) → no
    booking → reward 0, and the manifest shows which line the ear got wrong."""
    from env.env import EhrSchedulingEnv

    policy = _scripted_voice_policy(Ear(_shift_day))
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    r = await run_episode(env, policy, seed=5, family="booking")
    assert r.reward == 0 and any(d.startswith("booking_matches") for d in r.diffs), (r.diffs, r.error)
    tool_msgs = [m for m in policy.inner.client.chat.completions.calls[-1]["messages"] if m["role"] == "tool"]
    assert "has not given that date of birth" in tool_msgs[0]["content"]
    rows = env.manifest()["asr"]["lines"]
    dob_row = next(x for x in rows if env.task.patient["dob"] in x["said"])
    assert _shift_day(verbalize(env.task.patient["dob"])) in dob_row["heard"] and dob_row["similarity"] < 1.0


@local_only
async def test_production_policy_voice_hook_feeds_the_agents_the_transcript(medplum, monkeypatch):
    """The production path without the production LLMs: a fake drive_inbound runs the agents' side of a call, the
    cascade wraps the caller it is handed, and the env's record keeps said / heard apart."""
    import runtime.inbound as inbound
    from env.env import EhrSchedulingEnv

    seen = {}

    async def fake_resolve(tenant, phone):
        return None

    async def fake_drive_inbound(*, tenant, patient, profile, unknown_caller, collected, max_turns, simulator_wrap=None):
        sim = simulator_wrap(_FakeSim()) if simulator_wrap else _FakeSim()
        seen["sim"] = sim
        turns, user = [], await sim.first_utterance()
        for agent_line in ("Your date of birth, please?", "Thanks, I can book that. OK?", "Done. Goodbye."):
            turns.append(SimpleNamespace(agent_message=agent_line, patient_message=user, tool_calls=[], turn_ms=10.0, ttft_ms=None))
            user = await sim.reply_to(agent_line)
        return SimpleNamespace(turns=turns, stopped_reason="user_ends")

    monkeypatch.setattr(inbound, "drive_inbound", fake_drive_inbound)
    monkeypatch.setattr(inbound, "resolve_caller_by_phone", fake_resolve)
    policy = CascadedVoicePolicy(ProductionPolicy(), transcriber=Ear(_shift_day))
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    r = await run_episode(env, policy, seed=5, family="booking", tier=3)
    assert r.policy_id == "production+voice" and r.error == "" and r.turns == 3 and r.stopped_reason == "end_call"
    sim = seen["sim"]
    assert isinstance(sim, VoicedCaller) and sim.said[1] == "It's 1964-04-11." and "April 12" in sim.heard[1]
    assert env.caller_lines == sim.said  # verify_patient / the safety gate read what was said, not what was heard
    says = [s["observation"] for s in env.steps if s["action"]["kind"] == "say"]
    assert says[1]["patient_text_clean"] == "It's 1964-04-11." and "April 12" in says[1]["patient_text"]
    agent_turns = env.last_world_extra["agent_turns"]  # the tenant is gone after the episode; the env keeps its extras
    assert agent_turns[2]["caller_so_far"] == sim.said[:3]  # clean lines, up to and including the one that turn answers
    assert env.manifest()["asr"]["summary"]["lines"] == 4 and env.manifest()["asr"]["model"] == "echo-ear"


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()
