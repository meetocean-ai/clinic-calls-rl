"""Caller behaviour that only exists in speech (roadmap P2-02b): "hello?" after the agent's silence (and patience
running out), a date of birth voiced in pieces with pauses inside it, talking over the agent once — scripted, with
timestamps in the record, so turn-taking is measured, never judged. Tone audio; no model.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/audio/test_caller_speech.py -v
"""
from __future__ import annotations

import asyncio
import base64
import os

import pytest

from env.audio.channel import HESITATION_PAUSE_S, VoiceChannel, _date_pieces, wav_to_samples
from env.audio.tts import ToneTTS
from env.audio.turntaking import pauses
from env.env import INTERRUPT_ON_TURN, LATENCY_HELLO_S, TALK_OVER_ONSET_S, EhrSchedulingEnv, SchedulingAction

pytestmark = [pytest.mark.covers("rl-env-voice")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def test_date_pieces_split_month_day_year():
    assert _date_pieces("It's April 11, 1964.") == ["It's April", "11", "1964."]
    assert _date_pieces("My name is Pat Lee, and my date of birth is August 25, 1971.") == ["My name is Pat Lee, and my date of birth is August", "25", "1971."]
    assert _date_pieces("Yes, that works.") is None


def test_hesitant_dob_clip_has_pauses_inside_the_date_and_timestamped_events():
    ch = VoiceChannel.for_task(seed=5, tier=1, tts=ToneTTS())
    plain = ch.render("It's 1964-04-11.")
    hesitant = ch.render("It's 1964-04-11.", hesitate_dob=True)
    assert plain["events"] == [{"t": 0.0, "kind": "speech", "text": "It's April 11, 1964."}] and "caller_onset_s" not in plain
    kinds = [e["kind"] for e in hesitant["events"]]
    assert kinds == ["speech", "pause", "speech", "pause", "speech"] and hesitant["spoken"] == "It's April … 11 … 1964."
    pause_events = [e for e in hesitant["events"] if e["kind"] == "pause"]
    assert all(HESITATION_PAUSE_S[0] <= e["seconds"] <= HESITATION_PAUSE_S[1] for e in pause_events)
    assert hesitant["seconds"] > plain["seconds"] + sum(e["seconds"] for e in pause_events) - 0.05
    # the pause map the turn-taking metrics read finds the same two gaps
    samples, sr = wav_to_samples(base64.b64decode(hesitant["wav_b64"]))
    found = pauses(samples, sr)
    assert len(found) == 2 and all(abs((e - s) - p["seconds"]) < 0.1 for (s, e), p in zip(found, pause_events))
    assert ch.render("It's 1964-04-11.", hesitate_dob=True)["wav_b64"] == hesitant["wav_b64"]  # deterministic per seed
    talk_over = ch.render("Actually, wait — yes.", onset_s=-0.6)
    assert talk_over["caller_onset_s"] == -0.6


class SlowThenFast:
    """Agent that is silent for longer than the caller tolerates on its first two turns, then answers quickly."""

    id = "slow-then-fast"

    async def run(self, env):
        await asyncio.sleep(LATENCY_HELLO_S + 0.2)
        await env.step_async(SchedulingAction(kind="say", text="Hello, clinic front desk."))
        await asyncio.sleep(LATENCY_HELLO_S + 0.2)
        await env.step_async(SchedulingAction(kind="say", text="Could I have your date of birth, please?"))
        await env.step_async(SchedulingAction(kind="say", text="Thank you. One moment."))
        return await env.step_async(SchedulingAction(kind="end_call"))


@local_only
async def test_caller_reacts_to_silence_hesitates_on_the_dob_and_talks_over_once(medplum):
    from env.tasks import generate

    task = generate("booking", 5, tier=3)
    task.realism = sorted(set(task.realism) | {"hesitant_dob", "interruptions"})
    env = EhrSchedulingEnv(medplum, scripted_caller=True, audio=True, tts_prefer="tone")
    try:
        await env.reset(seed=task.seed, family=task.family, tier=task.tier, task=task)
        start_coop = task.cooperation
        await SlowThenFast().run(env)
        says = [s for s in env.steps if s["action"]["kind"] == "say"]
        # turn 1: the agent took > 1.5 s → the caller says hello first, cooperation drops a notch
        assert says[0]["observation"]["patient_text"].startswith("Hello? Are you still there?")
        ev1 = says[0]["observation"]["audio"]["caller_events"]
        assert ev1[0]["kind"] == "latency_hello" and ev1[0]["agent_latency_s"] > LATENCY_HELLO_S and ev1[0]["cooperation"] == max(1, start_coop - 1)
        # turn 2: silent again → the second line, patience lower; it is also the talk-over turn and carries the DOB → hesitant
        t2 = says[INTERRUPT_ON_TURN - 1]["observation"]
        dobs = {task.claimed_dob, task.misstated_dob or task.claimed_dob}  # dob_corrected: the first answer is the misstated one
        assert t2["patient_text"].startswith("Actually, wait — Hello?? I said —") and any(d in t2["patient_text"] for d in dobs)
        kinds = sorted(e["kind"] for e in t2["audio"]["caller_events"])
        assert kinds == ["hesitant_dob", "latency_hello", "talk_over"] and t2["audio"]["caller_onset_s"] == TALK_OVER_ONSET_S
        assert [e["kind"] for e in t2["audio"]["events"]].count("pause") == 2 and "…" in t2["audio"]["spoken"]
        # turn 3: fast → no reaction, no second talk-over, no second hesitation
        t3 = says[2]["observation"]
        assert not t3["patient_text"].startswith(("Hello", "Actually")) and "caller_events" not in t3["audio"] and "caller_onset_s" not in t3["audio"]
        m = env.manifest()
        assert m["caller_events"]["latency_hellos"] == 2 and m["caller_events"]["hesitant_dob"] is True and m["caller_events"]["talk_overs"] == 1
        assert m["caller_events"]["cooperation_end"] == max(1, start_coop - 2)
        # the clean DOB is still what verify_patient would accept (the ISO form is in the text)
        assert any(d in line for line in env.caller_lines for d in dobs)
    finally:
        await env.close()


@local_only
async def test_text_mode_caller_has_no_speech_behaviour(medplum):
    from env.tasks import generate

    task = generate("booking", 5, tier=3)
    task.realism = sorted(set(task.realism) | {"hesitant_dob", "interruptions"})
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    try:
        await env.reset(seed=task.seed, family=task.family, tier=task.tier, task=task)
        await asyncio.sleep(LATENCY_HELLO_S + 0.2)
        obs = await env.step_async(SchedulingAction(kind="say", text="Hello, clinic."))
        assert not obs["patient_text"].startswith("Hello?") and env.manifest()["caller_events"] is None
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
