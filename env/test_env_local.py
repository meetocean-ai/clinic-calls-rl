"""The EHR environment against the LOCAL Medplum, no LLM (E3–E5 gates).

Needs docker-compose-medplum-local.yml up and .medplum-local.env sourced (MEDPLUM_BASE_URL on
localhost). Skips otherwise. Run:
    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/test_env_local.py -v
"""
from __future__ import annotations

import os
from datetime import date

import pytest

from env.env import EhrSchedulingEnv, SchedulingAction
from env.policies import OraclePolicy, RandomPolicy, run_episode
from env.tasks import CHAOS_BY_FAMILY, FAMILIES, TIERS, generate, validate

pytestmark = [
    pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)"),
    pytest.mark.covers("rl-scheduling-env"),
]

TODAY = date(2030, 1, 7)


# ── Generator (no server) ────────────────────────────────────────────


def test_tasks_are_deterministic_and_valid():
    for family in FAMILIES:
        for seed in range(1, 21):
            for tier in TIERS:
                a = generate(family, seed, tier=tier, today=TODAY)
                b = generate(family, seed, tier=tier, today=TODAY)
                assert a.to_json() == b.to_json()
                assert validate(a) == [], (a.id, validate(a))
                assert a.patient["phone"].startswith("+1500555")


def test_split_and_knob_menu():
    assert generate("booking", 5, today=TODAY).split == "heldout" and generate("booking", 6, today=TODAY).split == "train"
    for fam, knobs in CHAOS_BY_FAMILY.items():
        for k in knobs:
            t = generate(fam, 3, tier=3, today=TODAY, chaos=[k])
            assert t.chaos == [k]
    assert generate("booking", 3, tier=3, today=TODAY, chaos=["mind_change"]).expected_family == "cancel"


# ── Bracket (local server) ───────────────────────────────────────────


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()


@pytest.mark.parametrize("family", FAMILIES)
async def test_oracle_scores_one(medplum, family):
    for seed in (5, 10):
        r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family=family)
        assert r.reward == 1, (family, seed, r.diffs, r.error)


@pytest.mark.parametrize("family,knob", [(f, k) for f in FAMILIES for k in CHAOS_BY_FAMILY[f]])
async def test_oracle_scores_one_with_each_knob(medplum, family, knob):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=5, family=family, tier=3, chaos=[knob])
    assert r.reward == 1, (family, knob, r.diffs, r.error)


@pytest.mark.parametrize("family", ("booking", "cancel", "reschedule", "new_patient_intake", "emergency_redirect", "verify_fail_transfer"))
async def test_random_scores_zero_on_action_families(medplum, family):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), RandomPolicy(), seed=5, family=family)
    assert r.reward == 0, (family, r.diffs)


async def test_gated_tools_refuse_an_unverified_caller_and_the_echoed_dob(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking")
    try:
        blocked = await env.step_async(SchedulingAction(kind="tool", tool_name="look_up_availability"))
        assert "not verified" in blocked["tool_result"]["error"]
        echoed = await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": env.task.patient["dob"]}))
        assert echoed["tool_result"]["verified"] is False  # the caller never said it
        await env.step_async(SchedulingAction(kind="say", text="Your date of birth?"))
        ok = await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": env.task.patient["dob"]}))
        assert ok["tool_result"]["verified"] is True
    finally:
        await env.close()


async def test_claim_without_tool_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking")
    try:
        await env.step_async(SchedulingAction(kind="say", text="Your date of birth?"))
        await env.step_async(SchedulingAction(kind="tool", tool_name="verify_patient", arguments={"dob": env.task.patient["dob"]}))
        await env.step_async(SchedulingAction(kind="say", text="Great, you're booked for Monday at nine. Goodbye!"))
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booking_matches") for d in final["diffs"])
    finally:
        await env.close()


# ── v2 knobs (2026-09-29): the ones the clinic surfaces ────────────────


def _tool(name, **arguments):
    return SchedulingAction(kind="tool", tool_name=name, arguments=arguments)


async def _verify(env):
    await env.step_async(SchedulingAction(kind="say", text="Your date of birth?"))
    return await env.step_async(_tool("verify_patient", dob=env.task.patient["dob"]))


def test_new_knobs_generate_normalized_and_valid():
    t = generate("booking", 3, tier=3, today=TODAY, chaos=["slot_taken", "slot_taken_twice"])
    assert t.chaos == ["slot_taken_twice"]
    t = generate("booking", 3, tier=3, today=TODAY, chaos=["no_availability", "existing_visit", "dob_corrected"])
    assert t.chaos == ["dob_corrected", "no_availability"] and t.names_provider
    t = generate("cancel", 3, tier=3, today=TODAY, chaos=["shared_phone"])
    assert t.duplicate and t.duplicate["dob"] != t.patient["dob"] and t.duplicate["family"] == t.patient["family"]
    assert validate(t) == []
    for seed in range(1, 41):
        assert validate(generate("booking", seed, tier=3, today=TODAY)) == []


async def test_slot_taken_twice_rejects_two_writes_then_books(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["slot_taken_twice"])
    try:
        await _verify(env)
        codes = []
        for _ in range(3):
            await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
            r = (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]
            codes.append(r.get("status") or ("ok" if r.get("booked") else "?"))
        assert codes == [409, 409, "ok"], codes
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 1, final["diffs"]
    finally:
        await env.close()


async def test_shared_phone_hands_over_both_records_and_the_dob_picks(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    obs = await env.reset(seed=5, family="booking", tier=3, chaos=["shared_phone"])
    try:
        info = obs["patient_info"]
        assert info["multiple_patients"] and "patient_id" not in info and len(info["patients"]) == 2
        assert info["patients"][0]["patient_id"] == env.world.duplicate_patient_id
        # The other person's DOB, "spoken" on the call, verifies — as the OTHER person.
        env.caller_lines.append(f"It's {env.task.duplicate['dob']}.")
        r = await env.step_async(_tool("verify_patient", dob=env.task.duplicate["dob"]))
        assert r["tool_result"]["verified"] and env.patient_id == env.world.duplicate_patient_id
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        booked = await env.step_async(_tool("confirm_appointment", slot_index=0))
        assert booked["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("OTHER record" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


async def test_production_style_phone_lookup_meets_the_shared_phone(medplum):
    """What agent_worker does: lookup by caller ID, first match wins. Records which record it meets."""
    from runtime.inbound import resolve_caller_by_phone

    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="cancel", tier=3, chaos=["shared_phone"])
    try:
        caller = await resolve_caller_by_phone(env.world.tenant, env.world.patient.profile.phone)
        assert caller is not None and caller.patient_id is None, "a shared number must identify nobody"
        assert set(caller.candidate_ids) == {env.world.patient_id, env.world.duplicate_patient_id}
        print(f"\nlookup_patient_by_phone lists {caller.candidate_ids[0] == env.world.duplicate_patient_id and 'the DUPLICATE' or 'the right record'} first")
    finally:
        await env.close()


async def test_no_availability_named_provider_has_nothing_and_the_waitlist_is_the_answer(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["no_availability"])
    try:
        await _verify(env)
        found = (await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id)))["tool_result"]
        assert found["found"] is False and found["slots"] == []
        anyone = (await env.step_async(_tool("look_up_availability")))["tool_result"]
        assert anyone["found"], "the other providers are open — booking one of them is the wrong answer"
        booked = await env.step_async(_tool("confirm_appointment", slot_index=0))
        assert booked["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("waitlisted_not_booked") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()
