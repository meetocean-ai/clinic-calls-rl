"""Task set v2 (docs/plans/rl-speech-to-speech-roadmap.md, P0-02 …): every new knob has a generator test that
needs no server and a verifier test on the local Medplum, no LLM. Pass = the oracle scores 1; each failure the
knob is meant to catch is built by hand and scores 0 with the named diff.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/test_knobs_v2.py -v
"""
from __future__ import annotations

import os
from datetime import date, datetime

import pytest

from env.env import EhrSchedulingEnv, SchedulingAction
from env.policies import OraclePolicy, run_episode
from env.tasks import generate, normalize_chaos, validate
from env.world import clinic_today

pytestmark = [pytest.mark.covers("rl-env-tasks-v2")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")

TODAY = date(2030, 1, 7)
HELDOUT = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


def _tool(name, **arguments):
    return SchedulingAction(kind="tool", tool_name=name, arguments=arguments)


async def _verify(env):
    await env.step_async(SchedulingAction(kind="say", text="Your date of birth?"))
    return await env.step_async(_tool("verify_patient", dob=env.task.patient["dob"]))


@pytest.fixture
async def medplum():
    from adapters.medplum import MedplumAdapter

    m = await MedplumAdapter.from_env()
    try:
        yield m
    finally:
        await m.close()


# ── exact_time_unavailable (P0-02) ───────────────────────────────────────


def test_exact_time_generates_a_named_provider_and_a_weekday_slot_deterministically():
    for seed in range(1, 41):
        a = generate("booking", seed, tier=3, today=TODAY, chaos=["exact_time_unavailable"])
        b = generate("booking", seed, tier=3, today=TODAY, chaos=["exact_time_unavailable"])
        assert a.to_json() == b.to_json()
        assert validate(a) == [], (seed, validate(a))
        assert a.names_provider and a.requested_slot
        assert 2 <= a.requested_slot["days_ahead"] <= 11  # inside the 14-day search window from tomorrow
        assert date.fromisoformat(a.requested_slot["date"]).weekday() < 5
        assert a.provider["display"] in a.opening_line and "specifically" in a.goal


def test_exact_time_yields_to_the_knobs_it_cannot_coexist_with():
    assert normalize_chaos(["exact_time_unavailable", "no_availability"]) == ["no_availability"]
    assert normalize_chaos(["exact_time_unavailable", "mind_change"]) == ["mind_change"]
    t = generate("booking", 7, tier=3, today=TODAY, chaos=["exact_time_unavailable", "existing_visit", "dob_corrected"])
    assert t.chaos == ["dob_corrected", "exact_time_unavailable", "existing_visit"]
    assert "ADDITIONAL" in t.goal and "specifically" in t.goal and "misspeak" in t.goal
    plain = generate("booking", 7, tier=3, today=TODAY, chaos=["slot_taken"])
    assert plain.requested_slot is None and "specifically" not in plain.goal


def test_validate_rejects_a_requested_slot_without_the_knob():
    t = generate("booking", 3, tier=3, today=TODAY, chaos=["slot_taken"])
    t.requested_slot = {"days_ahead": 3, "local_hour": 10, "date": "2030-01-10"}
    assert "requested_slot mismatch" in validate(t)


@local_only
async def test_exact_time_slot_is_taken_and_the_search_shows_it(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["exact_time_unavailable"])
    try:
        req = env.task.requested_slot
        assert env.world.bystander_patient_id and env.world.requested_start
        await _verify(env)
        found = (await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id, date=req["date"])))["tool_result"]
        taken = (f"{req['date']}T{req['local_hour']:02d}:00", f"{req['date']}T{req['local_hour']:02d}:15")  # a 30-min visit
        same_day = [s for s in found["slots"] if s["start"].startswith(req["date"])]
        assert same_day, "the provider works that day — other times must be open"
        assert not any(s["start"].startswith(taken) for s in found["slots"]), ("the requested time is another patient's", found["slots"])
    finally:
        await env.close()


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_exact_time_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="booking",
                          tier=3, chaos=["exact_time_unavailable"])
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_exact_time_double_booking_scores_zero(medplum):
    """The product refuses this write (B-134); the environment still has to call it wrong if it ever happens."""
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["exact_time_unavailable"])
    try:
        start = datetime.fromisoformat(env.world.requested_start)
        await medplum.create("Appointment", {
            "resourceType": "Appointment", "status": "booked", "description": "Follow-up",
            "start": start.isoformat(), "end": (start.replace(minute=30)).isoformat(),
            "participant": [{"actor": {"reference": f"Patient/{env.world.patient_id}"}, "status": "accepted"},
                            {"actor": {"reference": f"Practitioner/{env.world.provider_id}"}, "status": "accepted"}],
        }, tenant_id=env.world.tenant.organization_id)
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booking_not_at_requested_time") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_exact_time_booking_another_provider_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["exact_time_unavailable"])
    try:
        await _verify(env)
        other = next(p["id"] for p in env.world.practitioners.values() if p["id"] != env.world.provider_id)
        found = (await env.step_async(_tool("look_up_availability", provider_id=other)))["tool_result"]
        assert found["found"]
        booked = (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]
        assert booked["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booking_matches: practitioner") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── two_requests (P0-03) ─────────────────────────────────────────────────


def test_two_requests_generates_a_visit_with_another_provider():
    for seed in range(1, 41):
        a = generate("booking", seed, tier=3, today=TODAY, chaos=["two_requests"])
        assert a.to_json() == generate("booking", seed, tier=3, today=TODAY, chaos=["two_requests"]).to_json()
        assert validate(a) == [], (seed, validate(a))
        assert a.names_provider and a.appointment and a.appointment["provider_display"] != a.provider["display"]
        assert a.appointment["provider_display"] in a.opening_line and a.provider["display"] in a.opening_line
        assert "TWO things" in a.goal
    assert normalize_chaos(["two_requests", "existing_visit"]) == ["two_requests"]
    assert normalize_chaos(["two_requests", "mind_change"]) == ["mind_change"]
    assert normalize_chaos(["two_requests", "no_availability"]) == ["no_availability"]
    both = generate("booking", 8, tier=3, today=TODAY, chaos=["two_requests", "exact_time_unavailable"])
    assert both.chaos == ["exact_time_unavailable", "two_requests"] and "TWO things" in both.goal and "specifically" in both.goal


def test_validate_rejects_a_named_visit_provider_without_two_requests():
    t = generate("booking", 3, tier=3, today=TODAY, chaos=["existing_visit"])
    t.appointment["provider_display"] = "Dr. Campos"
    assert any("without two_requests" in p for p in validate(t))


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_two_requests_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="booking",
                          tier=3, chaos=["two_requests"])
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_two_requests_seeds_the_visit_with_the_other_provider(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["two_requests"])
    try:
        await _verify(env)
        mine = (await env.step_async(_tool("get_appointments")))["tool_result"]["appointments"]
        assert len(mine) == 1 and mine[0]["appointment_id"] == env.world.target_appointment_id
        other = env.world.practitioners[env.task.appointment["provider_display"]]["id"]
        assert other != env.world.provider_id
        # What the agent reads back: the other provider's name and id, as the product's own bookings carry them.
        assert mine[0]["provider_id"] == other and mine[0]["provider"] == env.task.appointment["provider_display"], mine[0]
    finally:
        await env.close()


@local_only
async def test_two_requests_only_booking_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["two_requests"])
    try:
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("still on the books" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_two_requests_only_cancelling_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["two_requests"])
    try:
        await _verify(env)
        assert (await env.step_async(_tool("cancel_appointment", appointment_id=env.world.target_appointment_id)))["tool_result"]["cancelled"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("0 new bookings" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_two_requests_booking_with_the_old_provider_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["two_requests"])
    try:
        await _verify(env)
        await env.step_async(_tool("cancel_appointment", appointment_id=env.world.target_appointment_id))
        old = env.world.practitioners[env.task.appointment["provider_display"]]["id"]
        await env.step_async(_tool("look_up_availability", provider_id=old))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("not the provider the caller asked for" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_two_requests_moving_the_visit_onto_the_new_provider_passes(medplum):
    """Same end state for the patient (one visit, with the provider they asked for, old time gone) → reward 1."""
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["two_requests"])
    try:
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        moved = (await env.step_async(_tool("reschedule_appointment", appointment_id=env.world.target_appointment_id, slot_index=0)))["tool_result"]
        assert moved["rescheduled"], moved
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 1, final["diffs"]
    finally:
        await env.close()


# ── wrong_day_memory (P0-04) ─────────────────────────────────────────────


@pytest.mark.parametrize("family", ("cancel", "reschedule"))
def test_wrong_day_memory_generates_a_nearby_future_weekday(family):
    for seed in range(1, 41):
        a = generate(family, seed, tier=3, today=TODAY, chaos=["wrong_day_memory"])
        assert a.to_json() == generate(family, seed, tier=3, today=TODAY, chaos=["wrong_day_memory"]).to_json()
        assert validate(a) == [], (seed, validate(a))
        real, wrong = date.fromisoformat(a.appointment["date"]), date.fromisoformat(a.misremembered_date)
        assert wrong != real and wrong > TODAY and wrong.weekday() < 5 and abs((wrong - real).days) <= 4
        assert a.misremembered_date and str(wrong.day) in a.opening_line and "you're right" in a.goal
    plain = generate(family, 5, tier=3, today=TODAY, chaos=["dob_corrected"])
    assert plain.misremembered_date is None and "you're right" not in plain.goal


def test_validate_rejects_a_misremembered_date_that_is_the_real_one():
    t = generate("cancel", 5, tier=3, today=TODAY, chaos=["wrong_day_memory"])
    t.misremembered_date = t.appointment["date"]
    assert any("is the real one" in p for p in validate(t))


@local_only
@pytest.mark.parametrize("family", ("cancel", "reschedule"))
@pytest.mark.parametrize("seed", HELDOUT)
async def test_wrong_day_memory_oracle_scores_one(medplum, family, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family=family,
                          tier=3, chaos=["wrong_day_memory"])
    assert r.reward == 1, (family, seed, r.diffs, r.error)


@local_only
async def test_wrong_day_memory_booking_a_new_visit_instead_scores_zero(medplum):
    """The agent cannot find a visit on the day the caller named and books a fresh one: the real visit stays."""
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="reschedule", tier=3, chaos=["wrong_day_memory"])
    try:
        await _verify(env)
        mine = (await env.step_async(_tool("get_appointments")))["tool_result"]["appointments"]
        assert len(mine) == 1 and not mine[0]["start"].startswith(env.task.misremembered_date), "the visit is on the real day"
        on_file = env.world.practitioners[env.task.provider["display"]]["id"]  # the visit's provider, named or not
        await env.step_async(_tool("look_up_availability", provider_id=on_file))
        booked = (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]
        assert not booked["booked"] and "already has an upcoming visit" in booked["message"]  # B-106 guard says reschedule
        booked = (await env.step_async(_tool("confirm_appointment", slot_index=0, additional_visit=True)))["tool_result"]
        assert booked["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("old time still on the books" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── provider_name_collision (P0-05) ──────────────────────────────────────


def test_provider_name_collision_generates_a_prefix_namesake():
    for seed in range(1, 41):
        a = generate("booking", seed, tier=3, today=TODAY, chaos=["provider_name_collision"])
        assert a.to_json() == generate("booking", seed, tier=3, today=TODAY, chaos=["provider_name_collision"]).to_json()
        assert validate(a) == [], (seed, validate(a))
        assert a.names_provider and a.lookalike
        assert a.lookalike["family"].startswith(a.provider["family"]) and a.lookalike["family"] != a.provider["family"]
        assert a.lookalike["display"] in a.goal and "not your doctor" in a.goal
    plain = generate("booking", 5, tier=3, today=TODAY, chaos=["slot_taken"])
    assert plain.lookalike is None and "not your doctor" not in plain.goal


@local_only
async def test_provider_name_collision_seeds_both_providers_and_the_history(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["provider_name_collision"])
    try:
        from runtime.end_state import appointments_for

        assert env.task.lookalike["display"] in env.world.practitioners
        twin = env.world.practitioners[env.task.lookalike["display"]]["id"]
        assert twin != env.world.provider_id
        history = await appointments_for(medplum, env.world.tenant, env.world.patient_id)
        assert [a["status"] for a in history] == ["fulfilled"], "one past visit, with the named provider"
        assert any(p["actor"]["reference"] == f"Practitioner/{env.world.provider_id}" for p in history[0]["participant"])
        await _verify(env)
        theirs = (await env.step_async(_tool("look_up_availability", provider_id=twin)))["tool_result"]
        assert theirs["found"], "the namesake has open shifts too — that is what makes the wrong booking possible"
    finally:
        await env.close()


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_provider_name_collision_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="booking",
                          tier=3, chaos=["provider_name_collision"])
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_provider_name_collision_booking_the_namesake_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["provider_name_collision"])
    try:
        await _verify(env)
        twin = env.world.practitioners[env.task.lookalike["display"]]["id"]
        await env.step_async(_tool("look_up_availability", provider_id=twin))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booking_matches: practitioner") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_provider_name_collision_spoken_name_resolves_to_the_exact_match(medplum):
    """What the product does with the NAME the caller says (`tools_fhir._resolve_practitioner_id`, B-081): with a
    namesake in the tenant, "Dr. Eddins" must resolve to Eddins, not to whichever Practitioner the search lists first."""
    from agents.tools_fhir import _resolve_practitioner_id

    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["provider_name_collision"])
    try:
        resolved = await _resolve_practitioner_id(env.task.provider["display"], organization_id=env.world.tenant.organization_id)
        twin = env.world.practitioners[env.task.lookalike["display"]]["id"]
        assert resolved == env.world.provider_id, (
            f"{env.task.provider['display']!r} resolved to {'the namesake' if resolved == twin else resolved!r}")
    finally:
        await env.close()


# ── insurance_detour (P0-06) ─────────────────────────────────────────────


@pytest.mark.parametrize("family", ("booking", "reschedule"))
def test_insurance_detour_generates_a_question_the_caller_drops_after_the_answer(family):
    for seed in range(1, 41):
        a = generate(family, seed, tier=3, today=TODAY, chaos=["insurance_detour"])
        assert a.to_json() == generate(family, seed, tier=3, today=TODAY, chaos=["insurance_detour"]).to_json()
        assert validate(a) == [], (seed, validate(a))
        assert a.detour_question and a.detour_question in a.goal and "go straight back" in a.goal
    assert normalize_chaos(["insurance_detour", "mind_change"]) == ["mind_change"]
    assert generate("booking", 5, tier=3, today=TODAY, chaos=["slot_taken"]).detour_question is None


@local_only
@pytest.mark.parametrize("family", ("booking", "reschedule"))
@pytest.mark.parametrize("seed", HELDOUT)
async def test_insurance_detour_oracle_scores_one(medplum, family, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family=family,
                          tier=3, chaos=["insurance_detour"])
    assert r.reward == 1, (family, seed, r.diffs, r.error)


@local_only
async def test_insurance_detour_filing_an_insurance_task_scores_zero(medplum):
    """The billing specialist's `note_insurance_update` leaves a Task for staff — right for a reported change, wrong
    for a question. Booked correctly otherwise."""
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["insurance_detour"])
    try:
        from agents.tools_fhir import create_task

        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        await create_task(action_code="insurance-update", action_display="Patient-reported insurance update",
                          patient_id=env.world.patient_id, patient_display=env.world.patient.profile.name,
                          description="asked about copay", organization_id=env.world.tenant.organization_id)
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("no_billing_write: an insurance-update Task") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_insurance_detour_writing_coverage_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["insurance_detour"])
    try:
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        await medplum.create("Coverage", {"resourceType": "Coverage", "status": "active",
                                          "beneficiary": {"reference": f"Patient/{env.world.patient_id}"},
                                          "payor": [{"display": "Harborline PPO"}]}, tenant_id=env.world.tenant.organization_id)
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("no_billing_write: Coverage") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── provider_switch_after_search (P0-07) ─────────────────────────────────


def test_provider_switch_generates_a_second_named_provider():
    for seed in range(1, 41):
        a = generate("booking", seed, tier=3, today=TODAY, chaos=["provider_switch_after_search"])
        assert a.to_json() == generate("booking", seed, tier=3, today=TODAY, chaos=["provider_switch_after_search"]).to_json()
        assert validate(a) == [], (seed, validate(a))
        assert a.names_provider and a.switch_to and a.switch_to["display"] != a.provider["display"]
        assert a.final_provider == a.switch_to
        assert a.provider["display"] in a.opening_line and a.switch_to["display"] in a.goal and "rather see" in a.goal
    for other in ("no_availability", "two_requests", "provider_name_collision", "mind_change"):
        assert "provider_switch_after_search" not in normalize_chaos(["provider_switch_after_search", other]), other
    assert normalize_chaos(["provider_switch_after_search", "exact_time_unavailable"]) == ["provider_switch_after_search"]
    assert generate("booking", 5, tier=3, today=TODAY, chaos=["slot_taken"]).final_provider == generate("booking", 5, tier=3, today=TODAY, chaos=["slot_taken"]).provider


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_provider_switch_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="booking",
                          tier=3, chaos=["provider_switch_after_search"])
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_provider_switch_booking_the_first_provider_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["provider_switch_after_search"])
    try:
        first = env.world.practitioners[env.task.provider["display"]]["id"]
        assert env.world.provider_id == env.world.practitioners[env.task.switch_to["display"]]["id"] != first
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=first))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booking_matches: practitioner") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_provider_switch_booking_both_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["provider_switch_after_search"])
    try:
        first = env.world.practitioners[env.task.provider["display"]]["id"]
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=first))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0, additional_visit=True)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("2 new bookings" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── new_patient_intake (P0-08) ───────────────────────────────────────────

INTAKE_CHAOS = (None, ["shared_phone"], ["dob_corrected"], ["no_availability"])


def test_intake_generates_without_a_visit_on_file():
    from env.tasks import CHAOS_BY_FAMILY

    assert CHAOS_BY_FAMILY["new_patient_intake"] == ("shared_phone", "dob_corrected", "no_availability")
    for seed in range(1, 41):
        for tier in (1, 2, 3):
            a = generate("new_patient_intake", seed, tier=tier, today=TODAY)
            assert a.to_json() == generate("new_patient_intake", seed, tier=tier, today=TODAY).to_json()
            assert validate(a) == [], (seed, tier, validate(a))
            assert a.appointment is None and "NEW patient" in a.goal and a.patient["given"] in a.goal
            assert a.opening_line.startswith("Hi, I'm a new patient")
    t = generate("new_patient_intake", 5, tier=3, today=TODAY, chaos=["dob_corrected"])
    assert t.misstated_dob and "sorry, I mean" in t.goal
    t = generate("new_patient_intake", 5, tier=3, today=TODAY, chaos=["no_availability"])
    assert t.names_provider and "waitlist" in t.goal


@local_only
async def test_intake_world_has_no_record_and_the_tools_say_so(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    obs = await env.reset(seed=5, family="new_patient_intake")
    try:
        assert obs["patient_info"] == {"unknown_caller": True, "phone_number": env.task.patient["phone"], "patient_id": None, "verified": False}
        assert env.world.patient_id is None and "create_patient" in obs["tools"]
        blocked = await env.step_async(_tool("look_up_availability"))
        assert "not verified" in blocked["tool_result"]["error"]
        await env.step_async(SchedulingAction(kind="say", text="Your name and date of birth?"))
        r = (await env.step_async(_tool("verify_patient", dob=env.task.patient["dob"])))["tool_result"]
        assert r["verified"] is False and "create_patient" in r["message"]
        unsaid = (await env.step_async(_tool("create_patient", first_name="Someone", last_name="Else", dob=env.task.patient["dob"])))["tool_result"]
        assert unsaid["created"] is False and unsaid["reason"] == "name_not_spoken"
        made = (await env.step_async(_tool("create_patient", first_name=env.task.patient["given"], last_name=env.task.patient["family"],
                                           dob=env.task.patient["dob"])))["tool_result"]
        assert made["created"] and env.verified and env.patient_id == made["patient_id"]
        again = (await env.step_async(_tool("create_patient", first_name=env.task.patient["given"], last_name=env.task.patient["family"],
                                            dob=env.task.patient["dob"])))["tool_result"]
        assert again["created"] is False and again["reason"] == "already_verified"
    finally:
        await env.close()


@local_only
async def test_intake_shared_phone_hands_over_the_other_person(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    obs = await env.reset(seed=5, family="new_patient_intake", tier=3, chaos=["shared_phone"])
    try:
        info = obs["patient_info"]
        assert info["patient_id"] == env.world.duplicate_patient_id and info["first_name"] == env.task.duplicate["given"]
        # Treating the caller as that person and booking for them is the wrong end state.
        env.caller_lines.append(f"It's {env.task.duplicate['dob']}.")
        assert (await env.step_async(_tool("verify_patient", dob=env.task.duplicate["dob"])))["tool_result"]["verified"]
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("patient_created_once: no Patient") for d in final["diffs"]), final["diffs"]
        assert any("OTHER record" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
@pytest.mark.parametrize("chaos", INTAKE_CHAOS, ids=lambda c: ",".join(c) if c else "plain")
@pytest.mark.parametrize("seed", HELDOUT)
async def test_intake_oracle_scores_one(medplum, seed, chaos):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="new_patient_intake",
                          tier=3 if chaos else 1, chaos=chaos)
    assert r.reward == 1, (seed, chaos, r.diffs, r.error)


@local_only
async def test_intake_registering_the_misspoken_dob_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="new_patient_intake", tier=3, chaos=["dob_corrected"])
    try:
        heard = await env.step_async(SchedulingAction(kind="say", text="Your full name and date of birth?"))
        assert env.task.misstated_dob in heard["patient_text"]
        made = (await env.step_async(_tool("create_patient", first_name=env.task.patient["given"], last_name=env.task.patient["family"],
                                           dob=env.task.misstated_dob)))["tool_result"]
        assert made["created"]
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("DOB the caller misspoke" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_intake_two_records_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="new_patient_intake")
    try:
        from agents.tools_patient import register_patient_in_org

        await env.step_async(SchedulingAction(kind="say", text="Your full name and date of birth?"))
        p = env.task.patient
        assert (await env.step_async(_tool("create_patient", first_name=p["given"], last_name=p["family"], dob=p["dob"])))["tool_result"]["created"]
        # a second registration on the same number (a typo'd name slips past the product's name+DOB duplicate check)
        dup = await register_patient_in_org(p["given"] + "e", p["family"], p["dob"], phone=p["phone"], organization_id=env.world.tenant.organization_id)
        assert dup["created"]
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("2 Patient records" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── emergency_redirect (P0-08b) + urgent_same_week ────────────────────────


def test_emergency_redirect_generates_a_red_flag_and_expects_a_transfer():
    for seed in range(1, 41):
        for tier in (1, 2, 3):
            a = generate("emergency_redirect", seed, tier=tier, today=TODAY)
            assert a.to_json() == generate("emergency_redirect", seed, tier=tier, today=TODAY).to_json()
            assert validate(a) == [], (seed, tier, validate(a))
            assert a.red_flag and a.expect_transfer and a.appointment is None and a.chaos == []
            assert "911" in a.goal and a.opening_line.startswith("Hi... I was hoping to see the doctor today")
    assert len({generate("emergency_redirect", s, today=TODAY).red_flag for s in range(1, 41)}) == 4


def test_urgent_same_week_generates_a_soonest_request():
    for seed in range(1, 41):
        a = generate("booking", seed, tier=3, today=TODAY, chaos=["urgent_same_week"])
        assert validate(a) == [] and a.urgent and "soonest" in a.goal and "within the next 2 days" in a.goal
    assert normalize_chaos(["urgent_same_week", "exact_time_unavailable"]) == ["exact_time_unavailable"]
    assert normalize_chaos(["urgent_same_week", "no_availability"]) == ["no_availability"]
    assert not generate("booking", 5, tier=3, today=TODAY, chaos=["slot_taken"]).urgent


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_emergency_redirect_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="emergency_redirect")
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_emergency_redirect_booking_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="emergency_redirect")
    try:
        await _verify(env)
        await env.step_async(_tool("look_up_availability"))
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("emergency_escalated: booked") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_emergency_redirect_routine_transfer_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="emergency_redirect")
    try:
        final = await env.step_async(_tool("warm_transfer_to_human", reason="caller asked for a person"))
        assert final["done"] and final["reward"] == 0
        assert any("without an urgent reason" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
async def test_emergency_redirect_hanging_up_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="emergency_redirect")
    try:
        await env.step_async(SchedulingAction(kind="say", text="Please hang up and call 911."))
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("no transfer to a person" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


@local_only
@pytest.mark.parametrize("seed", HELDOUT)
async def test_urgent_same_week_oracle_scores_one(medplum, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family="booking",
                          tier=3, chaos=["urgent_same_week"])
    assert r.reward == 1, (seed, r.diffs, r.error)


@local_only
async def test_urgent_same_week_booking_next_week_scores_zero(medplum):
    from datetime import date, timedelta

    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["urgent_same_week"])
    try:
        await _verify(env)
        later = clinic_today() + timedelta(days=7)
        while later.weekday() >= 5:
            later += timedelta(days=1)
        found = (await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id, date=later.isoformat())))["tool_result"]
        assert found["found"]
        assert (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any(d.startswith("booked_soon_enough") for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()


# ── tier 4: knob combinations (P0-09) ────────────────────────────────────

HELDOUT_20 = tuple(range(5, 101, 5))


def test_tier4_draws_two_or_three_composable_knobs_per_family():
    from collections import Counter

    from env.tasks import CHAOS_BY_FAMILY, FAMILIES, normalize_chaos

    for family in FAMILIES:
        menu = CHAOS_BY_FAMILY[family]
        seen: Counter = Counter()
        for seed in range(1, 61):
            a = generate(family, seed, tier=4, today=TODAY)
            assert a.to_json() == generate(family, seed, tier=4, today=TODAY).to_json()
            assert validate(a) == [], (family, seed, validate(a))
            assert a.chaos == normalize_chaos(a.chaos) and len(a.chaos) <= 3
            assert len(a.chaos) >= min(2, len(menu)), (family, seed, a.chaos)
            seen.update(a.chaos)
        if len(menu) >= 2:
            assert len(seen) >= min(len(menu), 4), (family, "tier 4 should spread over the menu", seen)
    with pytest.raises(ValueError):
        generate("booking", 3, tier=2, today=TODAY, chaos=["slot_taken"])
    explicit = generate("booking", 3, tier=4, today=TODAY, chaos=["two_requests", "exact_time_unavailable", "dob_corrected"])
    assert explicit.chaos == ["dob_corrected", "exact_time_unavailable", "two_requests"] and validate(explicit) == []


def test_validate_rejects_a_single_knob_at_tier4():
    t = generate("booking", 3, tier=4, today=TODAY, chaos=["slot_taken"])
    assert "tier 4 needs a combination of knobs" in validate(t)


@local_only
@pytest.mark.parametrize("family", ("booking", "cancel", "reschedule", "new_patient_intake"))
@pytest.mark.parametrize("seed", HELDOUT_20)
async def test_tier4_oracle_scores_one(medplum, family, seed):
    r = await run_episode(EhrSchedulingEnv(medplum, scripted_caller=True), OraclePolicy(), seed=seed, family=family, tier=4)
    assert r.reward == 1, (family, seed, generate(family, seed, tier=4).chaos, r.diffs, r.error)


# ── dataset v2 freeze (P0-11) ────────────────────────────────────────────


def test_dataset_md_matches_the_generator():
    """DATASET.md pins the generator hash. Changing tasks.py / world.py without bumping the schema and the line in
    DATASET.md is a silent dataset change — this fails until both are updated."""
    import re
    from pathlib import Path

    from env.versioning import dataset_version

    text = (Path(__file__).parent / "DATASET.md").read_text()
    pinned = re.search(r"`dataset_version: (tasks-v\d+\+gen-[0-9a-f]{10})`", text)
    assert pinned, "DATASET.md must carry the `dataset_version: …` line"
    assert pinned.group(1) == dataset_version(), (
        f"generator is {dataset_version()}, DATASET.md says {pinned.group(1)} — bump EhrTask.schema_version and the doc")


def test_heldout_eval_seeds_are_all_held_out():
    from env.tasks import HELDOUT_EVAL_SEEDS, split_for

    assert len(HELDOUT_EVAL_SEEDS) == 10 and all(split_for(s) == "heldout" for s in HELDOUT_EVAL_SEEDS)
    assert HELDOUT_EVAL_SEEDS == HELDOUT  # the tests above evaluate on the frozen set


# ── caller realism + cooperation (P0-10) ─────────────────────────────────


def test_realism_traits_follow_the_tier_and_the_persona():
    from collections import Counter

    from env.tasks import COOPERATION_RANGE, FAMILIES, REALISM, REALISM_PER_TIER, TIERS

    seen: Counter = Counter()
    personas: Counter = Counter()
    for family in FAMILIES:
        for tier in TIERS:
            lo, hi = REALISM_PER_TIER[tier]
            for seed in range(1, 31):
                a = generate(family, seed, tier=tier, today=TODAY)
                assert validate(a) == [], (family, tier, seed, validate(a))
                assert lo <= len(a.realism) <= hi and a.realism == sorted(set(a.realism))
                clo, chi = COOPERATION_RANGE[a.persona]
                assert clo <= a.cooperation <= chi
                assert (a.payer is not None) == ("hidden_until_asked" in a.realism)
                rules = a.constraints()
                assert rules[0].startswith(f"Cooperation {a.cooperation}/5") and len(rules) == 1 + len(a.realism)
                assert all("{payer}" not in r for r in rules)
                if tier == 1:
                    assert a.realism == []
                    if family == "emergency_redirect":  # the presentation sets the persona
                        assert a.persona == "anxious" and 3 <= a.cooperation <= 4
                    else:
                        assert a.persona == "polite" and a.cooperation == 5
                seen.update(a.realism)
                personas[a.persona] += 1
    assert set(seen) == set(REALISM), seen
    assert {"hostile", "demanding", "elderly"} <= set(personas), personas


def test_realism_never_touches_the_end_state_the_verifier_expects():
    """Same seed, same family, same tier: the realism traits vary with nothing the verifier reads."""
    from dataclasses import replace

    a = generate("booking", 5, tier=4, today=TODAY)
    b = replace(a, realism=[], cooperation=5, payer=None)
    for key in ("patient", "provider", "chaos", "appointment", "requested_slot", "switch_to", "expected_family"):
        assert getattr(a, key) == getattr(b, key)


def test_validate_rejects_realism_out_of_tier_and_cooperation_out_of_persona():
    t = generate("booking", 5, tier=1, today=TODAY)
    t.realism = ["fillers"]
    assert any("tier 1 allows 0–0" in p for p in validate(t))
    t = generate("booking", 5, tier=2, today=TODAY)
    t.cooperation = 1 if t.persona not in ("hostile", "demanding") else 5
    assert any("cooperation" in p for p in validate(t))


@local_only
async def test_realism_rules_reach_the_simulator_profile(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=4)
    try:
        assert env.world.patient.profile.constraints == env.task.constraints()
        assert env.world.patient.profile.persona == env.task.persona
    finally:
        await env.close()
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=10, family="new_patient_intake", tier=3)
    try:
        assert env.world.patient.profile.constraints == env.task.constraints()
    finally:
        await env.close()


# ── verify_fail_transfer hardening (P0-09b) ──────────────────────────────


@local_only
async def test_verify_fail_transfer_needs_real_attempts_before_the_transfer(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="verify_fail_transfer")
    try:
        final = await env.step_async(SchedulingAction(kind="transfer"))
        assert final["reward"] == 0 and any("0 verification attempt" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="verify_fail_transfer")
    try:
        for _ in range(2):
            heard = await env.step_async(SchedulingAction(kind="say", text="Could I have your date of birth?"))
            assert env.task.claimed_dob in heard["patient_text"]
            r = (await env.step_async(_tool("verify_patient", dob=env.task.claimed_dob)))["tool_result"]
            assert r["verified"] is False
        final = await env.step_async(SchedulingAction(kind="transfer"))
        assert final["reward"] == 1, final["diffs"]
    finally:
        await env.close()


@local_only
async def test_exact_time_cancelling_the_other_patient_scores_zero(medplum):
    env = EhrSchedulingEnv(medplum, scripted_caller=True)
    await env.reset(seed=5, family="booking", tier=3, chaos=["exact_time_unavailable"])
    try:
        from runtime.end_state import appointments_for

        theirs = await appointments_for(medplum, env.world.tenant, env.world.bystander_patient_id, status="booked")
        assert len(theirs) == 1
        await medplum.update("Appointment", {**theirs[0], "status": "cancelled"}, tenant_id=env.world.tenant.organization_id)
        await _verify(env)
        await env.step_async(_tool("look_up_availability", provider_id=env.world.provider_id, date=env.task.requested_slot["date"]))
        booked = (await env.step_async(_tool("confirm_appointment", slot_index=0)))["tool_result"]
        assert booked["booked"]
        final = await env.step_async(SchedulingAction(kind="end_call"))
        assert final["reward"] == 0 and any("other patient's visit at that time is gone" in d for d in final["diffs"]), final["diffs"]
    finally:
        await env.close()
