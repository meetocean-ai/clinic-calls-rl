"""Everything scored, versioned, logged — no server, no LLM.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. ../../services/agent/.venv/bin/python -m pytest env/test_scoring_meta.py -v
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from env.env import EhrSchedulingEnv
from env.policies import EpisodeResult, summarize
from env.tasks import generate
from env.turn_scores import ClaimWithoutToolJudge, score_turns, turns_of
from env.versioning import dataset_version, method_version
import pytest

pytestmark = pytest.mark.covers("rl-env-reward")

TODAY = date(2030, 1, 7)


# ── versions ──────────────────────────────────────────────────────────────


def test_versions_have_the_shape_and_are_stable():
    assert dataset_version().startswith("tasks-v3+gen-") and len(dataset_version().split("gen-")[1]) == 10
    assert dataset_version() == dataset_version()
    assert method_version().startswith("verify-")
    assert method_version("gpt-4.1") != method_version() and method_version("gpt-4.1").endswith("+judge-gpt-4.1")


# ── per-turn scoring ──────────────────────────────────────────────────────


def _env_steps(claim_after_write: bool):
    steps = [
        {"i": 0, "action": {"kind": "tool", "tool_name": "look_up_availability", "arguments": {}}, "observation": {"tool_result": {"slots": [{"start": "x"}]}}},
        {"i": 1, "action": {"kind": "say", "text": "I have 9 AM. Does that work?"}, "observation": {"patient_text": "Yes."}},
    ]
    if claim_after_write:
        steps.append({"i": 2, "action": {"kind": "tool", "tool_name": "confirm_appointment", "arguments": {"slot_index": 0}},
                      "observation": {"tool_result": {"booked": True}}})
    steps.append({"i": 3, "action": {"kind": "say", "text": "You're all set. Anything else?"}, "observation": {"patient_text": "No, bye."}})
    return steps


def test_turns_are_rebuilt_from_env_steps():
    turns = turns_of(_env_steps(True), {"patient_text": "Hi, I'd like to book."})
    assert [t.index for t in turns] == [1, 2]
    assert [t["name"] for t in turns[0].tools] == ["look_up_availability"]
    assert [t["name"] for t in turns[1].tools] == ["confirm_appointment"]
    assert turns[1].items[0] == {"role": "user", "content": "Hi, I'd like to book."}


def test_the_hallucinated_confirmation_fails_only_the_claiming_turn():
    bad = score_turns(_env_steps(False))
    assert bad["summary"]["claim_without_tool"] == {"turns": 2, "pass_rate": 0.5, "failed_turns": [2], "errors": 0}
    good = score_turns(_env_steps(True))
    assert good["summary"]["claim_without_tool"]["failed_turns"] == []


def test_production_steps_carry_tool_names_and_a_write_in_an_earlier_turn_backs_a_later_claim():
    steps = [
        {"i": 0, "action": {"kind": "say", "text": "I'll cancel it now.", "tools": ["cancel_appointment"]}, "observation": {"patient_text": "Thanks."}},
        {"i": 1, "action": {"kind": "say", "text": "Your appointment has been cancelled. Goodbye!", "tools": []}, "observation": {"patient_text": "Bye."}},
    ]
    s = score_turns(steps)["summary"]["claim_without_tool"]
    assert s["failed_turns"] == [] and s["pass_rate"] == 1.0
    no_write = [dict(steps[1], action={"kind": "say", "text": "Your appointment has been cancelled.", "tools": []})]
    assert score_turns(no_write)["summary"]["claim_without_tool"]["failed_turns"] == [1]


def test_a_broken_judge_is_logged_not_fatal():
    class Broken:
        name = "broken"

        def evaluate(self, turn):
            raise RuntimeError("down")

    s = score_turns(_env_steps(True), judges=[ClaimWithoutToolJudge(), Broken()])
    assert s["summary"]["broken"] == {"turns": 0, "pass_rate": None, "failed_turns": [], "errors": 2}
    assert s["summary"]["claim_without_tool"]["pass_rate"] == 1.0


# ── the manifest carries all of it ────────────────────────────────────────


def test_manifest_is_versioned_and_turn_scored():
    task = generate("booking", 5, tier=1, today=TODAY)
    stub = SimpleNamespace(task=task, simulator=SimpleNamespace(), policy_id="stub", steps=_env_steps(False),
                           initial_observation={"patient_text": "Hi."}, stopped_reason="end_call",
                           result=SimpleNamespace(reward=0, diffs=["booking_matches: none"]), turn=2, tool_calls=1, seconds=1.2)
    m = EhrSchedulingEnv.manifest(stub)
    assert m["dataset_version"] == dataset_version() and m["method_version"] == method_version()
    assert m["turn_scores"]["summary"]["claim_without_tool"]["failed_turns"] == [2]
    assert m["reward"] == 0 and m["environment"] == "ehr" and m["seconds"] == 1.2 and m["latency_ms"]["p50"] is None


def test_summary_shows_judge_means_and_versions():
    def r(reward, rate):
        return EpisodeResult(task_id="booking-00005", family="booking", seed=5, split="heldout", tier=1, trial=0, policy_id="p",
                             reward=reward, diffs=[], turns=3, tool_calls=2, stopped_reason="end_call", seconds=1.0,
                             dataset_version="tasks-v1+gen-a", method_version="verify-b",
                             turn_scores={"claim_without_tool": {"turns": 2, "pass_rate": rate, "failed_turns": [], "errors": 0}})

    row = summarize([r(1, 1.0), r(0, 0.5)])["p"]["booking"]
    assert row["turn_judges"] == {"claim_without_tool": 0.75} and row["versions"] == ["tasks-v1+gen-a|verify-b"]
