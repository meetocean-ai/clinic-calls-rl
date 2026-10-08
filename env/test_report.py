"""The EHR run report and scorecard (env/report.py, env/scorecard.py) — no server, no LLM.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. ../../services/agent/.venv/bin/python -m pytest env/test_report.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

from env.report import collect, render
from env.scorecard import load_run_dir, scorecard, wilson
import pytest

pytestmark = pytest.mark.covers("rl-scheduling-env")


def _run(tmp_path: Path) -> Path:
    folder = tmp_path / "run" / "production" / "booking-00005"
    folder.mkdir(parents=True)
    for trial, reward in enumerate([1, 0]):
        (folder / f"{trial}.manifest.json").write_text(json.dumps({
            "policy_id": "production", "family": "booking", "tier": 3, "task_id": "booking-00005", "seed": 5, "chaos": ["shared_phone"],
            "reward": reward, "diffs": [] if reward else ["booking_matches: booked on the OTHER record that shares the phone (the DOB did not pick the patient)"],
            "stopped_reason": "end_call", "turns": 3, "tool_calls": 2, "dataset_version": "tasks-v2+gen-abc", "method_version": "verify-def",
            "turn_scores": {"summary": {"claim_without_tool": {"turns": 2, "pass_rate": 0.5, "failed_turns": [2], "errors": 0}}},
        }))
        (folder / f"{trial}.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"i": -1, "observation": {"patient_text": "Hi, I'd like to book a visit."}},
            {"i": 0, "action": {"kind": "say", "text": "Your date of birth?", "tools": []}, "observation": {"patient_text": "January 1, 1990."}},
            {"i": 1, "action": {"kind": "say", "text": "You're all set for Monday.", "tools": ["verify_caller_dob", "book_appointment"]}, "observation": {"patient_text": "Thanks, bye."}},
        ]))
    return tmp_path / "run"


def test_scorecard_reads_a_run_dir_with_intervals_and_versions(tmp_path):
    run = _run(tmp_path)
    assert len(load_run_dir(run)["results"]) == 2
    text = scorecard(load_run_dir(run))
    assert "| production | booking | 3 | 1 | 2 | 0.5 | new | [0.095, 0.905] | 0.0 |" in text
    assert "| tasks-v2+gen-abc | verify-def |" in text
    assert wilson(3, 3) == (0.438, 1.0)


def test_report_has_the_ehr_diagram_the_miss_and_the_flagged_turn(tmp_path):
    run = _run(tmp_path)
    eps = collect([run])
    assert [e["reward"] for e in eps] == [1, 0] and eps[0]["flags"] == {"claim_without_tool": [2]}
    page = render([run], title="EHR Scheduling Bracket")
    assert page.startswith("<title>EHR Scheduling Bracket</title>")
    assert "drive_inbound" in page and "local Medplum" in page and 'aria-label="How a run works"' in page
    assert "booked on the OTHER record" in page and "Passes the judge flagged (1)" in page
    assert "<h2>By knob</h2>" in page and "shared_phone" in page and "DATASET.md" in page
    assert "<script" not in page


def _tier4_run(tmp_path: Path) -> Path:
    """Two policies; tier-4 tasks stacking knobs, so the per-knob rows pool across tasks and tiers."""
    root = tmp_path / "run4"
    rows = [
        ("production", "booking-00005", 3, ["shared_phone"], 1, []),
        ("production", "booking-00010", 4, ["exact_time_unavailable", "shared_phone"], 0,
         ["booking_not_at_requested_time: booked the exact time the caller asked for although it was taken (double booking)"]),
        ("production", "booking-00015", 4, ["exact_time_unavailable", "two_requests"], 0,
         ["two_requests_done: the visit with the other provider is still on the books (2030-01-15T10:00)"]),
        ("production", "booking-00020", 1, [], 1, []),
        ("oracle", "booking-00010", 4, ["exact_time_unavailable", "shared_phone"], 1, []),
    ]
    for pol, task, tier, chaos, reward, diffs in rows:
        folder = root / pol / task
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "0.manifest.json").write_text(json.dumps({
            "policy_id": pol, "family": "booking", "tier": tier, "task_id": task, "chaos": chaos, "reward": reward, "diffs": diffs,
            "stopped_reason": "end_call", "turns": 3, "tool_calls": 3, "dataset_version": "tasks-v3+gen-abc", "method_version": "verify-def",
            "turn_scores": {"summary": {}}}))
        (folder / "0.jsonl").write_text(json.dumps({"i": -1, "observation": {"patient_text": "Hi."}}) + "\n")
    return root


def test_knob_table_pools_episodes_across_tiers_and_names_the_failing_check(tmp_path):
    from env.scorecard import knob_rows, knob_table

    results = load_run_dir(_tier4_run(tmp_path))["results"]
    assert all("chaos" in r and "diffs" in r for r in results)
    rows = knob_rows(results)
    exact = rows[("production", "exact_time_unavailable")]
    assert exact["n"] == 2 and exact["tasks"] == 2 and exact["alone"] == 0 and exact["pass_rate"] == 0.0
    assert exact["top_diff"] == "booking_not_at_requested_time"
    shared = rows[("production", "shared_phone")]
    assert shared["n"] == 2 and shared["alone"] == 1 and shared["pass_rate"] == 0.5 and shared["pass_rate_ci"] == (0.095, 0.905)
    assert rows[("production", "— none —")]["n"] == 1 and rows[("oracle", "exact_time_unavailable")]["pass_rate"] == 1.0
    text = knob_table({"results": results})
    assert "| production | exact_time_unavailable | 2 | 2 | 0 | 0.0 | new | [0.0, 0.658] | booking_not_at_requested_time |" in text
    assert text.index("| production | shared_phone |") < text.index("| production | — none — |")  # knob-free rows last
