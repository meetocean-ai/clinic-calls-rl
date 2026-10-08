"""The clinic-calls builder (roadmap P2-08): a small package built from the env has every file the card promises, the
records validate against the schema, and the shipped verifier — run as a standalone script, standard library only —
re-grades the reference run to the same verdict.

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m pytest env/dataset/test_build.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from env.dataset.build import SCHEMA, build, caller_script, dataset_card
from env.tasks import generate

pytestmark = [pytest.mark.covers("clinic-calls-benchmark")]
local_only = pytest.mark.skipif("localhost" not in os.environ.get("MEDPLUM_BASE_URL", ""),
                                reason="local Medplum only (never run this against a hosted server)")


def test_caller_script_covers_the_facts_a_voice_policy_must_hear():
    t = generate("new_patient_intake", 5, tier=3)
    kinds = [l["kind"] for l in caller_script(t)]
    assert kinds[0] == "opening" and kinds[-2:] == ["yes", "goodbye"] and "name" in kinds and "dob" in kinds
    assert any(t.claimed_dob in l["text"] for l in caller_script(t))
    if "dob_corrected" in t.chaos:
        assert "dob_misstated" in kinds and kinds.index("dob_misstated") < kinds.index("dob")
    b = generate("booking", 10, tier=1)
    assert [l["kind"] for l in caller_script(b)] == ["opening", "dob", "yes", "goodbye"]


def test_schema_and_card_are_well_formed():
    assert SCHEMA["required"] and all(k in SCHEMA["properties"] for k in SCHEMA["required"])
    card = dataset_card({"tasks": 2, "by_family": {"booking": 2}, "by_tier": {"3": 2}, "audio_lines": 8, "reference_pass": 2},
                        {"dataset_version": "d", "method_version": "m", "verifier_version": "v"}, tts="tone")
    assert "CC BY-NC 4.0" in card and "PolyForm Noncommercial 1.0.0" in card and "| booking | 2 |" in card and "verify_json.py" in card


@local_only
async def test_build_writes_a_gradeable_package(tmp_path):
    out = tmp_path / "clinic-calls"
    manifest = await build(out, split="heldout", n_seeds=1, tiers=[3], families=["booking", "cancel"], tts="tone")
    assert manifest["counts"] == {"tasks": 2, "by_family": {"booking": 1, "cancel": 1}, "by_tier": {"3": 2}, "audio_lines": manifest["counts"]["audio_lines"], "reference_pass": 2}
    assert manifest["licenses"] == {"data": "CC-BY-NC-4.0", "code": "PolyForm-Noncommercial-1.0.0"} and manifest["tts"] == "tone"
    for f in ("README.md", "LICENSE", "LICENSE-DATA", "DATASET.md", "schema.json", "manifest.json", "tasks/heldout.jsonl", "verifier/verify_json.py"):
        assert (out / f).exists(), f
    assert "PolyForm Noncommercial License 1.0.0" in (out / "LICENSE").read_text() and "CC BY-NC 4.0" in (out / "LICENSE-DATA").read_text()
    records = [json.loads(l) for l in (out / "tasks" / "heldout.jsonl").read_text().splitlines()]
    assert [r["id"] for r in records] == ["booking-00005", "cancel-00005"] and [r["key"] for r in records] == ["booking-00005-t3", "cancel-00005-t3"]
    for r in records:
        for k in SCHEMA["required"]:
            assert k in r, k
        assert r["reference"]["reward"] == 1 and r["checks"][0] in ("booking_matches", "waitlisted_not_booked", "two_requests_done", "cancel_targeted_right_id")
        assert (out / r["files"]["snapshot"]).exists() and (out / r["files"]["reference_run"] / "oracle.manifest.json").exists()
        assert r["files"]["audio"] and all((out / a).exists() for a in r["files"]["audio"])
        script = json.loads((out / "audio" / r["key"] / "script.json").read_text())
        assert script["voice"]["tts"] == "tone" and len(script["lines"]) == len(r["files"]["audio"]) and script["lines"][0]["kind"] == "opening"
        snap = json.loads((out / r["files"]["snapshot"]).read_text())
        assert snap["task"]["id"] == r["id"] and snap["bundle"]["resourceType"] == "Bundle" and snap["world"]["baseline"]["n_patients"] >= 1
        m = json.loads((out / r["files"]["reference_run"] / "oracle.manifest.json").read_text())
        assert m["reward"] == 1 and m["json_verifier"]["reward"] == 1 and m["dataset_version"] == manifest["dataset_version"]
    assert set(manifest["files"]) >= {"README.md", "tasks/heldout.jsonl", "snapshots/booking-00005-t3.json"}

    # the shipped verifier grades the reference run on its own: a fresh interpreter, isolated mode, nothing on the path
    ref = out / "reference_runs" / "cancel-00005-t3"
    turns = json.loads((ref / "oracle.turns.json").read_text())
    (tmp_path / "turns.json").write_text(json.dumps(turns["agent_turns"]))
    proc = subprocess.run([sys.executable, "-I", str(out / "verifier" / "verify_json.py"), str(out / "snapshots" / "cancel-00005-t3.json"),
                           str(ref / "oracle.end_state.json"), "--turns", str(tmp_path / "turns.json"),
                           "--verify-attempts", str(turns["extra"]["verify_attempts"])], capture_output=True, text=True, check=True)
    verdict = json.loads(proc.stdout)
    assert verdict["reward"] == 1 and verdict["diffs"] == [] and verdict["checks_run"][0] == "cancel_targeted_right_id"
    # and it names the miss when the end state is the snapshot itself (nothing happened)
    proc = subprocess.run([sys.executable, "-I", str(out / "verifier" / "verify_json.py"), str(out / "snapshots" / "cancel-00005-t3.json"),
                           str(out / "snapshots" / "cancel-00005-t3.json")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    # a snapshot file is {"task", "world", "bundle"}: the verifier takes a Bundle or a list — hand it the bundle inside
    snap = json.loads((out / "snapshots" / "cancel-00005-t3.json").read_text())
    (tmp_path / "before.json").write_text(json.dumps(snap["bundle"]))
    proc = subprocess.run([sys.executable, "-I", str(out / "verifier" / "verify_json.py"), str(out / "snapshots" / "cancel-00005-t3.json"),
                           str(tmp_path / "before.json")], capture_output=True, text=True, check=True)
    nothing = json.loads(proc.stdout)
    assert nothing["reward"] == 0 and nothing["diffs"][0].startswith("cancel_targeted_right_id:") and "is not cancelled" in nothing["diffs"][0]
