"""Build the `clinic-calls` package from the environment (roadmap P2-08).

    cd tests/behavioral && PYTHONPATH=../../services/agent:. python -m env.dataset.build \
        --out results/env/clinic-calls-v1 --split heldout --seeds 10 --tiers 1,3,4 [--families …] [--tts kokoro|tone|none]

Per task: generate it, seed a fresh tenant, snapshot the compartment (`snapshots/<task_id>-t<tier>.json` = task + world facts +
FHIR Bundle), run the oracle as the reference policy, export the end state, grade it with BOTH the live verifier and
the shipped pure-JSON verifier (a disagreement fails the build), write the reference run (`reference_runs/<task_id>-t<tier>/
oracle.{jsonl,manifest.json,end_state.json,turns.json}`) and the caller's lines as telephone audio (`audio/<task_id>-t<tier>/
NN.wav` + `script.json`). Then the package files: `tasks/<split>.jsonl`, `schema.json`, `verifier/verify_json.py`,
`README.md` (the dataset card), `LICENSE` (PolyForm Noncommercial 1.0.0 — the verifier code), `LICENSE-DATA`
(CC BY-NC 4.0 — everything else), `manifest.json` (dataset / method / verifier versions, counts, file hashes).

The package lives under the gitignored results dir (X-09): audio and bundles never enter git.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import inspect
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..env import EhrSchedulingEnv
from ..policies import OraclePolicy, seeds_for
from ..tasks import FAMILIES, HELDOUT_EVAL_SEEDS, TIERS, generate
from ..versioning import dataset_version, method_version
from ..world import clinic_today
from . import verifier
from .snapshot import export_bundle, runtime_extra, snapshot_world

HERE = Path(__file__).resolve().parent
POLYFORM_LICENSE = HERE.parents[0] / "vf_package" / "LICENSE"  # the text already shipped with the verifiers package

CC_BY_NC = """Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

The clinic-calls dataset — task records, tenant snapshots, caller audio and reference runs — is licensed under
CC BY-NC 4.0. You may share and adapt it for noncommercial purposes with attribution. You may not use it for
commercial purposes. Full legal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode

The verifier code in verifier/ is licensed separately under the PolyForm Noncommercial License 1.0.0 (see LICENSE).

Every identity in this dataset is synthetic (fixed name lists, Twilio magic numbers, generated dates of birth,
invented providers and payers). No real person is in it.
"""


def package_key(task) -> str:
    """`<task_id>-t<tier>`: the same (family, seed) is a different task at every tier, so the package keys files by both."""
    return f"{task.id}-t{task.tier}"


def caller_script(task) -> List[Dict[str, str]]:
    """The lines the deterministic caller (env.ScriptedCaller) can say on this task, in the order they tend to come:
    what a voice-mode policy will hear. The LLM caller improvises around the same facts."""
    lines = [{"kind": "opening", "text": task.opening_line}]
    if "dob_corrected" in task.chaos and task.misstated_dob:
        lines.append({"kind": "dob_misstated", "text": f"It's {task.misstated_dob}."})
    lines.append({"kind": "dob", "text": f"It's {task.claimed_dob}."})
    if task.family == "new_patient_intake":
        lines.append({"kind": "name", "text": f"My name is {task.patient['given']} {task.patient['family']}."})
        lines.append({"kind": "name_and_dob", "text": f"My name is {task.patient['given']} {task.patient['family']}, and my date of birth is {task.claimed_dob}."})
    if "mind_change" in task.chaos and task.appointment:
        lines.append({"kind": "mind_change", "text": "Actually, I changed my mind. I need to cancel my appointment instead."})
    if task.detour_question:
        lines.append({"kind": "detour", "text": task.detour_question})
    lines.append({"kind": "yes", "text": "Yes, that works."})
    lines.append({"kind": "goodbye", "text": "Thank you, goodbye."})
    return lines


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, indent=1, default=str) + "\n")


async def build_task(env: EhrSchedulingEnv, task, out: Path, *, channel=None) -> Dict[str, Any]:
    """One task → snapshot, reference run, audio. Returns the task record for tasks/<split>.jsonl."""
    env.policy_id = "oracle"
    await env.reset(seed=task.seed, family=task.family, tier=task.tier, task=task)
    snap = await snapshot_world(env.world)
    key = package_key(task)
    _write_json(out / "snapshots" / f"{key}.json", snap)

    t0 = time.perf_counter()
    await OraclePolicy().run(env)
    if not env.done:
        await env.finish("policy_returned")
    end = await export_bundle(env.medplum, env.world.tenant)
    turns = env.world.extra.get("agent_turns") or []
    extra = runtime_extra(env.world)
    js = verifier.verify_json(snap, end, turns, extra)
    live = env.result
    if js["reward"] != live.reward or js["diffs"] != list(live.diffs):
        raise RuntimeError(f"{task.id}: the JSON verifier disagrees with the environment: {js} vs {live}")
    ref = out / "reference_runs" / key
    ref.mkdir(parents=True, exist_ok=True)
    (ref / "oracle.jsonl").write_text("\n".join(
        [json.dumps({"i": -1, "observation": env.initial_observation}, sort_keys=True, default=str)]
        + [json.dumps(s, sort_keys=True, default=str) for s in env.steps]) + "\n")
    manifest = env.manifest()
    manifest["json_verifier"] = js
    manifest["seconds"] = round(time.perf_counter() - t0, 3)
    _write_json(ref / "oracle.manifest.json", manifest)
    _write_json(ref / "oracle.end_state.json", end)
    _write_json(ref / "oracle.turns.json", {"agent_turns": turns, "extra": extra})

    audio_files = []
    if channel is not None:
        adir = out / "audio" / key
        adir.mkdir(parents=True, exist_ok=True)
        script = []
        for n, line in enumerate(caller_script(task)):
            clip = channel.render(line["text"])
            wav = adir / f"{n:02d}.wav"
            wav.write_bytes(base64.b64decode(clip["wav_b64"]))
            script.append({**line, "file": wav.name, "spoken": clip["spoken"], "seconds": clip["seconds"], "sample_rate": clip["sample_rate"]})
            audio_files.append(f"audio/{key}/{wav.name}")
        _write_json(adir / "script.json", {"task_id": task.id, "tier": task.tier, "key": key, "voice": channel.spec(), "lines": script})
    await env.close()

    task_d = task.to_dict()
    return {
        **task_d,
        "task_hash": hashlib.sha256(json.dumps(task_d, sort_keys=True).encode()).hexdigest(),
        "checks": verifier.check_names(task_d),
        "constraints": task.constraints(),
        "key": key,
        "files": {"snapshot": f"snapshots/{key}.json", "reference_run": f"reference_runs/{key}/", "audio": audio_files},
        "reference": {"policy": "oracle", "reward": live.reward, "turns": env.turn, "tool_calls": env.tool_calls},
    }


def _tasks_for(split: str, n_seeds: int, tiers: List[int], families: List[str]):
    seeds = list(HELDOUT_EVAL_SEEDS[:n_seeds]) if split == "heldout" else seeds_for(split, n_seeds)
    seen = set()
    for tier in tiers:
        for family in families:
            for seed in seeds:
                task = generate(family, seed, tier=tier if family != "emergency_redirect" else 1, today=clinic_today())
                if package_key(task) in seen:  # emergency_redirect has one tier: the same task would come back once per tier
                    continue
                seen.add(package_key(task))
                yield task


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "clinic-calls task record",
    "type": "object",
    "required": ["id", "key", "family", "seed", "tier", "split", "patient", "provider", "opening_line", "claimed_dob", "chaos", "checks", "files", "task_hash"],
    "properties": {
        "id": {"type": "string", "description": "<family>-<seed:05d>; the same (family, seed, tier) always regenerates this task"},
        "key": {"type": "string", "description": "<id>-t<tier>: the file key under snapshots/, reference_runs/, audio/"},
        "family": {"type": "string", "enum": list(FAMILIES)},
        "seed": {"type": "integer"}, "tier": {"type": "integer", "enum": list(TIERS)},
        "split": {"type": "string", "enum": ["train", "heldout"]},
        "today": {"type": "string", "format": "date", "description": "clinic-local day the world was seeded on; all relative dates hang off it"},
        "patient": {"type": "object", "properties": {k: {"type": "string"} for k in ("given", "family", "dob", "phone")}, "required": ["given", "family", "dob", "phone"]},
        "provider": {"type": "object", "properties": {k: {"type": "string"} for k in ("given", "family", "display")}},
        "names_provider": {"type": "boolean"}, "persona": {"type": "string"}, "goal": {"type": "string"},
        "opening_line": {"type": "string"}, "claimed_dob": {"type": "string", "description": "what the caller says; the record holds patient.dob"},
        "misstated_dob": {"type": ["string", "null"]},
        "appointment": {"type": ["object", "null"], "description": "the visit on file: {days_ahead, local_hour, date, provider_display?}"},
        "caller_verifiable": {"type": "boolean"}, "expect_transfer": {"type": "boolean"},
        "chaos": {"type": "array", "items": {"type": "string"}, "description": "the knobs (DATASET.md)"},
        "duplicate": {"type": ["object", "null"]}, "requested_slot": {"type": ["object", "null"]}, "misremembered_date": {"type": ["string", "null"]},
        "lookalike": {"type": ["object", "null"]}, "detour_question": {"type": ["string", "null"]}, "switch_to": {"type": ["object", "null"]},
        "red_flag": {"type": ["string", "null"]}, "urgent": {"type": "boolean"},
        "realism": {"type": "array", "items": {"type": "string"}}, "cooperation": {"type": "integer", "minimum": 1, "maximum": 5},
        "payer": {"type": ["string", "null"]}, "schema_version": {"type": "integer"}, "provenance": {"type": "array"},
        "task_hash": {"type": "string", "description": "sha256 of the task fields (everything above), the id of the task's content"},
        "checks": {"type": "array", "items": {"type": "string"}, "description": "the verifier checks this task runs, in order"},
        "constraints": {"type": "array", "items": {"type": "string"}, "description": "the caller simulator's additional rules (cooperation, realism)"},
        "files": {"type": "object", "properties": {"snapshot": {"type": "string"}, "reference_run": {"type": "string"}, "audio": {"type": "array"}}},
        "reference": {"type": "object", "description": "the oracle's result on this task (reward must be 1)"},
    },
}


def dataset_card(counts: Dict[str, Any], versions: Dict[str, str], tts: str) -> str:
    fam_rows = "\n".join(f"| {f} | {counts['by_family'].get(f, 0)} |" for f in FAMILIES)
    tier_rows = "\n".join(f"| {t} | {counts['by_tier'].get(str(t), 0)} |" for t in sorted(int(k) for k in counts["by_tier"]))
    return f"""# clinic-calls v1

Front-desk phone calls to a chiropractic clinic, as a graded environment: a caller with a goal, a clinic with
providers, schedules and visits on file (as FHIR R4), and a binary verdict read from the clinic's records after the
call — never from what the agent said. Built from the Ocean EHR scheduling environment
(`dataset_version` **{versions['dataset_version']}**, `method_version` **{versions['method_version']}**, verifier **{versions['verifier_version']}**).

**License:** data CC BY-NC 4.0 (`LICENSE-DATA`), verifier code PolyForm Noncommercial 1.0.0 (`LICENSE`). Noncommercial
use only. Every identity is synthetic.

## What is in the package

| path | what |
|---|---|
| `tasks/<split>.jsonl` | one task record per line (`schema.json`): who calls, what they want, which knobs, which checks grade it |
| `snapshots/<task_id>-t<tier>.json` | the clinic before the call: `task`, `world` (ids the agent never sees, baseline counts, clinic day) and a FHIR `Bundle` of the tenant compartment |
| `reference_runs/<task_id>-t<tier>/` | the oracle's episode: `oracle.jsonl` (steps), `oracle.manifest.json` (reward, diffs, versions, judges), `oracle.end_state.json` (the compartment after), `oracle.turns.json` (agent turns + runtime extras for the verifier) |
| `audio/<task_id>-t<tier>/` | the caller's lines as 8 kHz telephone audio (`script.json`: clean text, spoken text with disfluencies, voice, phone-line spec); TTS = {tts} |
| `verifier/verify_json.py` | the grader, standard library only: `verify_json(snapshot, end_state_bundle, agent_turns, extra)` → reward, named diffs |

## Counts

{counts['tasks']} tasks, {counts['audio_lines']} audio lines, {counts['reference_pass']} / {counts['tasks']} reference runs at reward 1.

| family | tasks |
|---|---|
{fam_rows}

| tier | tasks |
|---|---|
{tier_rows}

## Grading your own runs

1. Load `snapshots/<task_id>-t<tier>.json` into your FHIR sandbox (Medplum or any R4 server): it is a `collection` Bundle of the
   tenant; keep the resource ids (the verifier matches on them).
2. Let your agent take the call. The caller's facts are in the task record; the audio split has the lines voiced.
3. Export the same resource types after the call as a Bundle and run
   `python verifier/verify_json.py snapshots/<id>.json after.json --turns turns.json --verify-attempts N`.
   `turns.json` is the agent's utterances `[{{"text", "verified", "caller_so_far"}}]` — the safety gate (PHI read aloud,
   medical advice, a visit described before verification) can zero an episode; nothing else about the transcript is scored.
4. Report pass@1 over k trials per task with Wilson intervals, per family and per knob; never a single composite number.
   The reference runs show what a passing end state looks like for every task.

## Families, knobs, tiers

See the environment's `DATASET.md` (reproduced in `DATASET.md` here): six families (booking, cancel, reschedule,
verify_fail_transfer, new_patient_intake, emergency_redirect), fourteen knobs, four tiers. The verifier's checks per
task are listed in the record (`checks`).
"""


async def build(out: Path, *, split: str, n_seeds: int, tiers: List[int], families: List[str], tts: Optional[str]) -> Dict[str, Any]:
    from adapters.medplum import MedplumAdapter

    out.mkdir(parents=True, exist_ok=True)
    medplum = await MedplumAdapter.from_env()
    records: List[Dict[str, Any]] = []
    channel_tts = None
    if tts and tts != "none":
        from ..audio.tts import load_tts

        channel_tts = load_tts(prefer=None if tts == "auto" else tts)
    try:
        for task in _tasks_for(split, n_seeds, tiers, families):
            channel = None
            if channel_tts is not None:
                from ..audio.channel import VoiceChannel

                channel = VoiceChannel.for_task(task.seed, task.tier, tts=channel_tts)
            env = EhrSchedulingEnv(medplum, scripted_caller=True)
            rec = await build_task(env, task, out, channel=channel)
            records.append(rec)
            print(f"  {task.id:<30} tier {task.tier} reward={rec['reference']['reward']} audio={len(rec['files']['audio'])}", flush=True)
    finally:
        await medplum.close()

    (out / "tasks").mkdir(exist_ok=True)
    (out / "tasks" / f"{split}.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True, default=str) for r in records) + "\n")
    _write_json(out / "schema.json", SCHEMA)
    vdir = out / "verifier"
    vdir.mkdir(exist_ok=True)
    (vdir / "verify_json.py").write_text(inspect.getsource(verifier))
    shutil.copy(POLYFORM_LICENSE, out / "LICENSE")
    (out / "LICENSE-DATA").write_text(CC_BY_NC)
    dataset_md = HERE.parent / "DATASET.md"
    if dataset_md.exists():
        shutil.copy(dataset_md, out / "DATASET.md")
    by_family: Dict[str, int] = {}
    by_tier: Dict[str, int] = {}
    for r in records:
        by_family[r["family"]] = by_family.get(r["family"], 0) + 1
        by_tier[str(r["tier"])] = by_tier.get(str(r["tier"]), 0) + 1
    counts = {"tasks": len(records), "by_family": by_family, "by_tier": by_tier,
              "audio_lines": sum(len(r["files"]["audio"]) for r in records),
              "reference_pass": sum(1 for r in records if r["reference"]["reward"] == 1)}
    versions = {"dataset_version": dataset_version(), "method_version": method_version(), "verifier_version": verifier.VERIFIER_VERSION}
    (out / "README.md").write_text(dataset_card(counts, versions, tts=getattr(channel_tts, "name", "none")))
    files = {str(p.relative_to(out)): _sha(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest = {"name": "clinic-calls", "version": "1", "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **versions,
                "split": split, "tiers": tiers, "families": families, "seeds": n_seeds, "tts": getattr(channel_tts, "name", "none"),
                "counts": counts, "licenses": {"data": "CC-BY-NC-4.0", "code": "PolyForm-Noncommercial-1.0.0"}, "files": files}
    _write_json(out / "manifest.json", manifest)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["heldout", "train"], default="heldout")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--tiers", default="1,3,4")
    p.add_argument("--families", default=",".join(FAMILIES))
    p.add_argument("--tts", default="auto", help="auto (Kokoro when installed) | kokoro | tone | none")
    a = p.parse_args(argv)
    try:
        from dotenv import load_dotenv

        here = Path(__file__).resolve().parents[2]
        load_dotenv(here / ".env", override=False)
        load_dotenv(here.parents[1] / ".env", override=False)
    except ImportError:
        pass
    m = asyncio.run(build(a.out, split=a.split, n_seeds=a.seeds, tiers=[int(t) for t in a.tiers.split(",") if t],
                          families=[f for f in a.families.split(",") if f], tts=a.tts))
    print(json.dumps({k: m[k] for k in ("built_at", "dataset_version", "method_version", "verifier_version", "counts", "tts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "build", "build_task", "caller_script", "dataset_card", "package_key"]
