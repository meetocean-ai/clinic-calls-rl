# clinic-calls v1

Front-desk phone calls to a chiropractic clinic, as a graded environment: a caller with a goal, a clinic with
providers, schedules and visits on file (as FHIR R4), and a binary verdict read from the clinic's records after the
call — never from what the agent said. Built from the Ocean EHR scheduling environment
(`dataset_version` **tasks-v3+gen-c605a9eb4c**, `method_version` **verify-5b517a1f4e**, verifier **clinic-calls-verify-1**).

**License:** data CC BY-NC 4.0 (`LICENSE-DATA`), verifier code PolyForm Noncommercial 1.0.0 (`LICENSE`). Noncommercial
use only. Every identity is synthetic.

## What is in the package

| path | what |
|---|---|
| `tasks/<split>.jsonl` | one task record per line (`schema.json`): who calls, what they want, which knobs, which checks grade it |
| `snapshots/<task_id>-t<tier>.json` | the clinic before the call: `task`, `world` (ids the agent never sees, baseline counts, clinic day) and a FHIR `Bundle` of the tenant compartment |
| `reference_runs/<task_id>-t<tier>/` | the oracle's episode: `oracle.jsonl` (steps), `oracle.manifest.json` (reward, diffs, versions, judges), `oracle.end_state.json` (the compartment after), `oracle.turns.json` (agent turns + runtime extras for the verifier) |
| `audio/<task_id>-t<tier>/` | the caller's lines as 8 kHz telephone audio (`script.json`: clean text, spoken text with disfluencies, voice, phone-line spec); TTS = kokoro |
| `verifier/verify_json.py` | the grader, standard library only: `verify_json(snapshot, end_state_bundle, agent_turns, extra)` → reward, named diffs |

## Counts

160 tasks, 742 audio lines, 160 / 160 reference runs at reward 1.

| family | tasks |
|---|---|
| booking | 30 |
| cancel | 30 |
| reschedule | 30 |
| verify_fail_transfer | 30 |
| new_patient_intake | 30 |
| emergency_redirect | 10 |

| tier | tasks |
|---|---|
| 1 | 60 |
| 3 | 50 |
| 4 | 50 |

## Grading your own runs

1. Load `snapshots/<task_id>-t<tier>.json` into your FHIR sandbox (Medplum or any R4 server): it is a `collection` Bundle of the
   tenant; keep the resource ids (the verifier matches on them).
2. Let your agent take the call. The caller's facts are in the task record; the audio split has the lines voiced.
3. Export the same resource types after the call as a Bundle and run
   `python verifier/verify_json.py snapshots/<id>.json after.json --turns turns.json --verify-attempts N`.
   `turns.json` is the agent's utterances `[{"text", "verified", "caller_so_far"}]` — the safety gate (PHI read aloud,
   medical advice, a visit described before verification) can zero an episode; nothing else about the transcript is scored.
4. Report pass@1 over k trials per task with Wilson intervals, per family and per knob; never a single composite number.
   The reference runs show what a passing end state looks like for every task.

## Families, knobs, tiers

See the environment's `DATASET.md` (reproduced in `DATASET.md` here): six families (booking, cancel, reschedule,
verify_fail_transfer, new_patient_intake, emergency_redirect), fourteen knobs, four tiers. The verifier's checks per
task are listed in the record (`checks`).
