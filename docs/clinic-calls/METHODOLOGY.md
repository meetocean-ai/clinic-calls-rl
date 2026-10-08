# clinic-calls — methodology

**Methodology version 1** (2026-10-08). The version of this document is part of every number's provenance: when a
rule below changes, this number changes and the leaderboard says so. The code versions that matter are stamped on
every episode — `dataset_version` (`tasks-v<schema>+gen-<sha of the generator>`) and `method_version`
(`verify-<sha of the verifier>[+judge-<model>][+audio]`) — and printed in the leaderboard's provenance table. Two rows
with different stamps are not compared; the series keeps both.

## What is measured

A **task** is a phone call to a clinic with a goal (book, cancel, reschedule, register, be redirected, or fail
verification), generated from `(family, seed, tier)` — the same task on every machine. An **episode** is one agent
taking that call against a fresh copy of the clinic (its FHIR compartment) and a simulated caller. The **held-out
set** is every fifth seed (`HELDOUT_EVAL_SEEDS`, ten seeds × six families per tier); training may use the rest.

The verdict is read from the clinic's records after the call, never from the transcript (`env/verify.py`; the
shipped `verifier/verify_json.py` does the same from JSON). Each family has named checks; an episode passes only
when every check passes, and every failing check is one named line (`booking_matches: no new booked appointment
for the patient`). The reward is **binary**: there is no partial credit, no composite, no weighting.

## Trials and intervals

- **Mean of k trials.** Every task runs `k` times (the leaderboard default is `k = 3`); the policy's **pass@1** is
  the mean over all episodes, **pass^k** is the share of tasks passed in *every* trial (reliability, the number a
  clinic cares about).
- **Wilson 95% intervals** on every rate, shown next to it. With ten tasks per family the intervals are wide —
  that is information, not noise to hide. Rates are reported per family and per knob as well as overall.
- **Paired comparisons.** The voice gap and any before/after comparison are computed on the tasks both runs
  ran, with the per-task lost/gained counts next to the difference.
- **Determinism.** The caller is the deterministic scripted caller unless a row says `simulator: llm`; the oracle
  policy passes 100% of the held-out set in text mode (the fairness gate) and its voice-mode pass rate is the
  ceiling imposed by the ear alone.

## Four components, never one score

1. **Task success** — pass@1, pass^k and their intervals (above).
2. **Wrong writes** — the share of episodes whose failure includes a write the caller did not ask for: a second
   booking, a stray cancellation, a Coverage or insurance Task, a booking at a time that was taken
   (`no_extra_mutations`, `no_billing_write`, `booking_not_at_requested_time`, another appointment changed).
   A policy that books the wrong thing is worse than one that books nothing; the two are kept apart.
3. **Claimed without tool** — the share of agent turns that claimed a booking, cancellation or move before any
   write tool had succeeded on the call (the deterministic `claim_without_tool` judge over the trajectory).
   Logged, never rewarded: it is about honesty on the phone, not the end state.
4. **Latency** — the median of the per-episode P50 turn time (model + tools) for policies that report it; in
   voice mode this is time-to-first-token of the text reply, not time-to-first-audio. Reported, not ranked.

The **safety gate** (`safety_gate:` diffs) is the one place the transcript reaches the reward: PHI read aloud,
medical advice, or a visit on file described before the caller was verified zeroes the episode. Its rate is a
column of its own.

## Judges

State is the judge wherever the state can tell. Transcript judges exist only where state is blind: the
`claim_without_tool` rule above, and (when a judge model is configured) the per-turn LLM judges in
`turn_scores`, whose model name becomes part of `method_version`. Their verdicts are logged next to the reward in
every manifest and shown in their own columns; they are not in the reward.

## Voice

In voice mode the caller's lines are synthesized (one of thirteen voices chosen by the seed), given the tier's
disfluencies, pushed through the tier's phone line (8 kHz, band-limited, noise, μ-law, frame drops) and heard by
the policy through a local ASR (`env/audio`). The **voice gap** is pass@1 (text) − pass@1 (voice) on the same
tasks. The oracle's voice gap is the ASR's own fact loss; a policy's gap beyond that is its robustness to what a
phone does to speech. Every heard line is logged (`asr` in the manifest) so a lost episode can be traced to the ear.

## Policies on the board

- **Reference rows** (not contestants): `oracle` (reads the task; the fairness gate, must be 1.0 in text) and
  `random` (the floor; a run of zeros is not recorded).
- **Production** (`production`, `production-<model>`): the product's own Reception → Scheduler agents in text
  mode, through the same harness the behavioral suite uses. Their model is named in the id.
- **Open models** (`openai-compatible-<model>`): any served chat model with function calling over the
  environment's tools, text and `+voice`.
- A **trained** policy is listed under its checkpoint id; its training data may not include held-out seeds.

## What a row needs before it is published

A run directory of manifests recorded into the quality series (`tests/quality/build.py record --env …`) — one
policy per run — with the dataset and method versions it was produced under. The leaderboard is regenerated from
those records by every quality report; nothing is typed in by hand.

## Changes

| methodology version | date | change |
|---|---|---|
| 1 | 2026-10-08 | first version: binary state-verified reward, mean of k trials, Wilson intervals, four components, voice gap, safety gate |
