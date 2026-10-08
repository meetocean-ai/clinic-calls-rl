# Glossary — the terms a cold reader meets in this repo

**Task** — one phone call to solve, generated deterministically from `(family, seed, tier)`: who calls (synthetic
patient), what they want (goal + opening line), which difficulty knobs apply, what the clinic looks like. Id
`<family>-<seed:05d>`; the same seed is a different task at every tier, so files are keyed `<id>-t<tier>`.

**Family** — the kind of call: `booking` (a follow-up visit), `cancel`, `reschedule`, `verify_fail_transfer` (the
caller cannot give the DOB on file → no change + a warm transfer after real attempts), `new_patient_intake` (no record:
register + book), `emergency_redirect` (red-flag symptoms → never booked, told to call 911, urgent transfer).

**Knob (chaos)** — a composable difficulty: a change to the world or the caller plus the verifier check that goes with
it. Fourteen: `slot_taken`, `slot_taken_twice`, `existing_visit`, `dob_corrected`, `mind_change`, `shared_phone`,
`no_availability`, `exact_time_unavailable`, `two_requests`, `wrong_day_memory`, `provider_name_collision`,
`insurance_detour`, `provider_switch_after_search`, `urgent_same_week`. Exclusion/subsumption rules keep combinations
satisfiable.

**Tier** — 1: polite caller, no knobs. 2: a persona (anxious, rushed, hostile, elderly, demanding, …), no knobs.
3: one or two knobs. 4: two or three knobs that survive normalization together (the "bad phone line" in voice mode).

**Realism traits / cooperation** — caller behaviour the simulator is told to show (fillers, self-corrections,
interruptions, off-topic asides, limited English, hesitant DOB, facts hidden until asked) and a 1–5 cooperation level;
hashed into the task, never changing the right end state.

**Held-out split** — every fifth seed (5, 10, …, 50): ten tasks per family per tier. Training may use the rest.

**Oracle / random** — reference policies, not contestants. The oracle reads the task and uses the env's tools (must
score 1.0 everywhere — the *fairness gate*); random picks tools and utterances at random (the *floor*, ~0).

**Bracketing** — oracle × random × the untrained open model × production, per family and tier, before any training.

**Reward** — binary, from the FHIR end state of the episode's own tenant after the call: the family's named checks
(`booking_matches`, `cancel_targeted_right_id`, `reschedule_old_gone_new_present`, `waitlisted_not_booked`,
`booking_not_at_requested_time`, `two_requests_done`, `patient_created_once`, `emergency_escalated`,
`booked_soon_enough`, `no_billing_write`, `no_extra_mutations`, `transfer_matches`). Every failing check is one named
**diff** line. Nothing in the transcript earns reward.

**Safety gate** — the transcript's one way into the reward, and only to take it away: PHI read aloud (SSN, insurance
id, card), medical advice, or the visit on file described before the caller was verified → reward 0 (`safety_gate:`
diffs). The patterns are the product's own compliance observer, vendored into the shipped verifier.

**Judges (logged, not rewarded)** — deterministic or model verdicts stored next to the reward for analysis:
`claim_without_tool` (the agent claimed a booking/cancel/move before any write tool succeeded), audio judges (tone,
empathy, mispronunciation, artifacts), latency.

**Wrong write** — a write the caller did not ask for: an extra booking, a Patient created or removed when none should
be, an unverified caller changing a visit, a Coverage/insurance write, a double booking. A *missing* write is a failed
task, not a wrong write.

**pass@1 / pass^k / pass@k** — mean reward over all trials; share of tasks passed in *every* trial (reliability);
share passed in *any* trial. Always with a **Wilson 95% interval**.

**dataset_version / method_version** — `tasks-v<schema>+gen-<hash of generator + world>` and
`verify-<hash of verifier>[+judge-<model>][+audio]`, hashed when the package is imported and stamped on every episode.
Rows under different stamps are not compared.

**Episode record / manifest** — `<trial>.jsonl` (every step: action, observation) and `<trial>.manifest.json`
(reward, diffs, task hash, versions, judges, latency, voice spec, ASR log, turn-taking, errors), under
`<run>/<policy>/<task>-t<tier>/`.

**Quality series** — run records committed to git (slim: one row per test, messages capped) and dated reports that
diff against the previous one; the **leaderboard** is a view over the `rl-env-*` records.

**Voice mode** — the caller's lines also arrive as 8 kHz telephone audio: TTS (Kokoro voice chosen by the seed) →
disfluencies (never inside dates, names, numbers) → degradation (resample, 300–3400 Hz, noise at a tier-dependent
SNR, G.711 μ-law, 20 ms frame drops, gain wobble). `method_version` gets `+audio`.

**ASR cascade (`+voice` policy)** — a text policy that hears the caller through a local Whisper; the transcript becomes
its `patient_text`, the clean line stays in the record; the manifest logs said / spoken / heard per line.

**Voice gap** — pass@1 in text mode minus pass@1 in voice mode, same policy, same tasks. The oracle's gap is the ear's
own loss.

**Turn-taking metrics** — from the two audio streams' geometry: response latency, barge-ins classified as
takeover-in-pause / interruption / backchannel (Full-Duplex-Bench vocabulary), overlap, talk ratio; caller barge-ins
from the scripted talk-over.

**S2S (speech-to-speech) model** — one network from audio in to audio out, no STT/LLM/TTS cascade (Qwen3-Omni,
NemotronLabs VoiceChat, PersonaPlex, gpt-realtime, Gemini Live). **Delegation** — a duplex model without tools hands
the transcript to a text LLM that runs the tools and prefills the answer.

**Snapshot** — the tenant's FHIR compartment before the call (a `collection` Bundle) plus the world facts the grader
needs (ids the agent never sees, baseline counts, the clinic day). **End state** — the same export after the call.

**`verify_json`** — the standard-library grader shipped in the package: `verify_json(snapshot, end_state_bundle,
agent_turns, extra) → {reward, diffs, checks_run}`, same checks and diff strings as the live verifier.

**Shadow run / graduation gate** — a candidate checkpoint run through the held-out set as `candidate-<name>` next to
production; eligible to replace production only if pass@1 ≥ production with non-overlapping intervals, zero wrong
writes, zero claimed-without-tool.

**Tenant / compartment** — one FHIR `Organization` per clinic (and per episode here); every read and write is scoped
with `_compartment=Organization/<id>`.

**BAA** — Business Associate Agreement; PHI-bearing capabilities may only route to providers under one (or self-hosted).

**GRPO / LoRA / K** — the RL algorithm (group-relative policy optimisation: K rollouts per prompt, advantages
normalised within the group), the low-rank adapter trained instead of full weights (r = rank), and the group size.
