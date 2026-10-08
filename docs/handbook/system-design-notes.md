# System design notes — how the pieces are built, and how I would build them again

Four systems that exist in this work, each with the shape, the decisions, the failure modes met, and what I would
change. Numbers are the recorded ones.

---

## 1. A voice agent for a clinic front desk (the product this environment tests)

```
 PSTN ──Twilio SIP──▶ LiveKit room ──▶ Agent worker (Python)
                                        │  VAD (Silero) → STT (Deepgram nova-3, streaming)
                                        │  LLM via LiteLLM proxy (profile: Gemini / GPT-4.1 / Qwen on vLLM)
                                        │  tools → FHIR (Medplum) : verify DOB, search slots, book, cancel, waitlist, transfer
                                        │  TTS (Deepgram Aura / Cartesia) → back to the room
                                        └─ ComplianceObserver: PHI / medical-advice patterns on every Nth assistant turn
 Hatchet workflows: outbound campaigns, reminders, nightly syncs      Postgres marts: analytics, agent memory (RLS by org)
```

**Decisions.** One LLM proxy (LiteLLM) so model, cost and BAA routing live in one config; capability-level model
profiles (`voice`, `scribe`, `planner`, …) with per-tenant overrides stored on the tenant's FHIR Organization; every
FHIR write stamped with the tenant compartment; Reception → Scheduler as a handoff between two small agents with few
tools each (latency: fewer tools, shorter prompts). The agent verifies the caller by DOB before reading or changing
anything, and the verification is a tool whose output the harness can see — not a line in a prompt.

**Failure modes met.** Hallucinated bookings ("you're all set" with no write) → a deterministic claim-without-tool
judge over the trajectory. Shared phone numbers → first-match patient lookup books on the wrong record → the DOB picks
the record, verified by the environment's `shared_phone` knob. Namesake providers → a resolver that took the first
prefix match (fixed; exact name wins, ambiguity refuses). Latency: end-of-utterance detection is a ~1.5 s floor before
the model even starts; tools that take 7–9 s need an on-hold line. Code edits not live in the running worker (forked
job processes) cost a real test call.

**If I built it again.** Same cascade for a regulated domain (text tools stay auditable), with the S2S model as a
per-tenant option behind the same realtime interface (that option exists now, default off), a self-hosted model only
after it beats production on the held-out benchmark with non-overlapping intervals and zero wrong writes (the
graduation gate), and the environment below as the release gate from day one.

---

## 2. An RL environment for tool-using agents (this repo)

```
 reset(seed, family, tier, chaos) ─▶ generate task (deterministic from the seed) ─▶ seed a FRESH tenant on Medplum
                                      (providers, schedules, the patient, the visit on file, bystanders, the namesake…)
 step(action)  say → the caller replies (scripted or LLM simulator) [+ audio: TTS → disfluencies → phone line]
               tool → the product's own FHIR tool, tenant-scoped; results back as data
               end_call / transfer → finish()
 finish() ─▶ verify(world): read the compartment, run the family's named checks, reward ∈ {0, 1}, diffs = named misses
             safety gate: PHI read aloud / medical advice / visit described before verification → reward 0
             judges (claim-without-tool, audio tone…) → logged in the manifest, never in the reward
 manifest: task hash, dataset version, method version, steps hash, diffs, latency, voice spec, ASR log, turn-taking
```

**Decisions that mattered.**
- **State is the judge.** Reward from the FHIR end state only; the transcript can take reward away (safety gate) but
  never give it. A policy that says the right thing and does nothing scores 0; one that books the wrong provider scores 0
  with a named diff.
- **One tenant per episode.** No shared state between episodes, no cleanup races, parallel rollouts trivially safe;
  measured reset p95 0.2 s, 14k oracle episodes/hour at 8 concurrent — no pool needed.
- **Bracketing before any training.** Oracle (reads the task) must be 1.0 on every knob — the fairness gate; random
  must be ~0 — the floor. A knob the oracle cannot pass is unfair; a knob random passes is trivial. Oracle 120/120 at
  tiers 3–4, random 0/120, before a single LLM episode.
- **Difficulty as composable knobs**, each a world change + a verifier check + a hand-built failure case, with
  exclusion rules so combinations stay satisfiable (tier 4 = 2–3 knobs). Held-out = every fifth seed.
- **Version everything.** `dataset_version` = schema + hash of the generator; `method_version` = hash of the verifier
  (+ judge model, + audio). Hashed at import, so editing the verifier mid-sweep cannot relabel episodes (learned the
  hard way).
- **The environment's own writes are declared.** A bystander booked mid-episode is `env_writes`, so a grader working
  from a before-snapshot counts mutations the same way the live verifier does.
- **Keep tools textual even in voice mode** (SpeechGym's point): perceptual errors stay separable from reasoning errors.

**The voice layer.** Caller lines → Kokoro (seeded voice) → disfluencies (never inside dates, names, numbers) →
8 kHz phone line (seeded SNR / frame loss / gain wobble per tier) → the policy hears audio; the text stays in the
observation for text policies. The ASR cascade wraps any text policy (`+voice` id). The oracle through the cascade is
the ear's own loss: 0.000 at tier 3, +0.050 at tier 4 (DOB misheard on the bad line). Agent audio comes back as an
action; the env transcribes it for the judges, keeps Opus clips, and measures turn-taking (latency, takeover in a
pause, interruption, backchannel) from the two streams' geometry — no judge.

**Training loop (recipe, text run not yet paid).** prime-rl or TRL ← verifiers/OpenEnv env server ← Medplum; GRPO
K=8 (text) / K=4 (speech), LoRA r=16, binary reward now, per-turn tool credit only after its correlation with success is
measured on recorded episodes (dense rewards cost 14 pp in the IRC paper); evaluation every N steps with the same
held-out harness; stop when two evaluations show no held-out gain.

**Failure modes met.** Expired auth tokens kill hour-long sweeps (refresh on 401). A verifier exception produced a
silent `None` reward (now an episode error). Tests that hard-code `slot_index: 0` are order-dependent — a stub model
must book the provider the caller named. Whisper is not bit-deterministic on degraded audio (55 vs 57 of 60 across two
identical runs) — report intervals, not single counts.

---

## 3. A benchmark and leaderboard people can trust (`clinic-calls`)

- **Package = tasks + before-snapshot + reference run + end state + caller audio + a standard-library grader.** A lab
  loads the snapshot into any FHIR server, runs its agent, exports the compartment, and `verify_json` grades it with the
  same checks and the same diff strings as the live verifier — proven by running both on the same episodes (oracle on
  every family, random, a transfer-happy policy).
- **Methodology is versioned** like the code: mean of k trials, Wilson intervals on everything, four components kept
  apart (task success, wrong writes, claimed-without-tool, latency), no composite score, judges only where state is
  blind, text vs voice as two modes of one policy with the gap on paired tasks.
- **The leaderboard is a view over recorded runs**, regenerated by the same build that writes the quality report;
  nothing typed by hand; every row prints its run ids and version stamps; rows without stamps are marked.
- **Reviewers found what an author would not**: a wrong-write classifier that counted a *missing* registration as a
  wrong write; a production row stacking trials from overlapping suites; a test run that overwrote the committed board.
  Independent review before a number goes public is part of the design.

---

## 4. HIPAA / multi-tenant guardrails in an AI agent

- Tenant compartment on every FHIR read and write; a test asserts no search runs without `_compartment`; the
  environment's own export refuses a full page rather than truncate silently.
- PHI-bearing capabilities (voice, scribe, planner) may only route to BAA-covered or self-hosted providers; enforced at
  profile-config time by a CI test, not only at request time.
- A deterministic compliance layer on the transcript (SSN / insurance-id / card patterns, medical-advice phrasing, a
  visit described before verification) — the same patterns the product uses at runtime are vendored into the benchmark's
  verifier, with a test that they stay byte-identical.
- Synthetic identities only, enforced by a test over the whole generated space (names from fixed lists, Twilio test
  numbers, generated DOBs, invented providers and payers); the public export was scanned for keys, real phones and
  non-example emails before publishing.
- Secrets never in reprs or logs (a dataclass `repr=False` on the auth headers after pytest printed a token tail),
  never in git (a leaked client secret in history is on the owner's revoke list), tenant-supplied keys resolved at call
  time and cached briefly.
- Everything that costs money or leaves the machine is a ledger line with a cap before it starts.
