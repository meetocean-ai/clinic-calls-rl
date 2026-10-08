# Experience sheet — the production voice-agent side, in generic terms

The environment in this repo was built to test a real product: a multi-tenant voice AI that answers and places phone
calls for clinics (scheduling, reminders, intake, after-hours messages) against the clinic's own practice-management
system. This page lists what running that product taught, without customer or vendor names. Numbers are from our own
measurements on our own calls.

## The system

- **Stack**: LiveKit Agents (Python) for real-time voice, Twilio SIP for telephony, Deepgram streaming STT, LLM via one
  proxy (LiteLLM) with per-capability model profiles, TTS (Deepgram / Cartesia), Silero VAD; a Django REST backend
  (multi-tenant, JWT, tenant scoping on every query via decorators and a base viewset); Hatchet for workflows
  (outbound campaigns, nightly syncs); Postgres; a browser-automation service for practice-management systems that have
  no API; a Next.js dashboard. Prod on AWS ECS behind an ALB, admin reachable only over an SSM tunnel.
- **Design rule**: onboarding a clinic needs zero code and zero admin work — every tenant-specific thing (appointment
  types, insurance codes, prompts, flags) is dashboard configuration; a per-tenant config field beats a constant, a
  data-driven flow beats a per-tenant branch. Schema changes to tenant config ship with seed data, a back-compat reader,
  and a documented manual step in the same PR.

## Latency, measured

- End-of-utterance detection is the floor: ~1.5 s before the model starts; acknowledgement lines ("one moment") cover
  the 7–9 s the slow tools take; a 26 s silent loop found on a call traced to a tool-step cap dropping tool results.
- A cold token for the practice-management system plus multi-language STT explained a "laggy" owner call; keep the
  token warm and the language set narrow when the clinic is monolingual.
- Barge-in and VAD tuning, shared-phone DOB prompts and hold lines were the fixes that moved real calls, not model swaps.

## STT, measured

- Our benchmark on our call audio: Deepgram nova-3 7.9% WER at 0.19 s vs a Gemini streaming STT at 11.8% / 1.34 s —
  kept nova-3. A second vendor evaluation showed natural-speech WER under ~5.5% is vendor-level; the remaining error
  budget is start-of-call silence and word drops, not the recogniser.
- Chinese-language calls: the agent must switch language on the first Mandarin utterance; a caller who never got
  Chinese ranked as a top call failure, and English leaking into Chinese triage was its own incident.

## Evaluation and quality, how it is done

- Four layers: unit → behavioural (simulated caller against the real agent in text mode, judges + state assertions)
  → nightly end-to-end → voice happy-path regression (real calls, latency measured as "response heard"). Reports land in
  a dated, versioned quality series in git, every report diffing the previous one; a feature-coverage registry links
  every test (~1,400 in one repo, ~190 features in the other) to a feature id and fails CI on unknown ids.
- Eval stack: DeepEval + Ragas + Promptfoo over recorded runs, HTML comparison reports, pass^k for reliability; LLM
  judges are logged, the state assertions decide.
- A trust review of 64 behavioural transcripts found the suite lying in four ways (a mock ignoring the query, no tool
  outputs shown to the judge, a booking task blind to date and provider, regressions hidden by flaky passes) — the
  origin of "state is the judge" in this repo.
- Call audits (62 production calls at one point): the top defect was a transfer that looped back to the agent; others
  were hallucinated confirmations ("intake forms sent" with nothing sent), a greeting that read a name aloud before
  verification, and provider/time preferences not kept across turns. Each became a behavioural test before a fix.
- Release = PR merged by the same agent that ran the suites, release notes with an evidence table, reports attached as
  release assets; a 429 from a shared API key invalidates the run.

## Reliability incidents and what they taught

- A message-broker channel death stalled all workflows: health checks must exercise the consumer, not the process.
- `sh -c` as PID 1 swallowed SIGTERM and containers restarted dirty: exec the real process.
- A browser-automation service leaked headless Chromium processes until memory ran out: per-task lifetimes + a gauge.
- A worker that forks job processes from a preloaded server never picks up code edits: recreate, do not restart; compare
  container start time with the file mtime before trusting a test call.
- Cache truncation in a patient cache took a tenant's lookups down: cap sizes explicitly and alert on eviction.
- A production phrasebook got wiped by a sync: every destructive sync is dry-run first and diff-gated.

## Security and compliance habits

- Tenant scoping tested, not assumed (no global `.all()` on tenant models); production data queries run inside the live
  task over the ORM so no DB credentials are handled, results stay local and are cleared; anything with patient data
  never leaves the VPC.
- API hardening pass: auth on internal services, feature flags default off, credentials out of plaintext; a leaked
  client secret in git history is on the revoke list rather than quietly rewritten.
- PHI never in test code: synthetic names, Twilio test numbers, generated DOBs, enforced by a test.

## What I would say I know how to do

Build and run a voice agent end to end on telephony; measure it honestly (latency decomposition, STT WER, state-verified
task success with intervals); turn production call failures into regression tests; keep a multi-tenant healthcare
backend tenant-scoped and BAA-clean; ship with evidence; and, from this repo, turn the product into an RL environment
and a benchmark that outsiders can grade.
