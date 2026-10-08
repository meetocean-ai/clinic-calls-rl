# The production voice-agent platform — stack skeleton

A map of the other repo (the live product this environment tests): what each service is, what it is built with, how
it is tested, run and deployed. Names of the clinics and of the practice-management vendors it integrates with are left
out; everything else is the real layout, scanned 2026-10-08.

## Monorepo layout

```
agent/              LiveKit voice agents (Python)            — the thing on the phone
backend/            Django REST API, multi-tenant             — tenants, clinics, patients, appointments, waitlist, phrasebook, knowledge base
browseruse_agent/   FastAPI service, browser automation       — talks to practice-management systems that have no API
dashboard/          Next.js app                               — clinic-facing configuration and call review
common/             shared Python: tracing, logging, Hatchet config, Sentry, Discord notifications, middleware
db/                 database init
aws_lambda/         service-prober, session_time_limiter
grafana_dashboards/ LiveKit agent dashboards (dev / prod)
stt_bench/          the STT benchmark (corpus, audio, results)
specs/              implementation plans (time-bound) + specs/reference (long-lived design specs: agent architecture,
                    i18n, appointment types, release process, settings backup)
scripts/            prod tunnel (SSM), ECS exec query runner, Twilio/LiveKit setup, env check, unit-test runner
docs/, prds/        latency benchmarks, QA readiness, product requirement docs
```

## Services (`docker-compose-local.yml`)

| service | role |
|---|---|
| `backend` | Django API (port 8000), JWT auth, tenant scoping by URL path `/api/v1/tenants/<domain>/…` |
| `frontend` | Next.js dashboard |
| `postgres` | the database (+ pgvector) |
| `rabbitmq` | broker for Hatchet |
| `hatchet-engine`, `hatchet-dashboard`, `hatchet-worker`, `agent-hatchet-worker` | workflow orchestration (dashboard on 8888) |
| `scheduling_agent` | the LiveKit agent worker (forks a job process per call from a preloaded server) |
| `browseruse_agent` | browser automation (port 8001), Playwright + `browser-use` |
| `jaeger` | OpenTelemetry traces locally |
| `migration`, `setup-config` | one-shot migrations and config seeding |

`make world` builds everything, migrates and seeds test data; `make up/down/logs/ps`, `make migrate`, `make backend`
(shell), `make test-behavioral`, `make curation-apply/check` (tenant config export/apply/check), `make download-agent-models`.

## Stack per service

**Voice agent (`agent/`)** — `livekit`, `livekit-agents[openai, deepgram, elevenlabs, anthropic, cartesia, silero,
turn_detector, groq]`, `livekit-plugins-noise-cancellation` (Krisp), `livekit-plugins-google`, `hatchet-sdk`, `aiohttp`,
`rapidfuzz`/`fuzzywuzzy` (name and type matching), `sentry-sdk`, OpenTelemetry (API/SDK/OTLP exporter/logging),
`boto3`, `reportlab`, `psutil`; tests with `pytest`, `pytest-asyncio`, `pytest-mock`, `pytest-dependency`, `allure-pytest`.
Telephony: Twilio SIP trunks → LiveKit rooms; outbound calls dispatched from Hatchet workflows. Realtime path: Deepgram
nova-3 STT (multi-language), LLM (OpenAI Realtime / chat models, Anthropic, Groq, Google as plugins), TTS (Deepgram
Aura / ElevenLabs / Cartesia), Silero VAD + LiveKit turn detector.

**Backend (`backend/`)** — Django + Django REST Framework, SimpleJWT, nested routers, django-filter, CORS, cleanup,
environ, ipware, simple-history (audit trail), treebeard (hierarchies), OTP (2FA), `transitions` (state machines),
psycopg2 + SQLAlchemy + pgvector, gunicorn/uvicorn, whitenoise, S3 via django-storages, SES, Stripe, Twilio, LiveKit
server SDK, Hatchet SDK, OpenAI, PDF/DOCX parsing (PyPDF2, pdfplumber, python-docx), BeautifulSoup/lxml (knowledge-base
scraping), pymssql (a vendor database), Sentry, OpenTelemetry; tests with pytest-django, factory-boy, pytest-cov.
Apps: `api`, `core` (tenants, users, roles ADMIN/GENERAL, `TenantUser`), `clinic` (locations, providers, appointment
types), `assistant`, `waitlist`, `phrasebook`, `knowledge_base`, `state_management`, `integrations/{hatchet, twilio,
stripe, openai, <practice-management vendor>}`, `dashboard`, `data`, `testing_utils`.

**Browser automation (`browseruse_agent/`)** — FastAPI, `browser-use` + Playwright (Chromium), `langchain_openai`,
`pyotp` (vendor 2FA), httpx, Sentry, OpenTelemetry.

**Dashboard (`dashboard/`)** — Next.js + React, Redux Toolkit + redux-persist, react-hook-form + zod, Radix UI
primitives with Tailwind (shadcn-style), lucide icons, recharts, LiveKit React components + `livekit-client`, Sentry;
Vitest + Testing Library.

**Shared (`common/`)** — tracing (OTel), structured event logging, API request logging middleware, Hatchet client
configuration, Sentry integration, Discord notifications, constants.

## The agent, inside (`agent/agents/`)

- `base/`: `base_agent.py`, `base_task.py` (one task per conversational phase), `tasks/` — `verify_patient`, `name`,
  `collect_contact`, `new_patient`, `booking`, `reschedule`, `cancel`, `waitlist`, `insurance`, `email`,
  `clinic_question` (+ `knowledge_tools`), `collection_prompts`, `wide_eot` (end-of-turn widening for long answers);
  cross-cutting `guardrails.py`, `hallucination_gate.py` (no booking claim without a write), `hold_line.py`,
  `transfer_loopback.py`, `verification_state.py`, `pending_booking.py`, `keypad.py` (DTMF), `language.py` +
  `i18n/` (greeting, booking, cancellation, availability, language switch), `eot_config.py`, `turn_detection/`,
  `llm_error_policy.py`, `tool_steps.py`, `transition_metrics.py`, `observer.py`, `simulation.py`, `vision_mixin.py`.
- `events.py`, `tools.py`, `call_outcome.py`, `waitlist_outbound/`, and the vendor-specific integration agents.
- Entry: `agent.py` (`AGENT_MAP`), `agent_workflow.py`, `hatchet_worker.py`, `multi_agent_runner.py`,
  `metrics_exporter.py`, `memory_monitor.py`, `shutdown_handler.py`; `prompts/` (one universal prompt, config-flag
  driven), `KMS/`, `envs/`, `trajectories/` (recorded calls for replay).

## Hatchet workflows (`backend/integrations/hatchet/`)

call connected, callback + callback cron, outreach and sequential outreach, freed-slot matching, offer expiration
(+ cron), waitlist entry expiration cron, session workflow, trial reminder cron, cleanup calls cron, clinic readiness
check, health probe, knowledge-base scraping, phrasebook sync, Stripe plans sync, vendor appointment / patient /
dynamic sync + token refresh, error handling, browser-automation integration tests, unit-tests workflow.

## Tests — four layers (`agent/tests/`)

1. `unit/` — pure, run in the container (`docker exec … pytest /app/tests/unit`); no venv in `agent/`.
2. `behavioral/` — the simulated-caller harness against the real agent in text mode: `patient_simulator.py`,
   `patient_profiles.py`, `scenarios/` (declarative, loaded by `scenario_loader.py`), `task_harness.py`, `assertions.py`,
   `call_state.py`/`call_states/`, `judges.py` + `judge_calibration.py`, `golden.py`, `paraphrase.py`, `contracts/`,
   `lint_policy.py` (tests must follow the policy), reports to a store; costs real LLM money — run on demand
   (`behavioral-on-demand.yml`), never to "check things".
3. `e2e/` — nightly end-to-end.
4. `voice/` — real-call happy-path regression with latency measured as "response heard"; reports published to a store
   behind CloudFront; the quality series diffs each run against the previous.
Plus `bench/` and `stt_bench/` (STT WER/latency benchmark with its own corpus), `fixtures/`, `TEST_COVERAGE_PLAN.md`,
and the feature-coverage registry (every test linked to a feature id; `--check` in CI).

## CI / release / infra

- GitHub Actions: `pr-checks.yml` (lint, unit), `ci.yml` (build images, build dashboard, deploy prod on release),
  `behavioral-on-demand.yml`. Trunk-based: short feature branches, PR required, direct push to main blocked.
- Release: `scripts/release.sh` writes an evidence table into the notes and attaches the test reports; the OpenAI key
  is shared by the Mac workers and CI, so a 429 invalidates a run.
- AWS: ECS services behind an ALB, Aurora Postgres in a private VPC (no public endpoint), SSM bastion tunnel for the
  Django admin and the Hatchet dashboard (`scripts/prod_tunnel.sh`), `scripts/ecs_exec.sh` to run an ORM query inside the
  live task without handling DB credentials, spot-interruption capture, Lambdas for a service prober and a session time
  limiter; infrastructure in a separate devsecops repo (Terragrunt).
- Observability: OpenTelemetry → Jaeger locally, Grafana dashboards for the LiveKit agent (dev/prod), Sentry in every
  service, Discord notifications, structured event logging; prod log retention 60 days.
- Style: black (120), flake8, isort, bandit, pre-commit.

## Principles written into the repo

- Zero-code, zero-admin onboarding of a clinic: everything tenant-specific is dashboard configuration.
- All queries tenant-scoped; no global `.all()` on tenant models; `@tenant_user_api` / `@tenant_admin_api` / `BaseTenantViewSet`.
- Tenant-config schema changes ship with seed data, a back-compat reader, a documented manual step and a team notice.
- A code edit is not live in the running agent worker until the container is recreated (forked job processes).
- Never truncate pytest output; re-run only the failed ids.
- Specs: time-bound plans under `specs/<feature>.md`, long-lived design under `specs/reference/`, promoted when done.
