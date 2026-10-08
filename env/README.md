# EHR scheduling environment

The RL environment for the clinic's phone line: seeded synthetic tasks (`tasks.py`), a fresh Medplum tenant per
episode (`world.py`), the production tool functions behind an OpenAI-style tool interface (`env.py`), and a binary
reward read from the FHIR end state plus a deterministic safety gate over the agent's utterances (`verify.py`). Reference policies and the sweep CLI
live in `policies.py`; the plan that drives the work is `docs/plans/rl-speech-to-speech-roadmap.md`.

**What a task is** — `DATASET.md` is the frozen description (dataset v2, 2026-10-07): six families (booking, cancel,
reschedule, verify_fail_transfer, new_patient_intake, emergency_redirect), 14 knobs, four tiers (tier 4 = two or
three knobs together), seven caller-realism traits and a cooperation level, the split rule and the generator hash.
A task is `generate(family, seed, tier=…)`; `validate(task)` lists what is wrong with a hand-edited one.

```bash
cd tests/behavioral
PYTHONPATH=../../services/agent:. python -c "from env.tasks import generate; print(generate('booking', 5, tier=4).to_json())"
```

**Reward** — `verify.py` reads the episode tenant back and runs the family's checks (`CHECKS_BY_FAMILY`, plus the
knob-specific ones `checks_for` adds); reward 1 only when every check passes, otherwise each failure is one named diff
(`booking_matches: …`, `no_billing_write: …`). Nothing in the transcript earns reward — the safety gate can only take it away; the per-turn judges
(`turn_scores.py`) are logged next to the reward for analysis.

## Prerequisites

- Local Medplum stack up (`docker-compose-medplum-local.yml`) and `.medplum-local.env` sourced — `MEDPLUM_BASE_URL`
  must point at `localhost`; the environment refuses hosted servers.
- Local LiteLLM proxy up (`docker-compose-local.yml`, container `ocean-litellm`). The proxy reads
  `services/agent/litellm/config.yaml` **at start**: after editing the model list, recreate the container
  (`docker compose -f docker-compose-local.yml up -d litellm --force-recreate`), a `restart` is not enough.
- `PYTHONPATH=services/agent:tests/behavioral` and the agent's virtualenv (`services/agent/.venv`).

### The untrained open model (policy `qwen`)

The baseline column that the trained model has to beat runs the production agents on Qwen3-8B through the
`training-stack` profile (`infra/model_profiles.yaml`): only the agent capabilities move to the open model; the
patient simulator stays on `gpt-4.1-mini` so the caller is the same in every column.

- Ollama with the model pulled (`ollama pull qwen3:8b`) and a context window large enough for the agent prompt
  plus tools: start the server with `OLLAMA_CONTEXT_LENGTH=16384` (the default 4096 truncates the system prompt
  silently and the model stops calling tools).
- `OCEAN_MODEL_PROFILE=training-stack` — the `qwen` policy refuses to run under any other profile.
- `OCEAN_LLM_TIMEOUT_S=180` — LiveKit caps each LLM attempt at 10 s; an 8B model on a laptop takes 15–40 s a turn.
- `OCEAN_SIMULATOR_CAPABILITY=analytics` — keeps the caller on the hosted model under this profile.

```bash
cd tests/behavioral
OCEAN_MODEL_PROFILE=training-stack OCEAN_LLM_TIMEOUT_S=180 OCEAN_SIMULATOR_CAPABILITY=analytics \
PYTHONPATH=../../services/agent:. python -m env.policies --policy qwen --split heldout --seeds 5 --k 1 --tier 3 \
  --out results/env/<sweep> --json-out results/env/<sweep>.json
```

## Running a sweep

```bash
cd tests/behavioral
PYTHONPATH=../../services/agent:. python -m env.policies --policy oracle --policy random --split heldout --seeds 5 --tier 3 \
  --out results/env/<sweep> --json-out results/env/<sweep>.json
PYTHONPATH=../../services/agent:. python -m env.policies --policy production --seeds 2 --k 3 --tier 3 --out results/env/<sweep>
```

Every episode writes `<out>/<run-id>/<policy>/<task-id>/<trial>.{jsonl,manifest.json}`; the manifest carries the
reward, the named diffs, the dataset and method hashes (`versioning.py`) and the per-turn judge verdicts (logged,
never part of the reward). `--tier 4` runs the combinations; `--chaos a,b` pins the knobs (one family per run, the
menu differs per family); the held-out evaluation set is `--split heldout --seeds 10` (= `HELDOUT_EVAL_SEEDS`).

## Reading the scorecard

```bash
PYTHONPATH=../../services/agent:. python -m env.scorecard results/env/<sweep>/<run-id> --knobs [--baseline results/env/<previous>/<run-id>]
PYTHONPATH=../../services/agent:. python -m env.report results/env/<sweep>/<run-id> --title "…" --out results/env/<sweep>.html
```

The scorecard has one row per policy × family × tier: mean reward with its Wilson 95% interval, pass^k (every trial of
a task passed) and pass@k, the escalation rate, turns, tool calls, turn latency p50/p99, errors and the dataset /
method hashes the rows carry — two rows are only comparable when both hashes agree. `--knobs` adds the per-knob table:
pass rate over every episode whose task carries the knob (alone or stacked at tier 4), with the check that failed most
often — the view that says which difficulty a model is actually losing on. The HTML report has the same tables plus
every miss with its transcript and judge flags.

## Serving the environment (OpenEnv)

`env/server.py` puts `EhrSchedulingEnv` behind the OpenEnv `reset / step / state` contract — one environment per
WebSocket session — so a trainer or any OpenEnv client runs episodes over the network instead of in-process:

```bash
pip install -e tests/behavioral[env-server]            # openenv-core[core]
cd tests/behavioral
PYTHONPATH=../../services/agent:. python -m env.server   # 127.0.0.1:8011 (ENV_HOST / ENV_PORT / MAX_CONCURRENT_ENVS)
```

Reset carries `seed`, `family`, `tier` and optional `chaos`; a step is an `EhrAction` (`say` / `tool` / `end_call` /
`transfer`, the same four kinds as in-process); `state` reports the task, turn and tool counts, `verified`, the
reward and diffs once done, and the dataset / method hashes. `GET /schema` serves the pydantic schemas. The reference
client is `env.server.EhrEnvClient` (an OpenEnv `EnvClient`). The server binds to localhost and has no auth of its
own: the trainer and the environment share a box or a private network, like the local Medplum it writes to.
`EHR_ENV_SCRIPTED_CALLER=1` swaps the LLM caller for the deterministic one (what the tests use).

### As a `verifiers` environment (prime-rl, `vf-eval`)

`env/verifiers_env.py` wraps the served environment as a `verifiers.MultiTurnEnv` (`load_environment()` is the
entry point). It runs out of process in its own venv because `verifiers` depends on `openai-agents`, whose top-level
package `agents` collides with the product's:

```bash
cd tests/behavioral
python3 -m venv .venv-verifiers && .venv-verifiers/bin/pip install "verifiers==0.1.5" "websockets>=15" httpx
# server in the agent venv (above), then any OpenAI-compatible model — a local Ollama works:
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
  .venv-verifiers/bin/python -m env.verifiers_env --ws ws://127.0.0.1:8011/ws --model qwen3:8b --examples 2 --tier 3
```

The model speaks plain text to the caller or emits one JSON object `{"tool": …, "arguments": {…}}`; the rubric's only
term is the server's `verify()` reward on the last step.

### GRPO smoke on the Mac (TRL)

`env/trl_env.py` is the environment in TRL's `environment_factory` shape (public methods = tools, `reset(**kwargs)`
opens an episode on the server, the reward function reads `verify()`'s result) and `env/train_smoke.py` runs a few
GRPO steps of a small Qwen3 with a LoRA adapter on MPS through the whole loop — trainer → tool calls → OpenEnv server →
environment → Medplum → reward → update. It proves the wires, not a quality target; the real recipe is
`docs/rl-training-runbook.md`.

```bash
# server (agent venv) with the deterministic caller
cd tests/behavioral && EHR_ENV_SCRIPTED_CALLER=1 PYTHONPATH=../../services/agent:. python -m env.server
# trainer (training venv: python 3.12 + trl + torch + peft + openenv-core)
.venv-verifiers/bin/python -m env.train_smoke --steps 3 --generations 4 --model Qwen/Qwen3-0.6B
```

## The voice channel (`env/audio`)

- `tts.py` — the caller's voice: a roster of 13 Kokoro voices (gender and accent spread), `voice_for(seed)` picks one
  deterministically per task; `load_tts()` returns `KokoroTTS` when `tests/behavioral[voice]` is installed, else the
  dependency-free `ToneTTS` the tests use.
- `degrade.py` — the phone line: `DegradationSpec.from_seed(seed, tier)` (clean at tiers 1–2, a mobile call at 3, a bad
  one at 4) and `degrade(samples, sr, spec)` → 8 kHz, 300–3400 Hz, noise at the SNR, μ-law, frame drops, gain wobble.
- `disfluencies.py` — fillers, false starts and self-corrections injected into a caller line before TTS, never into
  dates, times, numbers or `Dr. Name`.
- `roundtrip.py` — the fairness check for speech: synthesize → (degrade) → transcribe with a local Whisper → the
  normalized similarity must be ≥ 0.85, or the line does not ship.
- `cascade.py` — the cascaded voice policy (P2-04): `CascadedVoicePolicy(inner)` puts a local ASR in front of any text
  policy, so the agent works from the transcript of the caller's phone audio while the env keeps scoring the FHIR end
  state and the safety gate keeps reading what the caller really said. Policy id = `<inner>+voice`; the manifest's `asr`
  block logs every line (said / spoken / heard / similarity). The production agents get it through
  `drive_inbound(simulator_wrap=…)` (`VoicedCaller`), step-driven policies through `HeardEnv`.

```bash
PYTHONPATH=../../services/agent:. python -m pytest env/audio -q      # signal-property tests; model tests skip without weights
```

### The voice gap

Same policy, same held-out tasks, once reading the caller's text and once hearing it (`--voice`; the env runs in
audio mode, Kokoro + mlx-whisper on the Mac). `--no-think` switches Qwen3's thinking off on Ollama's OpenAI endpoint
(`reasoning_effort: none` — `think: false` and `/no_think` are ignored there), which turns 40–60 s turns into 5–10 s.

```bash
PYTHONPATH=../../services/agent:. python -m env.policies --policy openai-compatible --endpoint http://localhost:11434/v1 \
  --model qwen3:8b --no-think --scripted --split heldout --seeds 10 --k 3 --tier 3 --out results/env/p204 --run-id t3
PYTHONPATH=../../services/agent:. python -m env.policies --policy openai-compatible --endpoint http://localhost:11434/v1 \
  --model qwen3:8b --no-think --scripted --voice --split heldout --seeds 10 --k 3 --tier 3 --out results/env/p204 --run-id t3
PYTHONPATH=../../services/agent:. python -m env.voice_gap --text results/env/p204/t3/openai-compatible-qwen3-8b \
  --voice results/env/p204/t3/openai-compatible-qwen3-8b+voice
```

`voice_gap` pairs the tasks, prints pass@1 for both modes with Wilson intervals, the gap, how many tasks were lost or
gained in voice, and what the ear did (mean transcript similarity, lines under the 0.85 floor). Record the two runs
into the quality series separately — the `+voice` id keeps them as two columns.

## Recording into the quality series

```bash
python3 tests/quality/build.py record --env tests/behavioral/results/env/<sweep>
python3 tests/quality/build.py report
```

The sweep becomes a run under `docs/quality/runs/` labelled `rl-env-<policies>-<families>`, one row per episode,
and the next report diffs it against the previous one. Sweeps that drive an LLM are kept as samples, never pruned.

Record one policy per run (point `--env` at `<out>/<run-id>/<policy>`): the oracle sweep is the fairness gate
(must be 100%), production and the open model are the columns being tracked. The random policy is the floor of the
bracketing table in the plan, not a quality signal — a run of expected zeros is never recorded.

## Fairness rule

A task or knob ships only when the oracle policy scores 1.0 on it over the held-out seeds (`seed % 5 == 0`; the
frozen set is `HELDOUT_EVAL_SEEDS`, ten per family) — tier 3 for each knob alone, tier 4 for the combinations
(`env/test_knobs_v2.py`). The random policy is the floor (0 on every action family); the production agents and the
untrained open model are the two columns in between. A production miss is triaged, never tuned away: either the task
was unfair (fix the task, oracle must still be 1.0) or it is a product bug (BLOCKERS entry with the episode id).

## Changing the dataset

`DATASET.md` pins `dataset_version` (a hash of `tasks.py` + `world.py`); `test_dataset_md_matches_the_generator`
fails when either file changes. A deliberate change bumps `EhrTask.schema_version`, updates the pinned line and the
history section, and re-runs the oracle gate. Runs recorded under different hashes are never compared as one series.
