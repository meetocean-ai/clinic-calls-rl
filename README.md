# clinic-calls-rl

A reinforcement-learning environment and benchmark for front-desk phone calls to a clinic, exported from the Ocean EHR
test infrastructure for a short review window. Everything here is **free for noncommercial use only**: code under the
PolyForm Noncommercial License 1.0.0 (`LICENSE`), the dataset under CC BY-NC 4.0 (`LICENSE-DATA`). Every identity in
the data is synthetic.

## What is here

| path | what |
|---|---|
| `clinic-calls/` | the built **clinic-calls v1** package: 160 held-out tasks (6 families × tiers 1/3/4), the clinic's FHIR snapshot before each call, the oracle's reference run and end state, the caller's lines as 8 kHz telephone audio (742 clips), and `verifier/verify_json.py` — a standard-library grader. Start with `clinic-calls/README.md`. |
| `env/` | the environment source as it lives in the private repo: task generator (`tasks.py`), world seeding (`world.py`), state-verified reward (`verify.py`), the `step/reset` env (`env.py`), reference policies and sweep CLI (`policies.py`), scorecard/report, OpenEnv server + client, `verifiers`/TRL adapters, the voice channel (`audio/`: Kokoro voices, telephony degradation, disfluencies, ASR cascade, turn-taking, audio judges), S2S adapters (`s2s/`), the dataset builder (`dataset/`). |
| `docs/clinic-calls/METHODOLOGY.md` | how numbers are produced: binary state-verified reward, mean of k trials, Wilson intervals, four components (task success, wrong writes, claimed-without-tool, latency), voice gap. |
| `docs/clinic-calls/README.md` | the leaderboard as generated from the recorded runs. |
| `docs/plans/rl-speech-to-speech-roadmap.md` | the plan with every task's status and evidence. |

## What runs here, and what does not

- **`clinic-calls/verifier/verify_json.py` runs anywhere** (Python ≥ 3.10, no dependencies): load a snapshot into a FHIR
  R4 server, let your agent take the call, export the compartment, grade. See `clinic-calls/README.md`.
- **`env/` is reference code.** It imports the private product's agent service (`agents.tools_fhir`, `runtime.*`,
  `fixtures.*`) for the live Medplum tools and the caller simulator, so it does not run from this repo alone. It is here
  so the reward, the knobs, the voice channel and the adapters can be read and reviewed.
- No model weights, no keys, no recorded LLM transcripts beyond the oracle reference runs.

## Headline numbers (held-out, one verifier stamp)

| policy | tier 3 | tier 4 |
|---|---|---|
| oracle (reads the task; the fairness gate) | 60/60 | 60/60 |
| oracle hearing the caller through a local Whisper (the ear's own loss) | 60/60 | 57/60 |
| random | 0/60 | 0/60 |
| Qwen3-8B, untrained, over the environment's tools | text 0.172 / voice 0.200 (only emergency_redirect passes) | — |
| the production agents (hosted LLM through LiteLLM), text, k=3 | 0.944 (0.90–0.97) | 0.978 (0.94–0.99) |

Every production miss is read and becomes either a task fix or a numbered product bug (B-137 … B-142 in the roadmap's P0-14 row) — the environment found six real defects in the agents it was built to train against.

Dataset `tasks-v3+gen-c605a9eb4c`, verifier `verify-b4440d9c68`, methodology v1 — see the docs for what each means.

## Attribution

MeetOcean AI, 2026. Noncommercial use only; contact the authors for anything else.
