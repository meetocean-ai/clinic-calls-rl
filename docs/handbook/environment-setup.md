# Recreating the development environment

The whole stack that produced this repo runs on one laptop (Apple-silicon Mac, 16 GB is enough; Linux works with the
noted substitutions) for $0: a local FHIR server, a local open LLM, local TTS and ASR, and Python. Nothing below needs a
paid account. Times are from a clean machine.

## 1. Base tooling (10 min)

```bash
# macOS
brew install git gh pnpm uv python@3.12 docker ollama ffmpeg
# Linux: apt install git docker.io ffmpeg; curl -LsSf https://astral.sh/uv/install.sh | sh; curl -fsSL https://ollama.com/install.sh | sh
gh auth login
```

- **uv** manages every Python environment (fast, lockfile, `uv pip install -e …`). Python **3.12** for anything that
  touches training libraries (TRL / torch / verifiers / openenv-core); the agent service itself ran on 3.14, which is
  why the training venv is separate.
- Check disk before installing torch or building images: `df -h /`. A 16 GB `uv` cache plus Docker image builds filled
  the disk once and crashed Docker Desktop (recovery: `uv cache clean`, `docker builder prune`, relaunch; never
  "Reset to factory defaults" — it wipes the database volumes).

## 2. The FHIR server — Medplum, local (15 min)

The environment seeds one Organization (tenant) per episode on a local Medplum (FHIR R4 server with auth, search,
compartments). In the private repo: `docker compose -f docker-compose-medplum-local.yml up -d`, then a bootstrap script
that creates a client application and writes `MEDPLUM_BASE_URL=http://localhost:8103`, `MEDPLUM_CLIENT_ID`,
`MEDPLUM_CLIENT_SECRET` into a local env file that is sourced before every run. Without the private repo: Medplum's own
`docker-compose.yml` (Postgres + Redis + server + app) at https://github.com/medplum/medplum gives the same server; create
a ClientApplication in the admin app and use its id/secret.

Facts the code depends on: compartment search (`_compartment=Organization/<id>`), `_count` max 1000, token lifetime
about one hour (the adapter refreshes on 401), `Task.code` token search, `Patient?phone=`.

## 3. The open model — Ollama (10 min + download)

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve &     # the default 4096 silently truncates the agent prompt + tools
ollama pull qwen3:8b                            # ~5 GB; the untrained baseline policy
```

- Served at `http://localhost:11434/v1` (OpenAI-compatible). Qwen3 thinks for ~1k tokens a turn unless told not to;
  on Ollama's OpenAI endpoint the only switch that works is `reasoning_effort: "none"` (`think: false` and the
  `/no_think` soft switch are ignored). 15–25 s per episode after that on an M1.
- vLLM is the same interface on a GPU box: `vllm serve Qwen/Qwen3-8B --enable-lora`; vLLM-Omni (`--omni`) for
  Qwen3-Omni audio in/out.

## 4. Voice: TTS and ASR, local (10 min + downloads)

```bash
uv pip install kokoro misaki "transformers>=4.46" "tokenizers>=0.21" soundfile scipy numpy
uv pip install mlx-whisper            # Apple silicon; elsewhere: faster-whisper
python -c "from huggingface_hub import snapshot_download; snapshot_download('hexgrad/Kokoro-82M', allow_patterns=['voices/*','*.json','*.pth'])"
python -c "import mlx_whisper, numpy as np; mlx_whisper.transcribe(np.zeros(16000,dtype='float32'), path_or_hf_repo='mlx-community/whisper-base.en-mlx')"
```

- Kokoro-82M (Apache-2.0): 13 voices used (`af_*`, `am_*`, `bf_*`, `bm_*`), 24 kHz, ~1.3 s per line on CPU. Download
  **all** voices once; with `HF_HUB_OFFLINE=1` a missing voice file crashes a sweep.
- Whisper `base.en` is enough for the round-trip check and the ASR cascade (95% of caller lines intelligible, 95% of
  facts heard at the tiers used). On Python 3.14 faster-whisper's `tokenizers` pin did not install — mlx-whisper did.

## 5. The Python environments (10 min)

```bash
# agent service + behavioural suite (what the env imports); in the private repo:
cd services/agent && uv venv .venv && uv pip install -e . -e ../../tests/behavioral[voice,env-server]
# training venv (TRL 1.14 / torch / verifiers 0.1.5 / openenv-core 0.3.0); separate because `agents` is a package name clash
cd tests/behavioral && uv venv --python 3.12 .venv-verifiers && .venv-verifiers/bin/uv pip install trl torch peft verifiers openenv-core
```

Run everything from `tests/behavioral` with `PYTHONPATH=../../services/agent:.` and the Medplum env file sourced.

## 6. Smoke, in order (5 min)

```bash
python -m pytest env -q                                            # ~420 tests against the live local Medplum, zero mocks
python -m env.policies --policy oracle --policy random --split heldout --seeds 2 --tier 3   # the bracket: 1.0 / 0.0
python -m env.policies --policy openai-compatible --endpoint http://localhost:11434/v1 --model qwen3:8b --no-think --scripted --seeds 1
python -m env.policies --policy oracle --voice --seeds 1 --tier 3  # Kokoro → phone line → Whisper → oracle
python -m env.dataset.build --out results/env/clinic-calls-v1 --seeds 2 --tiers 1 --tts tone   # the package, 2 minutes
```

## 7. Serving and training (when there is a GPU)

- `EHR_ENV_SCRIPTED_CALLER=1 python -m env.server` → OpenEnv at `127.0.0.1:8011` (`/ws`), `EHR_ENV_AUDIO=1` for audio mode;
  `docker compose -f docker-compose-env.yml up` for the containerised env + Medplum.
- `.venv-verifiers/bin/python -m env.train_smoke --steps 2 --generations 4 --model Qwen/Qwen3-0.6B` proves the
  TRL `environment_factory` loop on CPU/MPS (load the model on CPU first, `HF_HUB_OFFLINE=1` after `hf download`).
- Real runs: `docs/rl-training-runbook.md` (text, prime-rl, Qwen3-8B LoRA) and `docs/rl-s2s-training-recipe.md`
  (speech, Qwen3-Omni) in the private repo — the recipe is also summarised in `system-design-notes.md` here.

## 8. What is NOT reproducible from this repo

The agent service (LiveKit agents, the product's FHIR tools, the caller simulator) is private; `env/` imports it. The
dataset package (`clinic-calls/`) and its grader are self-contained and run anywhere with Python ≥ 3.10.
