# Tools and frameworks — what, why, and what we learned

Grouped by layer. "Lesson" is something we hit for real, not a reading note.

## Serving the environment to a trainer

| tool | what we used it for | why it, and the lesson |
|---|---|---|
| **OpenEnv** (`openenv-core` 0.3.0) | the env as a server: `reset`/`step` over `/ws`, typed `EhrAction` / `EhrObservation` / `EhrState`, a Hugging Face "space" layout with `openenv.yaml` | The emerging standard the trainers speak (TRL has a native `environment_factory` for it). Lesson: `openenv validate` wants a `uv.lock`, a `[project.scripts] server` entry, a `main()` and `server/Dockerfile`; the WebSocket protocol is `{"type":"reset","data":{…}}` and step data is the action dict. |
| **verifiers** (0.1.5) | the env as a `vf.Environment` for prime-rl / `vf-eval` | prime-rl's native interface. Lesson: the 0.1.5 API has no `vf.stop`, `evaluate` is sync, `state["prompt"]` is not reused after setup — deliver the opening line as the first env response; own venv because its `agents` import clashes with the OpenAI Agents SDK package name. |
| **TRL** (1.14) | GRPO smoke on the Mac with Qwen3-0.6B LoRA | Has `GRPOTrainer(environment_factory=…)`. Lessons: `max_prompt_length` was removed; the threaded loader segfaults on MPS — load the model on CPU and hand it over; `hf download` first and run with `HF_HUB_OFFLINE=1`. Reward 0 / std 0 for the bare 0.6B is expected — the wiring is the proof. |
| **prime-rl** | the recipe for the real text run (Qwen3-8B, LoRA r=16/α=32, K=8, async rollouts) | Trains against verifiers envs; the config keys are to be re-checked against its reference the day of launch. Not run (paid). |
| **SkyRL-train** | the alternative trainer in the speech recipe (vLLM-Omni rollout backend) | Config equivalents written next to the prime-rl ones. Not run. |
| **vLLM / vLLM-Omni** | serving a LoRA checkpoint (`--enable-lora`), Qwen3-Omni audio in/out (`--omni`, `audio_url` parts, `/v1/audio/speech`) | OpenAI-compatible, so one adapter (`OmniPolicy`) serves training rollouts and evaluation. |
| **Ollama** | the untrained Qwen3-8B baseline on the laptop | Free, OpenAI-compatible. Lessons: `OLLAMA_CONTEXT_LENGTH=16384` or the tools vanish; `reasoning_effort: "none"` is the only working no-think switch on its OpenAI endpoint. |
| **LiteLLM** | the product's single LLM proxy (every agent LLM call goes through it; cost tracking; model-name routing to OpenAI / Vertex / Ollama / vLLM) | One place to switch a model profile. Lessons: it reads its config at start (recreate, don't restart); a `429 no credits` from the upstream key blocks every LLM-driven column at once — probe before a sweep. |

## Models

| model | role | licence (checked 2026-10) |
|---|---|---|
| Qwen3-8B | untrained baseline, the text-training target | Apache-2.0 |
| Qwen3-Omni-30B-A3B | speech-to-speech training target; audio-native caller; audio auditor | Apache-2.0 |
| Kokoro-82M | caller voices (13) | Apache-2.0 |
| Whisper base.en (mlx-whisper / faster-whisper) | the ASR in the round-trip check and the cascade | MIT |
| NemotronLabs VoiceChat 11B | full-duplex S2S with a tool-call channel (adapter written) | OpenMDW v1.1, research only → benchmark rows, never a product path |
| PersonaPlex-7B (Moshi architecture) | full-duplex S2S, no tools → delegation to a text LLM (adapter written) | NVIDIA Open Model License (commercial OK, derivative terms) |
| gpt-realtime, Gemini Live | closed references (adapters written, per-minute pricing, capped) | hosted |
| gpt-4.1-mini | the LLM caller simulator in the product's harness | hosted |

## Voice stack

| tool | what | lesson |
|---|---|---|
| **LiveKit Agents** (Python) | the production voice agent: `AgentSession(stt, llm, tts, vad)`, SIP via Twilio, text mode for the behavioural harness | The worker forks job processes from a preloaded server: a code edit is NOT live until the container is recreated. Its default LLM timeout (10 s, 3 retries) kills a 30 s local model — `OCEAN_LLM_TIMEOUT_S` for the harness only. `openai.realtime.RealtimeModel(base_url=…)` is how a self-hosted S2S model becomes the whole pipeline (no STT/TTS). |
| **Deepgram nova-3** | production STT | In our own benchmark it beat Gemini streaming STT (7.9% vs 11.8% WER, 0.19 s vs 1.34 s); natural-speech WER under ~5.5% is vendor level. |
| **Deepgram Aura / Cartesia** | production TTS | — |
| **Silero VAD** | turn detection | End-of-utterance delay is the latency floor a cascaded agent cannot beat (~1.5 s ceiling measured). |
| **G.711 μ-law / 8 kHz** | the phone-line model in the degradation chain | Pure numpy/scipy: resample, 300–3400 Hz band-pass, SNR noise, μ-law round trip, 20 ms frame drops, gain wobble — seeded per task. |
| **soundfile (libsndfile ≥ 1.1)** | WAV and Ogg/Opus writing | Opus for the agent's recorded clips (3× smaller than WAV). |

## Data and backend

| tool | what | lesson |
|---|---|---|
| **Medplum** (FHIR R4) | the EHR: Patient, Practitioner, Schedule, Appointment, Task, Coverage, AuditEvent; one Organization compartment per tenant/episode | Reward is read from the compartment, never from text. Token lifetime ~1 h: refresh on 401 or an hour-long sweep dies. `_count` caps at 1000. |
| **FHIR R4** | the data model | `Appointment.participant[].actor` carries Patient and Practitioner refs + display; `Task.code.coding.code` for agent actions (`warm-transfer`, `waitlist`); `Coverage` for insurance; `AuditEvent` grows on every read (exclude from snapshots). |
| **Postgres** (marts) + **Hatchet** | analytics marts and workflow orchestration in the product | Not touched by the RL work; the tenant export list (`_EXPORT_RESOURCE_TYPES`) is reused for snapshots. |
| **Docker / docker compose** | Medplum stack, LiteLLM, the env container | Disk fills fast with image builds; prune. |

## Evaluation and quality

| tool | what | lesson |
|---|---|---|
| **pytest** (+ `pytest.mark.covers`) | every suite; a feature registry links each test to a feature id, `build.py --check` fails on unknown ids | Never truncate pytest output; capture to a file and grep. `-x` + a `grep failed` gate let a collection *error* through once — gate on `failed|error`. |
| **Wilson 95% intervals, pass@1 / pass^k** | every rate on the leaderboard | Ten tasks per family → wide intervals; show them, do not hide them. |
| **Quality series in git** (own tool) | dated reports, per-test history, runs as slim records; the leaderboard is a view over them | "Latest run per suite" needs the suite label to carry the tier, or two tiers overwrite each other's history. |
| **Allure** | replaying the committed reports with chained history | — |
| **DeepEval / Ragas / Promptfoo** | the product's judge/eval stack (private repo) | Judges are logged next to the state-verified reward, never in it. |
| **τ-bench / τ-voice** (Sierra) | the task format and the telephony-degradation recipe we borrowed | Open, seeded, trajectories shipped. |
| **Full-Duplex-Bench v3** | turn-taking metric definitions (takeover, backchannel, barge-in) | Our `turntaking.py` computes them from the two audio streams' geometry. |
| **SpeechGym** (paper, no code) | the audio-native GRPO recipe reproduced in `rl-s2s-training-recipe.md` | Outcome-only groups were 84% all-zero without a process reward — hence the curriculum and the (calibrated-first) tool credit. |
| **Iterative Reward Calibration** (paper) | why dense per-turn rewards are gated on correlation with success | Naive dense rewards cost 14 pp on τ-bench airline. |
| **VAmoS Bench** | "claimed success without tool evidence" as a first-class metric | Our `claim_without_tool` judge and the wrong-write component. |

## Hubs and licences

| thing | note |
|---|---|
| Hugging Face Hub | dataset + env packages (dataset card, `openenv.yaml` space); publishing waits for the account |
| Prime Intellect environments hub | the verifiers package (`vf_package`) |
| PolyForm Noncommercial 1.0.0 | all published code |
| CC BY-NC 4.0 | the dataset |
| Never MIT / Apache / CC-BY for our publications | the product owner's decision |
