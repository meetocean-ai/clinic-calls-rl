# Talking points — numbers, decisions, lessons (two pages)

## What this is, in one breath

A reinforcement-learning environment for a clinic's front-desk phone agent: every episode is a fresh FHIR tenant, the
caller is simulated (text or synthesized phone audio), the agent uses the product's real scheduling tools, and the
reward is binary, read from the clinic's records after the call — never from what the agent said. On top of it: a
published benchmark (`clinic-calls`, 160 held-out tasks, grader included), a versioned methodology, a leaderboard, and
the recipe to train open text and speech models against it with GRPO.

## Numbers (held-out, 10 seeds × 6 families, verifier `verify-b4440d9c68`)

| what | number |
|---|---|
| Oracle (reads the task): fairness gate | 60/60 at tier 3, 60/60 at tier 4 |
| Oracle hearing the caller through Whisper: the ear's own loss | 60/60 tier 3 (gap 0.000); 57/60 tier 4 (gap +0.050) |
| Random policy: the floor | 0/60 at both tiers |
| Qwen3-8B untrained, over the env's tools, scripted caller, tier 3 | text 0/132 recorded so far (role-plays the caller, hallucinates dates, never calls end_call); voice 4/110 recorded so far |
| Caller audio quality (180 clips, Kokoro → phone line → Whisper) | 95% intelligible, 95% of the facts heard; tier 1 98%, tier 3 93%, tier 4 82% (tier 4 is the bad line by design) |
| Environment throughput on a laptop | reset p95 0.2 s; 14,400 oracle episodes/hour at 8 concurrent; no tenant pool needed |
| Dataset package | 160 tasks, 742 audio lines, 160/160 reference runs re-graded identically by the shipped grader, 51 MB |
| Tests | ~425 environment tests against a live local FHIR server with zero mocks; 1,021 agent unit tests; coverage registry links every test to a feature |
| Spend so far | $0 on GPUs; the hosted-model columns are blocked by an empty OpenAI balance, not by code |

## Decisions I would defend

1. **State-verified binary reward; judges logged, not rewarded.** It removes reward hacking by phrasing and makes every
   miss a named, reproducible diff. Dense per-turn rewards are added only after their correlation with success is
   measured (the IRC paper shows the naive version costs 14 pp).
2. **Bracket before training.** Oracle 1.0 and random 0 on every knob — or the knob is wrong, not the model.
3. **Difficulty as composable, verifiable knobs** with exclusion rules, tiers, and a held-out split by seed.
4. **Version stamps on every episode**, hashed at import; rows under different stamps are never compared.
5. **Tools stay textual in voice mode** so perceptual errors are separable; the voice gap is the measurement.
6. **One tenant per episode** instead of a pool: simpler, and the numbers said the pool was unnecessary.
7. **Ship the grader, not the runtime**: a pure-JSON verifier with proven parity lets anyone grade without our stack.
8. **Independent review before publishing numbers**: two reviewer passes found three real reporting bugs.
9. **Free first.** Everything above ran on one laptop; every paid step has a cap and a ledger line before it starts.

## Lessons (things that actually bit)

- Auth tokens expire at one hour; a sweep that runs longer must refresh — two long sweeps died at exactly 60 minutes.
- Editing the verifier while a sweep runs relabelled half its episodes — hash versions at import, not per call.
- A verifier exception became a silent `None` reward; make it an error with a message.
- Hard-coded `slot_index: 0` in stubbed models is order-dependent; a stub must pick the provider the caller named.
- Whisper is not bit-deterministic on degraded audio: 55 vs 57 of 60 across two identical runs — report intervals.
- A reporting test wrote to the committed leaderboard; generated artefacts must write next to the data they render.
- Disk filled by a torch install + image builds took Docker down; check `df` before big installs.
- Ollama ignores `think:false` on its OpenAI endpoint; `reasoning_effort:"none"` works. The default 4k context drops the tools silently.
- pytest's assertion introspection printed a dataclass repr with an auth header — `repr=False` on secrets.
- A `grep failed` gate let a *collection error* through to a commit — gate on `failed|error`.

## What is next (and what it needs)

| next | needs |
|---|---|
| Production-agent columns (text + voice gap), triage of every miss, difficulty loop to 30–70% for the untrained model | OpenAI credits on the right project |
| Reward calibration (per-turn tool credit vs success correlation) | the columns above |
| First text GRPO run: Qwen3-8B LoRA, 2–4 h on one H100 (≤ $150/attempt) | a go + a ledger line |
| Open S2S rows (Qwen3-Omni, VoiceChat, PersonaPlex) on a rented GPU (≤ $50) | a go + a ledger line |
| Publish `clinic-calls` v1 and the env packages | a Hugging Face account |
| Speech GRPO (Qwen3-Omni thinker LoRA, 8-GPU node, recipe written) | $5–15k, after the text run shows a held-out gain |

## If asked "how would you build X" — where the answer is

`system-design-notes.md`: §1 voice agent, §2 RL environment + training loop, §3 benchmark/leaderboard, §4 HIPAA
guardrails. `tools-and-frameworks.md` has the catalog with one lesson per tool. The plan
(`docs/plans/rl-speech-to-speech-roadmap.md`) has every task's evidence if someone wants the receipts.
