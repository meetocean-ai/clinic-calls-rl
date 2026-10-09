# Morning glance — one screen

Read once with coffee. Everything else in this repo exists so you don't have to hold it in your head.

## The one-liner

I built a voice AI that answers a clinic's phone, then turned it into an RL environment and a benchmark: every call
runs against a fresh FHIR tenant, the reward is read from the clinic's records after the call — never from what the
agent said — and anyone can grade their own agent with the shipped verifier.

## Five things I know cold

1. **State is the judge.** Binary reward from the end state; the transcript can only *lose* points (PHI, medical
   advice, visit described before verification). Judges are logged, not rewarded.
2. **Bracket first.** Oracle must be 1.0 on every knob, random ~0 — or the task is wrong, not the model. It was:
   60/60 and 0/60 at both hard tiers.
3. **Voice is measured as a gap.** Same policy, text vs heard-through-Whisper, same tasks. The oracle's gap is the ear's
   own loss: 0.000 at tier 3, +0.050 on the bad phone line.
4. **Version everything, interval everything.** Dataset and verifier hashes on every episode; Wilson 95% on every rate;
   ten tasks per family is honest only with the interval shown.
5. **Free first, then a ledger line.** Everything so far cost $0 of GPU; each paid step has a cap written before it starts.

## Three numbers

160 held-out tasks · 95% of caller audio intelligible / 95% of facts heard · ~425 live-FHIR environment tests, zero mocks · untrained Qwen3-8B: 0.17 text, 0.20 voice (only emergencies pass) — the bar a trained model must clear · the production agents: 0.94 (tier 3) and 0.98 (tier 4) in text, and every miss became a numbered bug (six found, six fixed).

## Three honest lessons

Tokens expire at an hour and killed two long runs · editing the verifier mid-run relabelled half the episodes · a
reviewer found the leaderboard counting a *missing* write as a *wrong* write. All fixed; all written down.

## If a question lands

- "How would you build…" → `system-design-notes.md` (voice agent §1, RL env §2, benchmark §3, HIPAA §4).
- "What did you use…" → `tools-and-frameworks.md` (one lesson per tool).
- "What have you run in production…" → `experience-sheet.md`, `voice-agent-stack.md`.
- A term → `glossary.md`.

It is fine to say "let me think" and it is fine to say "I'd look that up". The work is real and it is here.
