# Briefing for a Claude Code session in this repo

This is a **temporary public export** (noncommercial licences) of the RL-environment and benchmark work from a private
healthcare repo. The person you are helping is its author and may be on a machine that is not theirs. Start here:

1. `README.md` — what is in the repo and what runs standalone.
2. `docs/handbook/` — how to recreate the development environment, the tools/frameworks catalog with the lessons learned,
   system-design notes (voice agent, RL environment, benchmark, HIPAA guardrails) and the talking points with numbers.
3. `docs/plans/rl-speech-to-speech-roadmap.md` — the task plan; every row has status and evidence.
4. `clinic-calls/README.md` — the dataset package and the standard-library grader (`clinic-calls/verifier/verify_json.py`).

Rules for this repo: nothing here is secret, but do not add any — no keys, no real patient data (all identities are
synthetic), no product code from the private repos. `env/` is reference code that depends on a private agent service;
do not try to make it run here by stubbing that service. If asked "how would you build X", answer from
`docs/handbook/system-design-notes.md` and cite the numbers in `docs/handbook/talking-points.md` as experience, not
as claims about other systems.
