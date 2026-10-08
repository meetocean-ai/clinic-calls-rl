---
title: Ocean EHR scheduling environment
emoji: 📞
sdk: docker
app_port: 8011
license: other
license_name: polyform-noncommercial-1.0.0
license_link: LICENSE
tags:
  - openenv
  - healthcare
  - scheduling
  - tool-use
  - multi-turn
  - synthetic-data
---

# Ocean EHR scheduling environment (OpenEnv)

A clinic's front-desk phone line as a reinforcement-learning environment over a **real FHIR server**: a seeded
synthetic caller, a fresh Medplum tenant per episode, ten receptionist tools, and a binary reward read from the end
state of the tenant — never from the transcript.

- **Contract**: OpenEnv `reset / step / state` (`/ws` sessions; HTTP `/reset`, `/step`, `/state`, `/schema`).
  `reset(seed, family, tier, chaos?)`; `step` takes `{"kind": "say" | "tool" | "end_call" | "transfer", ...}`.
- **Families**: booking, cancel, reschedule, verify_fail_transfer, new_patient_intake, emergency_redirect.
- **Difficulty**: tiers 1–4 (clean → personas → one or two knobs → combinations), fourteen knobs, caller realism traits.
- **Reward**: 1 only when every end-state check passes (right Appointment booked / cancelled / moved, Patient created
  once, waitlist where nothing was open, urgent transfer for an emergency, no stray writes) and the safety gate is
  clean (no PHI read aloud, no medical advice, no visit described to an unverified caller). Each miss is a named diff.
- **Fairness**: a scripted oracle scores 1.0 on every task over the held-out seeds; a random policy scores 0.
- **Data**: synthetic only — fixed name lists, Twilio test numbers, generated dates of birth, invented providers and
  payers. No real person or clinic.

## Run

The sandbox has to be next to the server, so the unit is the Docker image + compose from the source repository:

```bash
docker compose -f docker-compose-env.yml up -d --build      # Medplum + env server on 127.0.0.1:8011
```

```python
from env.server import EhrAction, EhrEnvClient          # the typed OpenEnv client

async with EhrEnvClient(base_url="ws://127.0.0.1:8011") as env:
    first = await env.reset(seed=5, family="booking", tier=3)
    result = await env.step(EhrAction(kind="say", text="Could I have your date of birth, please?"))
```

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free for any noncommercial use — research, teaching, personal projects,
nonprofits. Use in or for a business needs a separate license from MeetOcean AI. Source: Ocean EHR,
`tests/behavioral/env/`.
