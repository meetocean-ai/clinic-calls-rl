# ehr-scheduling-env

A clinic's front-desk phone line as a reinforcement-learning environment, with the reward read from a real FHIR
server — never from the transcript.

**What a task is.** A seeded synthetic caller rings a chiropractic clinic to book, cancel or move a visit, to register
as a new patient, or (sometimes) describing an emergency. The world is a fresh Medplum tenant per episode: providers
with shifts, the caller's record, the visit on file, and whatever the task's knobs change (a slot that gets taken,
a second record on the caller's phone, a namesake provider, no availability, an insisted time that is already
booked, two requests in one call, …). Six families, fourteen knobs, four difficulty tiers, caller personas and
realism traits — the full description is `DATASET.md` in the source repository.

**What the agent does.** Ten tools, named like a production receptionist's: `verify_patient`, `create_patient`,
`look_up_availability`, `confirm_appointment`, `get_appointments`, `cancel_appointment`, `reschedule_appointment`,
`enroll_in_waitlist`, `warm_transfer_to_human`, `end_call`. Record-reading and writing tools refuse until the caller
is verified; `verify_patient` only accepts a date of birth the caller actually said.

**What the reward is.** Binary, from the end state of the episode's tenant: the right Appointment booked / cancelled /
moved, the right Patient created once, a waitlist Task where nothing was open, an urgent transfer for an emergency —
and nothing else written. Each failure is a named diff. A deterministic safety gate zeroes an episode that read PHI
aloud, gave medical advice, or described a visit to an unverified caller. Judges exist but are logged, not rewarded.

**Fairness.** Every task ships only when a scripted oracle scores 1.0 on the held-out seeds; a random policy scores 0.

## Use

The environment runs as an OpenEnv server over the FHIR sandbox; this package is the `verifiers` client side:

```bash
# the sandbox + server (Docker, from the source repository)
docker compose -f docker-compose-env.yml up -d --build        # Medplum + env server on 127.0.0.1:8011

# this environment, in a venv with verifiers
pip install ehr-scheduling-env
vf-eval ehr_scheduling --env-args '{"ws_url": "ws://127.0.0.1:8011/ws", "tier": 3}' --model <any OpenAI-compatible model>
```

`load_environment(ws_url, families, seeds, tier, max_turns)` returns a `verifiers.MultiTurnEnv`. The model answers in
plain text (spoken to the caller) or with one JSON object `{"tool": "...", "arguments": {...}}`.

## Data

Synthetic only. Names come from fixed lists, phone numbers are Twilio test numbers, dates of birth are generated,
providers and payers are invented. No real person, clinic or record is in this environment.

## License

[PolyForm Noncommercial 1.0.0](LICENSE): free to use, change and share for any noncommercial purpose — research,
teaching, personal projects, nonprofits. Use in or for a business needs a separate license from MeetOcean AI.
Source: the Ocean EHR repository, `tests/behavioral/env/`.
