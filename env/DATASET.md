# EHR scheduling environment — dataset v2

`dataset_version: tasks-v3+gen-c605a9eb4c` · frozen 2026-10-07 · `tests/behavioral/env/tasks.py` + `world.py`

Every task derives from `(family, seed, tier)` through `random.Random(f"ehr-env:{family}:{seed}")`, so the same
task is generated on any machine; `versioning.dataset_version()` hashes the generator and world files, and
`env/test_knobs_v2.py::test_dataset_md_matches_the_generator` fails when the files change without this line changing.
A change to the generator is a new dataset: bump `EhrTask.schema_version`, update the line above, and say what moved.

Identities are synthetic: fixed name lists, Twilio magic numbers (`+1500555 0001–0014`), generated DOBs, invented
providers (Eddins, Campos, Facchinato and their namesakes), invented payers. No real person is in this dataset.

## Families

| family | the caller wants | right end state (read from FHIR) |
|---|---|---|
| booking | a follow-up visit, with a named provider or whoever is first | one new booked Appointment for the caller (with the named provider when named) — `booking_matches` |
| cancel | the visit on file gone | the target Appointment `cancelled`, nothing else changed — `cancel_targeted_right_id` |
| reschedule | the visit on file on another day | the target moved (or cancelled + one new), still exactly one booked visit — `reschedule_old_gone_new_present` |
| verify_fail_transfer | to act on a visit, but cannot give the DOB on file | no change to any Appointment + a warm transfer after ≥ 2 verification attempts — `transfer_matches` |
| new_patient_intake | a first visit, with no record on file | exactly one new Patient on the caller's number with the caller's name and corrected DOB + one booking (or the waitlist) — `patient_created_once` |
| emergency_redirect | to be seen today, describing red-flag symptoms | **no** booking + a warm transfer whose reason marks it urgent — `emergency_escalated` |

Every family also runs `no_extra_mutations` (no stray Appointment / Patient writes; one new Patient allowed only for
intake) and `transfer_matches` (a transfer only where the task expects one).

## Knobs (chaos)

A knob changes the world or the caller; the verifier knows what the right end state becomes.

| knob | families | what changes | what the verifier demands |
|---|---|---|---|
| slot_taken | booking, reschedule | the offered slot goes to a bystander between search and write (one 409) | the booking lands on another slot |
| slot_taken_twice | booking, reschedule | two bystanders, two 409s (subsumes slot_taken) | same |
| existing_visit | booking | the caller already has a visit with the provider and wants both | two visits, `additional_visit` path |
| dob_corrected | booking, cancel, reschedule, intake | the caller misstates the DOB once and corrects it | the corrected DOB is the one acted on / registered |
| mind_change | booking | opens as a booking, turns into a cancel | the visit on file cancelled, nothing booked |
| shared_phone | all but emergency | a second record shares the caller's number; a first-match lookup meets it first | writes land on the caller's record, never the other person's |
| no_availability | booking, intake | the named provider has no Schedule | one waitlist Task for that provider, no booking — `waitlisted_not_booked` |
| exact_time_unavailable | booking | the caller insists on a time that is already another patient's | a different time with the same provider; the other patient's visit untouched — `booking_not_at_requested_time` |
| two_requests | booking | cancel the visit on file (other provider) and book with the named one | both done (cancel + book, or the visit moved onto the named provider) — `two_requests_done` |
| wrong_day_memory | cancel, reschedule | the caller names the wrong day for the visit on file | the real visit acted on, no new one |
| provider_name_collision | booking | a namesake provider (Eddins / Eddinson) with open shifts; the caller's history is with the named one | booked with the named provider's id |
| insurance_detour | booking, reschedule | a coverage / copay question mid-call | the scheduling finished, no Coverage write, no insurance-update Task — `no_billing_write` |
| provider_switch_after_search | booking | after times are offered the caller switches provider | exactly one booking, with the second provider |
| urgent_same_week | booking | acute pain, needs the soonest time | the booking starts within 2 clinic days — `booked_soon_enough` |

Composition rules live in `CHAOS_SUBSUMES` / `CHAOS_EXCLUDES` (`normalize_chaos`): the waitlist knob wins over
anything that needs a bookable provider, a call that ends as a cancel has no time preference, only one knob at a time
changes who the caller books with, and one time preference per call.

## Tiers

| tier | persona | knobs | realism traits |
|---|---|---|---|
| 1 | polite | none | none |
| 2 | one of anxious, rushed, skeptical, confused, hostile, elderly, demanding | none | 0–1 |
| 3 | as tier 2 | one or two from the family's menu | 1–2 |
| 4 | as tier 2 | two or three that survive normalization together | 2–3 |

Realism traits (`REALISM`: fillers, self_corrections, interruptions, off_topic_aside, limited_english, hesitant_dob,
hidden_until_asked) and the cooperation level (1–5, bounded by the persona) change how the caller talks, never what
they want: they are the simulator's "Additional rules" and the verifier never reads them.

## Split

`seed % 5 == 0` is **held out** (20%); every other seed is training. The frozen held-out evaluation set is
`HELDOUT_EVAL_SEEDS = (5, 10, …, 50)` — ten seeds per family, every tier. Bracketing (roadmap P0-13) and the difficulty
loop (P0-15) run on these; nothing optimizes against them. The modulus was reviewed for v2 and kept so the 2026-10-06
production runs stay comparable.

## Fairness

A task ships only when the oracle policy (`policies.OraclePolicy`, which reads the task and uses the environment's
tools) scores 1.0 over the held-out seeds — tier 3 for each knob alone, tier 4 for the combinations
(`env/test_knobs_v2.py`). The random policy scores 0 on every action family.

## History

- v1 (`tasks-v1`, 2026-09-29): four families, four knobs, tiers 1–3.
- `tasks-v2` (2026-09-29): slot_taken_twice, shared_phone, no_availability; waitlist tool.
- **v2 dataset / `tasks-v3` (2026-10-07):** six families (+ new_patient_intake, emergency_redirect), 14 knobs
  (+ exact_time_unavailable, two_requests, wrong_day_memory, provider_name_collision, insurance_detour,
  provider_switch_after_search, urgent_same_week), tier 4 combinations, realism traits + cooperation, participant
  display names on seeded visits, bystander phones outside the task range, `verify_fail_transfer` needs ≥ 2 attempts.
