# clinic-calls leaderboard

Generated 2026-10-08 from the recorded `rl-env-*` runs under `docs/quality/runs/` by `tests/quality/build.py` (every quality report regenerates it). How to read it: [METHODOLOGY.md](METHODOLOGY.md). Four components, never one score. Held-out tasks (`--split heldout`, dataset v2); `k` = trials per task; intervals are Wilson 95%.

## Policies

| policy | mode | tier | tasks | k | pass@1 | 95% CI | pass^k | wrong writes | safety gate | claimed w/o tool | latency p50 | ASR sim. | errors | latest run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `production` | text | 3 | 8 | 3 | **1.000** | 0.86–1.00 | 1.000 | 0.000 | 0.000 | – | – | – | 0 | 2026-10-06 |

## Voice gap

pass@1 in text mode minus pass@1 with the caller heard through the ASR cascade, same policy, same tasks (`lost` / `gained` = tasks whose mean changed in that direction).

| policy | tier | tasks | text pass@1 | 95% CI | voice pass@1 | 95% CI | voice gap | lost / gained |
|---|---|---|---|---|---|---|---|---|
| `oracle` | 3 | 60 | 1.000 | 0.94–1.00 | 1.000 | 0.94–1.00 | **+0.000** | 0 / 0 |
| `oracle` | 4 | 60 | 1.000 | 0.94–1.00 | 0.950 | 0.86–0.98 | **+0.050** | 3 / 0 |

## Graduation gate

A candidate replaces production only when, on the held-out set and in the same mode and tier, its pass@1 is at least production's with non-overlapping Wilson intervals, its wrong-write rate is 0 and its claimed-without-tool rate is 0 (roadmap P4-03). Rows come from shadow runs (`env.shadow`, suites `rl-env-candidate-*`).

No candidate rows recorded.

## Reference policies (fairness gate and floor, not contestants)

| policy | mode | tier | tasks | k | pass@1 | 95% CI | pass^k | wrong writes | safety gate | claimed w/o tool | latency p50 | ASR sim. | errors | latest run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `oracle` | text | 3 | 60 | 1 | **1.000** | 0.94–1.00 | 1.000 | 0.000 | 0.000 | 0.000 | – | – | 0 | 2026-10-08 |
| `oracle` | voice | 3 | 60 | 1 | **1.000** | 0.94–1.00 | 1.000 | 0.000 | 0.000 | 0.000 | – | 0.937 | 0 | 2026-10-08 |
| `oracle` | text | 4 | 60 | 1 | **1.000** | 0.94–1.00 | 1.000 | 0.000 | 0.000 | 0.000 | – | – | 0 | 2026-10-08 |
| `oracle` | voice | 4 | 60 | 1 | **0.950** | 0.86–0.98 | 0.950 | 0.000 | 0.000 | 0.000 | – | 0.867 | 0 | 2026-10-08 |

## Provenance

| policy | mode | tier | families | runs | dataset version | method version |
|---|---|---|---|---|---|---|
| `production` | text | 3 | booking, cancel, reschedule, verify_fail_transfer | `20261006_223549_rl-env-production-booking-cancel-reschedule`, `20261006_224324_rl-env-production-verify_fail_transfer`, `20261006_225100_rl-env-production-booking-reschedule`, `20261006_225431_rl-env-production-booking` | unstamped | unstamped |
| `oracle` | text | 3 | booking, cancel, emergency_redirect, new_patient_intake, reschedule, verify_fail_transfer | `20261008_014632_rl-env-oracle-booking-cancel-emergency_redirect-new_patient_intake-reschedule-verify_fail_transfer` | `tasks-v3+gen-c605a9eb4c` | `verify-b4440d9c68` |
| `oracle` | voice | 3 | booking, cancel, emergency_redirect, new_patient_intake, reschedule, verify_fail_transfer | `20261008_014632_rl-env-oracle+voice-booking-cancel-emergency_redirect-new_patient_intake-reschedule-verify_fail_transfer` | `tasks-v3+gen-c605a9eb4c` | `verify-b4440d9c68+audio` |
| `oracle` | text | 4 | booking, cancel, emergency_redirect, new_patient_intake, reschedule, verify_fail_transfer | `20261008_012634_rl-env-oracle-booking-cancel-emergency_redirect-new_patient_intake-reschedule-verify_fail_transfer-tier4` | `tasks-v3+gen-c605a9eb4c` | `verify-b4440d9c68` |
| `oracle` | voice | 4 | booking, cancel, emergency_redirect, new_patient_intake, reschedule, verify_fail_transfer | `20261008_012634_rl-env-oracle+voice-booking-cancel-emergency_redirect-new_patient_intake-reschedule-verify_fail_transfer-tier4` | `tasks-v3+gen-c605a9eb4c` | `verify-b4440d9c68+audio` |

A row whose method or dataset version differs from another row's is not comparable with it; the series keeps both. *unstamped* = recorded before the series carried version stamps (dataset v1, 2026-10-06): historical, not comparable with v2 rows.
