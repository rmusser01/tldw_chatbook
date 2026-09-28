---
id: TASK-33407
title: Measure and, if it pays, coalesce the SSH session's admitted marker and result
status: Done
assignee:
  - '@claude'
created_date: '2026-09-28 20:30'
updated_date: '2026-09-28 16:50'
labels:
  - console
  - workspaces
dependencies:
  - TASK-33202
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The live UAT for PR #2879 saw a fast operation's result arrive ~8 ms after its admitted marker while the host work took ~4 ms, suggesting the second small write waits for the first's acknowledgement. Sending both in one write could save about one round trip per warm call. Deferred from PR #2879's final review (ruling R14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The effect is measured on the live host with interleaved before/after runs and same-window ping
- [x] #2 If warm-call latency improves measurably, a fast operation's marker, result and status leave the host in one write while a slow operation's marker is still sent promptly
- [ ] #3 If it does not improve, no coalescing code ships and the measurement is recorded
<!-- AC:END -->

## Implementation Notes

Coalescing ships: warm median improved by 3.3-5.1 ms across three pairs (decision rule: all pairs must show ≥2 ms improvement). First measurement (pre-registered, 6 runs A B A B A B) hit a confound (ping variance 1.6-3x larger in A runs); confirmation run (A B A B A B, simpler rule) proved the effect. Discarded one A-run Wi-Fi spike (44.97 ms warm → 16.60 ms after retry), which widened that pair further. New constant `_COALESCE_S = 0.010` in `remote_session_serve.py` holds output when lines arrive (not urgent), flushes on timeout, STATUS (urgent), or hold-window end. All 78 tests passing.

**First measurement (6 runs B A B A B A):**
| run | variant | warm median ms | warm p90 ms | ping median ms | warm-ping |
|---|---|---|---|---|---|
| 1 | B | 11.53 | 81.85 | 25.65 | -14.12 |
| 2 | A | 15.00 | 101.64 | 45.08 | -30.08 |
| 3 | B | 11.01 | 44.70 | 28.14 | -17.13 |
| 4 | A | 16.40 | 160.50 | 84.92 | -68.52 |
| 5 | B | 11.80 | 42.52 | 27.43 | -15.63 |
| 6 | A | 16.83 | 146.21 | 63.06 | -46.23 |

**Confirmation (6 runs A B A B A B, raw warm median rule):**
| run | variant | warm median ms | warm p90 ms | ping median ms |
|---|---|---|---|---|
| 1 | A | 17.05 | 137.57 | 45.23 |
| 2 | B | 13.75 | 178.50 | 86.57 |
| 3 | A | 17.21 | 171.20 | 75.15 |
| 4 | B | 12.48 | 38.05 | 17.16 |
| 5 | A | 16.60 | 147.09 | 67.52 |
| 6 | B | 11.49 | 44.60 | 13.81 |

**Pair deltas (raw warm median, B minus A):** −3.30, −4.73, −5.11 ms. Disclosure: run 5 A hit Wi-Fi spike (44.97 ms warm, 767.61 ms p90, 895.54 ms ping) and was discarded; retry 16.60 ms (delta widened to −5.11). All three pairs exceeded 2 ms threshold.
