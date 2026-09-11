---
id: TASK-32260
title: >-
  Library opens in 12.6 s on a seeded profile of 27 items versus 2.7 s on a
  fresh one
status: Done
assignee:
  - '@claude'
created_date: '2026-09-10 18:05'
updated_date: '2026-09-11 15:36'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Measured by the evidence assessor across the two profiles used for this review: a fresh profile opens Library in 2.7 s; the seeded profile (10 notes, 11 media, 6 conversations) takes 12.6 s. Twenty-seven items is not a data-volume effect at any plausible per-item cost, so something on the seeded path is doing per-open work the empty path skips. Cause untraced.

Filed because a 12.6 s open on a trivially small corpus predicts a considerably worse number on a real one, and because everything else measured on this screen was fast (35 KB note in 0.7 s, 179-path vault tree in 0.39 s, 59-file import under 2 s to review).

Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-10T17-32-16Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev e6cb464239, 2026-09-10, dual-assessor live review at 235x52 and 100x30/60x24 on two fresh profiles and one seeded profile with a git-backed 71-file Obsidian vault fixture; the reconciling parent re-tested every disagreement on a third pair of profiles and traced each surviving cause to code). Captures under the session scratchpad `notes-crit2/{C,D,R}/caps/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause of the seeded-versus-fresh open-time gap is identified and named
- [x] #2 Library opens on a 27-item profile within a documented budget
- [x] #3 A regression test or a recorded measurement pins the open time
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rebuild the critique's profile with seed_power_profile.py (10 notes, 11 media, 6 conversations).
2. Measure the open live at 235x52 on this branch AND at the critique's own base commit e6cb464239, plus a fresh profile.
3. Profile the seeded open in-process (cProfile) to name what the empty path skips.
4. Fix the cause if the measurements find one; otherwise record the numbers and pin a budget test on a seeded fixture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Profiled before optimising, and there is nothing to optimise.** The
critique's profile was rebuilt with the same seed script (10 notes, 11
media, 6 conversations = 27 items) and the open was measured on an
M-series Mac:

| measurement | fresh | seeded |
| --- | --- | --- |
| live, this branch (dev 4a14b3f36f), shell painted | 0.67 s | 0.68–0.79 s |
| live, this branch, counts fully resolved | n/a (no counts) | 1.01 s |
| live, a pristine never-opened seeded profile | — | 1.01 s |
| live, the critique's OWN base e6cb464239 | — | 1.24 s |
| in-process (`run_test`, ctrl+3 to arrival) | 0.90 s | 1.09 s, 1.15 s |

So the gap is ~0.2 s, not 9.9 s, and it is the same at the exact commit
the critique measured on. The 12.6 s / 2.7 s pair is not reproducible
from the code; the likeliest explanation is the measuring session itself
(several app instances at once, and a cold import cache — the crit-base
boot measured 9.1 s cold here against 4.1 s warm).

**AC#1, the gap named.** cProfile over the seeded open, cumulative:
one private-SQLite helper connection per database (12 connects, 1.02 s;
12 closes, 1.15 s), ChaChaNotes connection setup (27 calls, 0.94 s), and
skill trust-store manifest validation (8 calls, 0.87 s, for the profile's
2 skills). All of it is overlapped worker work proportional to "this
profile has content at all", not to the number of items — which is why
27 items cost 0.2 s and not 10 s. Nothing here is worth removing.

**AC#2/#3, the budget.** What was actually missing is a pin: no test
measured the open on a profile with content, which is how a 12x claim
stood for a week.
`Tests/Performance/test_ui_latency_guardrails.py::test_library_opens_within_budget_on_a_seeded_profile`
seeds 27 items, asserts the corpus really reached the screen (so it can
never quietly measure an empty profile), and holds the same generous 10 s
budget the empty destination tour uses — which the reported 12.6 s would
fail.

**Files.** `Tests/Performance/test_ui_latency_guardrails.py`.
