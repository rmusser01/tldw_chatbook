---
id: TASK-32639
title: 'Console rail: agent-progress timer kills the app when its button is absent'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 10:30'
updated_date: '2026-09-29 19:28'
labels:
  - console
  - rail
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleLeftRail._sync_progress_count` runs on a 0.5s interval armed in
`on_mount`, and called `query_one("#console-agent-progress")` with no guard.
That button is composed only while `_open_agent_progress` is set, so a tick
landing while the agent section is between recomposes raises `NoMatches` out
of the timer callback. Textual re-raises a timer exception at the app, so the
app exits — and under `run_test` the exception surfaces as the test failing
at `async with app.run_test(...)`.

Found in CI, not by reading: `Tests/Performance/test_ui_latency_guardrails.py::test_library_opens_within_budget_on_a_seeded_profile`
failed with exactly that `NoMatches`, having done nothing but be slow enough
to land in the window. It is a race, so it reads as a flake until you look at
the callback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A progress tick with an absent button skips only the label write and raises no NoMatches; the app survives
- [x] #2 The counts comparison and navigation refresh still run while the label is absent
- [x] #3 A mounted removal regression and historical unguarded negative control reproduce and pin the original NoMatches defect
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the mount, timer, unmount and navigation paths; reproduce the missing-label gap in a private-profile mounted regression.
2. Observe RED for changed counts/navigation while the button is absent, then restrict the absence guard to the label write.
3. Verify startup label, descendant replacement, changed navigation and timer retirement with targeted checks; leave independent review and criteria signoff pending.
ADR required: no
ADR path: backlog/decisions/136-scoped-child-progress-and-supervisor-relay.md; backlog/decisions/150-design-token-system-and-design-language.md
Reason: routine timer race fix within existing progress ownership and UI boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The progress tick now reads its snapshot and compares navigation counts even when the label is absent; only the label write uses an empty-query loop. No broad exception handling was added. The existing on_unmount timer stop remains unchanged.

Tests/UI/test_console_rail_progress_timer.py now uses the existing private_profile_test harness. Its mounted callback verifies initial Progress: 3 queued, changed cached count and visible navigation at 5 with the button removed, Progress: 5 queued after recompose through a real timer interval, and no navigation change across a timer interval after rail removal.

RED on the incoming guard: the cached count remained {"current": 3} instead of {"current": 5}. Captured under /private/tmp/tldw-progress-timer-red. Historical unguarded-query negative control: NoMatches: No nodes match '#console-agent-progress' on ConsoleLeftRail, propagated out of run_test; /private/tmp/tldw-progress-timer-unguarded-red.
GREEN: main checkout .venv Python 3.12, python -m pytest Tests/UI/test_console_rail_progress_timer.py Tests/Backup_Recovery/test_console_progress_timer.py -q --basetemp=/private/tmp/tldw-progress-timer-green: 2 passed in 18.87s.
Supporting local warm-up: Tests/Performance/test_ui_latency_guardrails.py::test_library_opens_within_budget_on_a_seeded_profile, /private/tmp/tldw-progress-timer-library-green: 1 passed in 49.48s.

Test lint and format checks pass; source fatal-rule check passes. Full source lint retains the same seven findings as HEAD (I001, two UP037, RUF012, three BLE001); source format retains the same three regions as HEAD. The first sandbox RED had pytest-cache write warnings; later escalated runs were clean.
ADR required: no; existing ADR-136 and ADR-150 apply. Modified production/test files: left_rail.py and test_console_rail_progress_timer.py. TASK-32517 records the related historical hypothesis. Criteria remain pending independent review; no historical CI qualification is claimed.

Final disposition 2026-09-29: independent read-only implementation review approved the scoped repair with no actionable findings. The targeted acceptance checks and changed-line static checks recorded above pass; inherited whole-file lint/format debt remains outside this correctness task. All acceptance criteria are checked and this task is Done. No full-suite or live-provider qualification is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
