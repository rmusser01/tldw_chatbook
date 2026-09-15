---
id: TASK-32639
title: >-
  Console rail: agent-progress timer kills the app when its button is absent
status: To Do
assignee: []
created_date: '2026-09-15 10:30'
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
- [x] #1 A progress tick that finds no button updates nothing and raises nothing; the app survives.
- [x] #2 The rest of the callback (the counts comparison and the navigation refresh) still runs when the button is absent — the guard covers the label write only.
- [x] #3 Pinned with the button removed from a mounted rail, with the pin's RED captured by restoring the unguarded `query_one` (reproduces CI's exact `NoMatches` text).
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One-line fix: the label write became a loop over `self.query(...).results(Button)`,
which is empty rather than raising when the button is gone. No try/except, so
a genuine query error elsewhere still surfaces.

Modified: `tldw_chatbook/UI/Console_Modules/left_rail.py`,
`Tests/UI/test_console_rail_progress_timer.py` (new).
<!-- SECTION:NOTES:END -->
