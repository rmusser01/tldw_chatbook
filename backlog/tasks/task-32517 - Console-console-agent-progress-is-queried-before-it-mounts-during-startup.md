---
id: TASK-32517
title: "Console: '#console-agent-progress' is queried before it mounts during startup"
status: To Do
assignee: []
created_date: '2026-09-13 00:13'
labels:
  - library
  - notes
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During CI's warm-up for
`test_library_opens_within_budget_on_a_seeded_profile`, the Console left
rail's `_sync_progress_count` (`tldw_chatbook/UI/Console_Modules/left_rail.py`)
raised a Textual `NoMatches` for `#console-agent-progress`: the count sync
runs before the progress widget has mounted, so the first query on a cold
start finds nothing. The test recovered on its own, so today this is a
startup-log error rather than a failure, but a sync that runs against a
not-yet-mounted widget is a real ordering bug and will surface as a hang or
a crash the day the handler stops swallowing it.

Evidence: CI log for the wave-3 layout PR run of
`test_library_opens_within_budget_on_a_seeded_profile` (task-32260),
recorded in the wave-3 docs sweep, 2026-09-12.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cold start never logs or raises NoMatches for '#console-agent-progress' — the progress count is synced only once the widget is mounted
- [ ] #2 The count shown after startup matches the number of active agent runs, with a test that starts the app cold and reads it back
- [ ] #3 The seeded-profile Library open-budget test's warm-up runs clean on CI (no NoMatches in its log)
<!-- AC:END -->
