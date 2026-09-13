---
id: TASK-32517
title: "Console: '#console-agent-progress' is queried before it mounts during startup"
status: To Do
assignee: []
created_date: '2026-09-13 00:13'
labels:
  - console
  - rider
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During CI's warm-up for
`test_library_opens_within_budget_on_a_seeded_profile`, the Console left
rail's `_sync_progress_count` (`tldw_chatbook/UI/Console_Modules/left_rail.py`)
raised a Textual `NoMatches` for `#console-agent-progress`. Cause INFERRED,
not proven: either the 0.5 s count sync fires before the progress widget has
mounted on a cold start, or it keeps firing after the widget is torn down —
`left_rail.py` guards both the compose and the sync with the same
`_open_agent_progress` flag, so a query after unmount is at least as likely
as one before mount; the fix has to establish which. The test recovered on
its own, so today this is a startup-log error rather than a failure, but a
sync that queries a widget that is not there is an ordering bug and will
surface as a hang or a crash the day the handler stops swallowing it.

Evidence: the non-required "UI latency" CI job of PR #2654's run (wave-3
backlinks-table, 2026-09-13), red on
`test_library_opens_within_budget_on_a_seeded_profile` with this NoMatches
during the test's Chat warm-up — not a budget overrun (branch 7.15 s vs dev
8.90 s, both under 10 s). Recorded by the T11 landing pass; carried here by
the wave-3 docs sweep.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A cold start never logs or raises NoMatches for '#console-agent-progress' — the progress count is synced only once the widget is mounted
- [ ] #2 The count shown after startup matches the number of active agent runs, with a test that starts the app cold and reads it back
- [ ] #3 The seeded-profile Library open-budget test's warm-up runs clean on CI (no NoMatches in its log)
<!-- AC:END -->
