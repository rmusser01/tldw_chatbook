---
id: TASK-31506
title: Session switcher modal recomputes the full projection at 5 Hz
status: To Do
assignee: []
created_date: '2026-09-04 19:30'
labels:
  - performance
  - console
dependencies: []
priority: low
---

## Description (the why)

`Widgets/Console/console_session_switcher_modal.py:55,253` polls the active
projection every 200 ms while the modal is open. Each tick calls
`console_session_switcher_active_entries()`
(`UI/Console_Modules/workspace.py:2719`), which iterates `store.sessions()`
calling `controller.activity_for` per session, then iterates the produced
rows a second time calling `activity_for` AND `run_state_for` per session
again, then builds the results object -- all unconditionally, purely to
compute a change fingerprint. O(open sessions) x 5/s; fleet users with many
sessions pay most. Evidence:
`Docs/Design/2026-09-04-holistic-perf-review.md` section 7.

## Acceptance Criteria (the what)

- [ ] The per-tick steady-state work no longer performs duplicate `activity_for` lookups per session (single pass, or a cheap fingerprint that avoids the full projection build when nothing changed)
- [ ] Modal responsiveness to real activity changes is preserved (existing switcher tests stay green)
