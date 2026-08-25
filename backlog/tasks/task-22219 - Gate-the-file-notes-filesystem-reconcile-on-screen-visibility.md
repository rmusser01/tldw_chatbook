---
id: TASK-22219
title: >-
  Gate the file-notes filesystem reconcile on screen visibility
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - library
  - notes
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22219).

Pre-existing. `Widgets/Library/library_file_notes_workspace.py:1453-1456` arms a 1.5 s
`set_interval` (`pause=False`) whose fire runs `to_thread(service.reconcile)` — a
walk/stat of the notes root — 40x/minute for the Library screen's lifetime once the File
Notes surface has been opened, including while other screens or modals are on top (the
only gates are `_active`/transitioning/in-flight; no `screen.is_active`).

## Acceptance Criteria

- [ ] No reconcile fires while the Library screen is not active or is covered (probe)
- [ ] Polling resumes on return to the screen; a change-driven or backoff cadence is considered and the choice stated
- [ ] Filesystem scan count per minute measured before/after in the covered state
