---
id: TASK-22864
title: Make every-24-hours briefing schedules immediately observable
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
labels:
  - watchlists
  - scheduling
  - console
  - briefings
dependencies:
  - TASK-22863
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
  - backlog/decisions/019-watchlist-scheduler-migration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Store interval-based briefing cadence, wake the running scheduler, and return an honest receipt that distinguishes reload request from acknowledgement and global disablement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Console-only schedule command supports `every_12_hours`, `every_24_hours`, `every_7_days`, `off`, and advanced 3,600–2,678,400-second intervals; “Every 24 hours” stores exactly 86,400 seconds.
- [ ] #2 A never-attempted enabled schedule is immediately eligible; later eligibility is the latest attempt plus interval, including after failure, and overdue work runs when the app/scheduler resumes.
- [ ] #3 `off` clears only cadence and preserves prior briefings, preset, and selection mode; optional supplied preset/selection changes validate and commit with cadence atomically, while omitted values reuse the collection's stored briefing preset and existing app/provider defaults rather than the current Console conversation model.
- [ ] #4 A successful write requests immediate scheduler reload and returns stored cadence, gate/running state, next eligibility, last attempt/success, and separate reload-requested versus reload-acknowledged values.
- [ ] #5 A new bounded acknowledgement token completes only after the running loop reloads its queue; stopped, timed-out, disabled, or failed callbacks never report acknowledgement.
- [ ] #6 Artifacts and Settings show stored-but-inactive schedules honestly and use “Every 24 hours,” UTC storage/comparison, local display time, and app-open limitation copy.
- [ ] #7 Projection, loop, command, persistence-failure, timezone, gate, restart, and UI receipt tests pin the interval contract.
<!-- AC:END -->
