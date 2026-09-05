---
id: TASK-31661
title: >-
  Environment 10s poll silently resets rail focus when the file set changes
status: In Progress
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique P1, live-measured: with focus parked on a rail row, an external
file change makes the next 10s poll recompose the section and throw focus
to a widget above the section header (invisible focus, no indicator moves;
two Tabs to recover). Fires repeatedly during agent runs — the panel's core
workflow. The activation path already restores focus by row_id
(_request_console_environment_row_focus); the poll/sync path does not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A poll-driven recompose restores focus to the row with the same row_id when it still exists, else to the nearest surviving row in the same section
- [ ] #2 Focus never lands on a widget with no visible indication as a result of a background sync
- [ ] #3 Wiring test: park focus on a row, land a snapshot that changes the row set, assert focus location after the sync
<!-- AC:END -->
