---
id: TASK-15779
title: 'Watchlists artifacts pane: briefing selection destroys the briefings table and its keyboard focus'
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - watchlists
  - ux
priority: medium
---

## Description

Pre-existing UX defect found and recorded ("worth its own task") in
task-15461's Implementation Notes (input-latency burn-down's Watchlists
scoped-rebuild work). Task-15461 reduced the artifacts pane's recompose count
on a briefing selection from 2 down to 1, but that remaining recompose still
rebuilds the pane wholesale: selecting a briefing recomposes `ArtifactsPane`,
which destroys and rebuilds the briefings `DataTable`, which loses keyboard
focus. A second arrow-key press then does nothing until the user manually
re-focuses the table — measured on the pre-task-15461 code too, so this is
not something task-15461 introduced, just something it exposed by getting
the recompose count down to a level where the remaining one is now the
visibly broken step.

## Acceptance Criteria

- [ ] Selecting a briefing in the artifacts pane does not destroy the
      briefings `DataTable` widget instance (in-place update instead of a
      recompose that tears it down), OR the recompose explicitly restores
      focus and cursor position to the table afterward
- [ ] A second arrow-key press immediately after selecting a briefing moves
      the cursor to the next row (regression test — this is the concrete
      symptom: today it does nothing until the user re-focuses)
- [ ] The scripts/audio/citations clearing task-15461 folded into
      `watch_selected_briefing` (via `set_reactive`) keeps working unchanged
- [ ] `Tests/Watchlists/test_watchlists_artifacts_pane.py` stays green
