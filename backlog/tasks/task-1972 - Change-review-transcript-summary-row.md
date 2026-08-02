---
id: TASK-1972
title: 'Change review: per-turn transcript summary row + inspector action'
status: To Do
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - console
  - change-review
  - ux
dependencies:
  - TASK-1971
  - TASK-1973
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Display surfaces for a turn's changes: a transcript row in the TOOL-marker display-only family (`✎ Edited 3 files  +92 −468 — review with `v``, markup=False), a `review` selected-row action, and a 'Review changes' row in the run inspector's actionable group. Deliberately NO destructive control here: Undo-all lives on the Review screen behind a confirm — a one-keystroke destructive action in the transcript would repeat the approval-card mistake TASK-1845 fixed. Honesty degradations surface here too: 'change tracking failed (reason)', 'N nested repositories not tracked'.

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A turn with changes renders the summary row with real counts; a turn without renders nothing
- [ ] #2 The row survives subsequent messages and session switch/resume (TOOL-marker anchoring rules)
- [ ] #3 `v` on the selected row and the inspector action both open the Review screen for THAT turn
- [ ] #4 tracking_error and nested-repo warnings render on the row/inspector, in monochrome-legible text
- [ ] #5 No control in the transcript mutates files
<!-- AC:END -->
