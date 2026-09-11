---
id: TASK-32330
title: >-
  Staged sources that deliver nothing are labeled honestly
status: To Do
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C1. The Sources tray renders 'ready' for staged handoffs that send nothing to the model: skills, watchlists/collections snapshots, quizzes, personas (task-2375), and media/conversation handoffs currently deliver only a short label (task-2376). The docs disclose this; the UI does not. Until delivery lands, label those rows so the user can see the model will not receive the content.

Filed from the 2026-09-10 Console rail UX review (review item C1).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Staged rows for handoff kinds that currently deliver no model payload render a distinct not-delivered status (e.g. 'listed - not sent') instead of 'ready'
- [ ] #2 Rows whose delivery is partial (short label only) render an honest distinct status
- [ ] #3 Status chip / Source Readiness counts remain consistent with the tray (no disagreement between surfaces)
- [ ] #4 User-guide context-and-rag.md updated to match the new row vocabulary
<!-- AC:END -->
