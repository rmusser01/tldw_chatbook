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

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PREMISE MOSTLY GONE on dev: task-2375/2376 handoff kinds (skills/watchlists/quizzes/personas) can no longer be staged; staging funnels through Library RAG evidence (library_rag_state.py:53-59); send delivers only available_references() (citation_evidence_models.py:367-376). RESIDUAL TO CHECK: rows whose reference status is neither available nor blocked/missing render 'Warning' in the tray (console_display_state.py:826-834) - do they deliver? And the generic live-work fallback item (console_live_work.py:169-196) stages a navigation-only item - does it honestly render?
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
