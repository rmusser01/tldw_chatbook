---
id: TASK-22866
title: Remediate Watchlists bulk authoring and Artifacts workflow UX
status: To Do
assignee: []
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 04:17'
labels:
  - watchlists
  - ux
  - textual
  - briefings
dependencies:
  - TASK-22862
  - TASK-22863
  - TASK-22864
  - TASK-22865
references:
  - Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - Docs/superpowers/plans/2026-08-27-watchlists-feed-and-interface-uat-remediation.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make direct Watchlists inspection and recovery support bulk source entry, multi-source membership, legible operational states, and reliable briefing visibility at supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sources provides multiline/bulk entry with persistent labels, row-level validation/duplicate feedback, draft preservation, and an explicit partial-result decision.
- [ ] #2 Users can keyboard-select multiple sources, understand filtered/select-all/range semantics, and create a Watchlist from the selected set without repeated membership dialogs.
- [ ] #3 Implemented shortcuts are discoverable in valid footer/command-palette hints; primary meanings never depend on tooltips or color alone.
- [ ] #4 Artifacts foregrounds Generate/Schedule when empty and moves downstream actions into selected-briefing context or a labeled disclosure.
- [ ] #5 The collection automation receipt shows interval, app-open limitation, next eligibility, last attempt/success, reload state, and attention/recovery state.
- [ ] #6 Briefing refresh/generation retains the last good table/body, shows inline progress, preserves content on failure with Retry, and provides a recoverable reload diagnostic when durable storage and the pane disagree.
- [ ] #7 Production-shaped Textual tests cover first-time and power-user paths at the supported 160x42 pressure point and a normal size, including focus order, Escape, draft preservation, stale/error states, and receipt deep-links.
<!-- AC:END -->
