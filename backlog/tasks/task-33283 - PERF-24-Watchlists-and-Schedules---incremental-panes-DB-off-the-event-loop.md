---
id: TASK-33283
title: 'PERF-24: Watchlists and Schedules - incremental panes, DB off the event loop'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33268
labels:
- performance
- watchlists
- scheduling
- ui
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
documentation:
- Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md
updated_date: 2026-10-25 23:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Schedules:
- moving the cursor onto a reminder runs 3 synchronous ScheduledTasksDB reads on the loop (132 ms)
- each visit runs about 145 ms in on_mount before first paint, plus about 385 ms per visit
- SchedulingService and SyncEngine hide synchronous sqlite behind async def
- mark-all-read is N+1

Watchlists:
- the left-rail tree is remounted on every count refresh
- inspector and content panes recompose on every selection, including j/k navigation Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-24; every issue with file:line is listed under PERF-24 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).

### Post-PR #3247 book audit: Sources create editing context

Additional bounded evidence for the existing incremental-pane objective (AC #4), rather than a duplicate task: on merged dev `79da83a8dca98eab6ca854a7505b8abd264eb959`, assigning a changed Sources list to the mounted production SourcesPane replaces a settled create-form Name or URL Input. Draft text survives, but a real keyboard selection of characters 2-4 becomes select-all (0-9); the next X replaces AlphaFeed with X instead of AlXaFeed. The busy-source update control preserves the same widget, selection and expected edit. Both fields were checked with production CSS and actual Pilot keys. The screen _load_sources -> _seed_sources_pane path publishes the same sources reactive; the controlled pane probe does not claim a full-screen background-refresh journey, natural frequency or pointer behavior.

Preserve the active create-form value, focus, caret and selection while implementing in-place Sources refresh, alongside the existing tree/inspector/content work. Related Done owners [TASK-15460](./task-15460%20-%20Watchlists---replace-per-keystroke-pane-teardowns-with-in-place-updates.md) (typing filters), [TASK-3071](./task-3071%20-%20SourcesPane-search-loses-focus-and-query-on-recompose.md) (search focus), [TASK-1035](./task-1035%20-%20Create-source-form-cannot-be-filled-in-through-the-UI.md)/[TASK-1345](./task-1345%20-%20Select-Input-mount-race-makes-the-Watchlists-create-form-tests-order-dependent.md) (form-open focus) and [TASK-35868](./task-35868%20-%20Preserve-unsaved-Watchlists-rule-fields-across-refresh-and-section-return.md) (Rules drafts) cover different triggers/fields. Existing [ADR-011](../decisions/011-chatbook-workbench-ui-system.md) and [ADR-042](../decisions/042-watchlists-reader-first-ia.md) apply; no new ADR is required for this bounded correction. Source: tldw_chatbook/UI/Watchlists_Modules/sources_pane.py:153,690-733,1311-1477. Book: Chapter 9, PDF176-180 / printed167-171. [Audit and reproducible observations](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3247.md).

### Post-PR #3248 book audit: Notifications keyboard selection

Additional evidence for existing AC #4, already enumerated in PERF-24's original appendix (NotificationsPane table/toolbar recomposition, line 745): a real mounted Notifications table loses focus when the first Down selects row 1, because the table is replaced. A second Down leaves that row selected. Explicitly refocusing the live table then Down reaches row 2. The destination, production CSS, actual Pilot keys and a private file-backed ClientNotificationsDB are exercised. This strengthens the existing pane-update owner rather than creating a duplicate task. Keep the previously recorded Sources create-field editing-context witness here as well.

Related completed [TASK-15779](./task-15779%20-%20Watchlists-artifacts-pane-briefing-selection-destroys-the-briefings-table-and-its-focus.md) preserves Artifacts briefing selection, a sibling surface rather than this inbox table. Canonical [ADR-011](../decisions/011-chatbook-workbench-ui-system.md), [ADR-033](../decisions/033-application-session-state-ownership.md) and [ADR-042](../decisions/042-watchlists-reader-first-ia.md) apply. Book: Chapter 9 PDF179-180 / printed170-171. Audited merged dev: `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05`. [Audit and observations](../../Docs/superpowers/qa/2026-10-25-textual-book-after-pr3248.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Schedules interaction performs synchronous sqlite on the event loop
- [ ] #2 SchedulingService async methods do their DB work off the loop
- [ ] #3 Mark-all-read runs as a batched write
- [ ] #4 Watchlists tree and panes update in place on count refresh and selection changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
