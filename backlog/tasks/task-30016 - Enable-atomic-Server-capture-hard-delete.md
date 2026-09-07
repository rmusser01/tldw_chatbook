---
id: TASK-30016
title: Enable atomic Server capture hard delete
status: To Do
assignee: []
created_date: '2026-09-03 02:56'
updated_date: '2026-09-03 02:58'
labels:
  - library
  - collections
  - reading-list
  - deletion
  - server-parity
dependencies:
  - TASK-18919
references:
  - TASK-18919
  - backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
  - Docs/superpowers/specs/2026-09-01-collections-followup-backlog-design.md
  - 'tldw_server:TASK-13153'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reference S1 as `tldw_server:TASK-13153` rather than using a same-repository Backlog dependency or a pre-merge `blob/dev` URL. Chatbook keeps Server hard delete visibly unavailable until exact `hasReadingOptimisticDeletesV1=true` is positively established and refuses a response with a missing/invalid revision instead of using its current fallback value. It sends the loaded revision with the destructive request, preserves the item on conflict or unknown outcome, refreshes authoritative state, and removes the row only after confirmed deletion. Confirmation and cleanup semantics continue to follow ADR-055 and ADR-107, and linked Media or Notes are never presented as deletion targets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Server hard delete remains visibly disabled with a reason unless `hasReadingOptimisticDeletesV1=true` is positively established for the active authority.
- [ ] #2 Server captures with a missing/invalid revision fail closed; no fallback revision is used for destructive actions.
- [ ] #3 Confirmed deletion sends the loaded revision, requires title-specific permanent confirmation, and removes the row only after authoritative success.
- [ ] #4 Conflict or unknown outcome preserves the capture, marks state stale, and offers authoritative Refresh without automatic retry; external Media and Notes are never deletion targets.
- [ ] #5 Local behavior remains unchanged and service/controller/mounted/live regressions cover source switches, late responses, narrow layouts, and ADR-055/ADR-107 semantics.
<!-- AC:END -->
