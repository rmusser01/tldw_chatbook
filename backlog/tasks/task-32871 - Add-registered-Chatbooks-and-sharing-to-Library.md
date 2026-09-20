---
id: TASK-32871
title: Add registered Chatbooks and sharing to Library
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 22:00'
updated_date: '2026-09-20 07:18'
labels:
  - library
  - artifacts
dependencies:
  - TASK-32870
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Browse registered Chatbooks and Console saved responses in the Library artifact reader, with capability-based actions and a visible link to the existing ZIP-pack manager. Keep multi-item sharing and its application-owned lifetime. Governed by ADR-172 and stage 2 of Docs/superpowers/plans/2026-09-19-library-artifacts.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All matching registered Chatbooks are reachable, Console response excerpts are labeled, and sharing is offered only for usable exported bundles.
- [x] #2 Manage Chatbook packs opens the existing manager with its creation, import, template, export, and delete workflows intact.
- [x] #3 Multi-item sharing remains available and Library exposes Manage and Stop across types, filters, pane collapse, and navigation.
- [x] #4 Targeted registry, manager-link, sharing, production-CSS, and worker-lifecycle checks pass without altering storage or publication authority.
- [x] #5 Delayed share dialogs cannot publish after Library suspension, canvas navigation, or leave-and-return; publication is rechecked on the UI thread immediately before push.
- [x] #6 Library being covered by its own share modal does not invalidate that exact dialog’s explicit result; accepted sharing and existing shares survive presentation invalidation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Reason: Implements accepted registered artifact and app-owned sharing boundaries. 1. Extend the approved catalog with one-parse registry snapshots and exact saved response details. 2. Add Chatbooks and the existing manager link to the Library reader. 3. Add a Library-wide share strip and presentation-fenced dialog adapter preserving accepted modal results and app-owned operations. 4. Verify targeted registry, manager, sharing, lifecycle and production CSS evidence. Groundwork may run alongside Reports UI verification once catalog contracts are stable.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented complete registered Chatbook browsing with full saved responses, labeled excerpts, valid-ZIP sharing capabilities, exact source/Console handoffs, and the existing pack-manager link. Library-wide sharing preserves accepted app-owned work across navigation and exposes URLs/Manage/Stop. 60 registry checks, 25 sharing-owner checks and 7 mounted Chatbooks checks pass; a real local server served a staged ZIP and stopped cleanly. Markdown restoration waits for actual mount, documented as a lesson. ADR: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Evidence and explicit baseline limits: Docs/superpowers/plans/2026-09-19-library-artifacts-verification.md. User guide updated. No full suite requested; changed/new focused tests pass, added-line Ruff diagnostics are zero, generated CSS checks pass, and native private-profile TldwCli was verified. Existing recovery-initialization and repository-wide screen-size/workflows-style failures remain documented; their limits were not raised.
<!-- SECTION:NOTES:END -->
