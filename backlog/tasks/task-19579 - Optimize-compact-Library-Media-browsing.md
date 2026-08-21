---
id: TASK-19579
title: Optimize compact Library Media browsing
status: In Progress
assignee: []
created_date: '2026-08-21 17:43'
updated_date: '2026-08-21 17:56'
labels:
  - library
  - ux
  - textual
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the compact Library Media browse experience so regular technical and non-technical users can scan and act on several records efficiently without weakening truthful paging, focus, recovery, or wide-screen behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At exact 100x30, a settled populated Media browse paints at least five one-line rows using title, media type, and relative age while the preview is neither painted nor keyboard-focusable.
- [ ] #2 Activating a compact row opens the existing Media viewer and Back restores the applied page, focused row, and list scroll position.
- [ ] #3 At 170x48, the existing two-line rows and side-by-side preview remain unchanged.
- [ ] #4 Crossing the existing 120-column breakpoint performs no Media read, page reset, filter reset, selection reset, or user-focus steal.
- [ ] #5 Compact Select, loading, stale, Retry, paging, mutation receipt, and disabled-reason states remain truthful and keyboard accessible.
- [ ] #6 Focused Textual geometry and interaction tests cover the production CSS at 100x30 and 170x48; relevant user documentation is updated.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Correct the returning-user Media test harness baseline.
2. Add compact row presentation and vertical allocation.
3. Preserve breakpoint focus without Media reads or state reset.
4. Restore semantic row and scroll position after viewer Back.
5. Verify state variants, production CSS geometry, documentation, and closeout.

Detailed plan: Docs/superpowers/plans/2026-08-21-library-compact-media-browsing.md
ADR required: no
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: ADR-067 already owns authoritative Media paging, stale recovery, and mutation refresh; this task changes only responsive presentation and focus restoration.
<!-- SECTION:PLAN:END -->
