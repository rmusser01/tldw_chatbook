---
id: TASK-19579
title: Optimize compact Library Media browsing
status: In Progress
assignee: []
created_date: '2026-08-21 17:43'
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
