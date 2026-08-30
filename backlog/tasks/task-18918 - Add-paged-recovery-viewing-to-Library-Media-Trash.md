---
id: TASK-18918
title: Add paged recovery viewing to Library Media Trash
status: In Progress
assignee: []
created_date: '2026-08-15 02:51'
updated_date: '2026-08-30 15:43'
labels:
  - library
  - pagination
  - media-trash
  - follow-up
dependencies:
  - TASK-18912
  - TASK-18913
  - TASK-18914
  - TASK-18915
  - TASK-18916
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - >-
    Docs/superpowers/specs/2026-08-30-task-18918-library-media-trash-paging-design.md
  - >-
    Docs/superpowers/plans/2026-08-30-task-18918-library-media-trash-paging.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every deleted Media item reachable in the nested Trash recovery surface through bounded pages while preserving restore, permanent-delete, selection, and recovery semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Media Trash exposes coherent exact-total bounded pages with deterministic ordering and complete-source filtering before slicing.
- [ ] #2 Restore and permanent delete reconcile or relocate the affected stable ID truthfully, clamp emptied pages, and never misreport a committed mutation as failed.
- [ ] #3 Trash selection cannot remain invisibly active across page or scope changes, and stale refresh failures disable destructive actions until authoritative recovery.
- [ ] #4 Loading, empty, failure, Retry, focus, back navigation, and narrow-terminal pager behavior match the established Library pagination convention.
- [ ] #5 Request generations, unmount fencing, malformed envelopes, concurrent shrink, and privacy-safe diagnostics have regression coverage.
- [ ] #6 Automated database/service/state and mounted Textual tests plus isolated live verification with more than 40 synthetic Trash records pass.
<!-- AC:END -->

## Implementation Plan

1. Add a coherent local-only database page/count/facet contract.
2. Propagate and canonically validate the exact envelope through Media services.
3. Add immutable Trash paging state plus a Trash-specific request controller.
4. Wire screen entry, paging/filter generations, Back receipt, and lifecycle fencing.
5. Render the bounded pager/filter/confirmation surface at all supported sizes.
6. Reconcile Restore and permanent deletion through the shared Media mutation owner.
7. Run focused automated/live verification, review, documentation, and closeout.

ADR required: no

ADR path: `backlog/decisions/067-library-top-level-pagination-contracts.md`

Reason: ADR-067 already governs exact source-owned pages and stale mutation recovery.
