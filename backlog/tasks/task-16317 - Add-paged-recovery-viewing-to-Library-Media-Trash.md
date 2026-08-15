---
id: TASK-16317
title: Add paged recovery viewing to Library Media Trash
status: To Do
assignee: []
created_date: '2026-08-15 02:51'
labels:
  - library
  - pagination
  - media-trash
  - follow-up
dependencies:
  - TASK-16311
  - TASK-16312
  - TASK-16313
  - TASK-16314
  - TASK-16315
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
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
