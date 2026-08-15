---
id: TASK-16234
title: Align RAG scope tests with user-facing cause copy
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:26'
updated_date: '2026-08-14 09:27'
labels:
  - testing
  - rag
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep scope-pipeline evidence aligned with the shared plain-language empty-scope notice instead of requiring internal diagnostic cause tokens in user-visible notifications.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scope tests assert the canonical user-facing notice
- [x] #2 User-visible assertions never require raw internal cause tokens
- [x] #3 Diagnostic state assertions retain the exact internal causes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the four raw-token notification failures as RED evidence.
2. Replace raw-token substring assertions with exact `scope_empty_notice` expectations.
3. Retain diagnostic cause assertions and run the full scope-pipeline module plus static checks.

ADR required: no
ADR path: N/A
Reason: This updates stale test evidence without changing the RAG scope boundary or copy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Updated four stale notification assertions to use the shared scope_empty_notice formatter and exact warning severity. Internal causes remain asserted in the diagnostics payload, while user-visible copy is now correctly protected from raw tokens such as deleted-items and workspace-scope-unavailable. Verification: the complete scope-pipeline module passed 77 tests; Ruff lint/format, py_compile, and git diff --check passed.
<!-- SECTION:NOTES:END -->
