---
id: TASK-16240
title: Remove obsolete legacy Parakeet buffer vertical-slice tests
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 10:03'
updated_date: '2026-08-14 10:07'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Transcription test suite aligned with the app-owned shared local STT executor after the legacy Parakeet buffer backend became intentionally unreachable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The three stale no-dispatch Parakeet buffer tests are removed.
- [x] #2 Current shared-runtime and facade coverage remains green.
- [x] #3 The affected Parakeet Transcription module is green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failures are caused by the intentional shared-dispatcher boundary and map each assertion to current shared-runtime or facade coverage.
2. Remove only the three obsolete legacy-buffer tests.
3. Run the focused replacement coverage, full affected module, chunk, and static checks.

ADR required: no
ADR path: N/A
Reason: This is a test-only cleanup aligning stale coverage with the existing STT ownership boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Removed three tests that called the intentionally unreachable legacy Parakeet buffer path without the app-owned dispatcher.
- Kept coverage in the current shared Parakeet runtime and facade suites for in-memory PCM, no file staging, language semantics, source resolution, and required-dispatcher behavior.
- Verified 28 focused/current-architecture tests, the full remaining vertical-slice module, Ruff check/format, and diff hygiene. The enclosing sweep separately exposed unrelated unbounded UI test waits.
<!-- SECTION:NOTES:END -->
