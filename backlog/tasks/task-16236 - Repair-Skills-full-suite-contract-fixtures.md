---
id: TASK-16236
title: Repair Skills full-suite contract fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 09:38'
updated_date: '2026-08-14 09:48'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the Skills full-suite chunk by aligning end-to-end fixtures with the current Console provider and Library editor readiness contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remote skill confirmation tests exercise the real bridge with a valid provider resolution.
- [x] #2 Dirty skill-editor navigation evidence explicitly establishes the editor's existing dirty-tracking readiness.
- [x] #3 The full Skills chunk and focused static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterize the failing bridge and Library fixture boundaries.
2. Replace the obsolete provider-resolution sentinel and explicitly establish the existing editor-ready signal in the isolated veto test.
3. Run the focused regressions, full Skills chunk, and scoped static checks.
4. Record verification evidence and close the task.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture repair that preserves existing provider and Library ownership boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced obsolete object() provider sentinels with contract-valid ConsoleProviderResolution fixtures and explicitly armed the isolated Library dirty-veto test before its programmatic edit. Focused regressions passed 3/3; the complete 25-file Skills chunk passed 320 tests. Ruff check/format remain exactly baseline-red in the two legacy test files, with no new diagnostic class; git diff --check passed. ADR required: no (test-fixture alignment only).
<!-- SECTION:NOTES:END -->
