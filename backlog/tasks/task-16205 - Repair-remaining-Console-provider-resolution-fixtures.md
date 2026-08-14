---
id: TASK-16205
title: Repair remaining Console provider-resolution fixtures
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 06:38'
updated_date: '2026-08-14 06:38'
labels:
  - test-health
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore diff-channel and parked-cancellation tests on the current Console provider-resolution contract so they exercise their intended bridge behaviors instead of terminating during provider setup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Diff-channel bridge tests reach and verify tool-marker behavior with a valid resolution.
- [x] #2 The parked cancellation test reaches the provider stream and persists cancellation after the controller task returns.
- [x] #3 Focused and containing-chunk tests plus scoped static/diff gates pass, with mutation evidence for both obsolete fixtures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: test doubles are aligned with an existing Console provider boundary.

1. Reproduce both stale resolution shapes and compare them with current canonical Console fixtures.
2. Replace the opaque and partial values with minimal contract-valid ConsoleProviderResolution instances.
3. Prove each old shape fails, then run focused, chunk, lint, format-drift, and diff gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the diff-channel bridge's opaque resolution and the parked-cancellation gateway's partial namespace with minimal current `ConsoleProviderResolution` values. The pre-change chunk provided strict RED evidence: all three diff-channel outcomes ended `error`, and the surviving parked bridge thread persisted `error` instead of `cancelled`. GREEN: 54 focused tests and the entire 25-file containing chunk (1,336 tests) passed. Scoped Ruff check and diff-check passed. Both test files are Ruff-format-red identically on the implementation base, so the narrow fixture edits preserve inherited formatting instead of rewriting unrelated tests. ADR required: no; test doubles only.
<!-- SECTION:NOTES:END -->
