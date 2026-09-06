---
id: TASK-31814
title: Close fixture-owned durable-turn and recovery SQLite handles
status: Done
assignee:
  - '@codex'
created_date: '2026-09-06 05:13'
updated_date: '2026-09-06 05:34'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the reproduced 378-descriptor growth from the durable-turn and recovery test selection without weakening ownership, rollback or retry assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every diagnosed fixture-owned controller drains before its exact database handles close and the zero-connection guard remains active.
- [x] #2 Complete affected recovery files pass without retained fixture SQLite descriptors or the recorded growth warnings.
- [x] #3 Reuse the explicit test ownership fixture with no production, shared-conftest, garbage-collection or resource-threshold changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A. Reason: test-only lifecycle repair using the existing shutdown/quiescence APIs. 1. Preserve current108-test native descriptor attribution: 378 retained descriptors from seven recovery/acceptance files. 2. Explicitly import TASK31812 close_owned_console_resources into those files; retain all real constructors and behavioral assertions. 3. Reproduce the other recorded234-FD three-file boundary selection and include only files with attributed test handles. 4. Verify all affected complete files with native per-test FD attribution and the zero-registry assertion; inspect exceptional teardown and no foreign-path capture. 5. Scoped lint/format, independent review, checkpoint evidence and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reproduced378 retained descriptors in108 durable/recovery cases and234 in54 boundary cases via native post-finalizer attribution. Added explicit ownership-fixture imports only to those ten diagnosed files; all behavior assertions unchanged. Uses TASK31812 shutdown/quiescence/zero-registry fixture, including error/cancellation containment. Complete final combined247-case resource selection passed147.85s with no retained SQLite descriptors or growth warning; three dependency warnings remain. Scoped lint/import-region format/diff checks and independent review pass; checkpoint and testing lesson updated. ADR required:no; no production/shared-conftest/GC/threshold changes.
<!-- SECTION:NOTES:END -->
