---
id: TASK-636
title: Account for Library ingest preflight thread worker
status: Done
assignee:
  - '@codex'
created_date: '2026-07-25 19:31'
updated_date: '2026-07-25 19:31'
labels:
  - library
  - workers
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the unified-shell worker-policy sentinel synchronized with the reviewed Library ingest preflight filesystem worker without weakening its exact-count guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The sentinel documents ingest preflight analysis as a legitimate Library thread worker and all worker-loop exceptions remain explicitly annotated.
- [x] #2 The Library allowlist retains an exact count and still rejects unreviewed thread-worker growth.
- [x] #3 The focused policy sentinel and unified-shell release-gate block pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the exact-count failure and audit all five Library `@work(thread=True)` call sites.
2. Update the policy rationale and exact Library count for the synchronous filesystem preflight scan.
3. Run the focused sentinel, unified-shell gate block, and static checks.

ADR required: no
ADR path: N/A
Reason: This updates a test allowlist for an existing reviewed worker; it does not change the worker boundary or application behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Audited all five Library thread-worker decorators and updated the exact-count
  sentinel plus its rationale for synchronous ingest-path preflight analysis.
- The corrected count exposed a previously masked formatting mismatch on one
  legitimate `asyncio.run` exception; moved its existing policy annotation
  onto the AST call line, matching the other two exceptions and the sentinel.
- The focused policy test and complete 26-test unified-shell block passed.
  Ruff, formatting, compile, and diff checks passed.
<!-- SECTION:NOTES:END -->
