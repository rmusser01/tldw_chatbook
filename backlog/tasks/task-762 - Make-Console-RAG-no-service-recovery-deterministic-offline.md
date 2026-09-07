---
id: TASK-762
title: Make Console RAG no-service recovery deterministic offline
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 17:57'
updated_date: '2026-07-27 19:38'
labels:
  - console
  - rag
  - baseline
  - offline
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the Console Library RAG no-service path stage its recoverable blocked state without attempting embedding-model initialization or network access, eliminating the deterministic offline baseline failure inherited from dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No-service Console RAG action stages a blocked recoverable result.
- [x] #2 The no-service path performs no embedding download or network access.
- [x] #3 Existing configured-service RAG staging remains unchanged.
- [x] #4 The exact no-service regression and focused Console RAG tests pass offline.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Correct the Console no-service regression setup so the app has no Library RAG search service and assert the shared RAG factory is never reached.
2. Preserve the existing configured-service test as the compatibility guard; make no production-code changes unless the corrected regression exposes a real behavior defect.
3. Run the exact no-service regression, the configured-service Console RAG regression, and the focused Console RAG subset offline; run Ruff on the touched test and git diff --check.
4. Complete task acceptance criteria, implementation notes, self-review, and Backlog status.

ADR required: no
ADR path: N/A
Reason: Routine correction of a stale test fixture that exercises an existing service boundary; no storage, ownership, service-contract, security, or architectural decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected the stale Console no-service regression so it explicitly removes the
app-owned Library RAG search service, records that the shared RAG factory is never
reached, and waits for the final blocked state instead of an intermediate card. No
production code changed; the existing configured-service staging test remains the
compatibility guard.

ADR required: no. ADR path: N/A. This is a test-fixture correction at an existing
service boundary.

Verification was intentionally scoped to touched Console RAG behavior: the exact
regression passed; five `console_rag` tests in the touched file passed with
`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; the underlying Library RAG
no-service adapter test passed offline; Ruff check and format check passed for the
touched test file; and `git diff --check` passed.
<!-- SECTION:NOTES:END -->
