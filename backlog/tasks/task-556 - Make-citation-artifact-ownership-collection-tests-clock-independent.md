---
id: TASK-556
title: Make citation artifact ownership collection tests clock-independent
status: Done
assignee: []
created_date: '2026-07-24 21:53'
updated_date: '2026-07-24 22:20'
labels:
  - rag
  - citations
  - tests
dependencies:
  - TASK-553.8
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep artifact-owner foreign-key and collection-race coverage deterministic after the wall clock passes the suite's fixed collection instant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Artifact ownership fixtures anchor owner, message, and conversation timestamps before the fixed collection clock.
- [x] #2 The shared-database foreign-key collection case exercises deletion instead of being skipped by retention.
- [x] #3 The collection-before-artifact race reaches its transaction hook and preserves an ordinary ungrounded save.
- [x] #4 The artifact ownership suite and citation foundation gate pass without production behavior changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm the failure is caused by real-clock message and conversation timestamps remaining later than the fixed collection clock.
2. Add a test-local persistence wrapper that anchors conversation, message, and message-owner timestamps before NOW while preserving the production repository seam.
3. Rerun the two regressions, the complete artifact ownership suite, and the citation foundation gate; run Ruff and diff checks.
4. Obtain independent review, record implementation notes, complete acceptance criteria, and mark the task Done.

ADR required: no
ADR path: N/A
Reason: This is a deterministic test-fixture correction only; production storage, lifecycle policy, and cross-module contracts do not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a test-local persistence wrapper that delegates canonical fixture creation to the repository helper, then normalizes conversation, message, and message-owner timestamps to one day before the fixed collection clock. This restores deterministic execution of the shared-database foreign-key deletion and collection-before-artifact race paths without changing production code or retention policy.

Verification: the two original regressions passed (2/2); the complete artifact ownership suite passed (48/48); the citation foundation gate passed (762/762); qualification was eligible and overall-pass with all budgets green, zero migration restart duplicates, and about 1,044 messages/second median throughput; ChaChaNotes/DB passed (332/332); Ruff check, Ruff format check, and git diff check passed. Independent review approved after the acceptance criteria were split into four traceable outcomes. An additional full-repository soak was interrupted after two non-reproducible setup errors appeared while two other worktrees were concurrently running full suites; the narrowed Auth/CI/early-ChaCha boundary passed 85/85, so that contaminated soak is not claimed as a pass.
<!-- SECTION:NOTES:END -->
