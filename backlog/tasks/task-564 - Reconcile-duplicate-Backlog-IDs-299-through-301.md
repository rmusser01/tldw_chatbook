---
id: TASK-564
title: Reconcile duplicate Backlog IDs 299 through 301
status: Done
assignee: []
created_date: '2026-07-25 19:05'
updated_date: '2026-07-25 19:05'
labels:
  - backlog
  - reliability
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the repository-wide unique-task-ID invariant by applying the existing reviewed renumbering of the later unstarted RAG closeout tasks to 391 through 393.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scheduling TASK-299, mypy TASK-300, and model-catalog TASK-301 retain their established IDs and references.
- [x] #2 The unstarted RAG closeout tasks are renamed to TASK-391 through TASK-393 in both filenames and frontmatter.
- [x] #3 The Backlog frontmatter uniqueness sentinel and product-maturity harness pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is repository task-ledger hygiene and applies an existing reviewed renumbering; no architecture or product boundary changes.

1. Verify all duplicate pairs, creation status, references, and the existing reconciliation commit.
2. Apply reviewed commit 78fd975b2, which keeps the established/Done task IDs and moves only the later unstarted RAG trio to 391-393.
3. Run the frontmatter uniqueness sentinel, the complete Phase-1 harness file, Backlog CLI hydration checks, and diff/static checks.
4. Record the reconciliation evidence, close the task, and resume TASK-546 fail-fast verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Applied the repository's existing reviewed reconciliation commit 78fd975b2 (cherry-picked as f410b3934). The established scheduling TASK-299, mypy TASK-300, and completed model-catalog TASK-301 retain their IDs and references. Only the later unstarted RAG closeout trio moved: hybrid retrieval key namespaces to TASK-391, RAG admin-surface decision to TASK-392, and orphan-widget audit to TASK-393; filenames and frontmatter IDs both changed.

Verification: the complete Tests/UI/test_product_maturity_phase1_harness.py passed 2/2; Backlog CLI individually hydrated TASK-299, 300, 301, 391, 392, and 393 to the expected unique files; git diff --check passed. The reviewed commit's history audit confirmed no RAG-side dependencies/subtasks required reference updates and all scheduling TASK-299.x references remain valid. ADR required: no. Modified: the three renamed task records plus this reconciliation task.
<!-- SECTION:NOTES:END -->
