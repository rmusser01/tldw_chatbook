---
id: TASK-34659
title: Exercise-independent-SQLite-cursor-retirement-after-managed-transaction
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-07 14:29'
updated_date: '2026-10-07 18:27'
labels: []
dependencies: []
priority: high
type: bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the retained closed-cursor baseline by exercising an independent native cursor rather than reusing the transaction-owned cursor after scope exit. Preserve real maintenance exclusion and source-owner retirement.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All five core owners demonstrate committed data and live independent cursor exclusion until explicit retirement.
- [x] #2 Transaction-owned CharactersRAGDB cursors still close at scope exit; production transaction and admission semantics remain unchanged.
- [x] #3 The original failure and final affected checks are retained without skips or warning suppression.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Retain the original CharactersRAGDB closed-cursor RED.
2. Acquire an independent cursor from the already-admitted connection; pin the managed cursor closure contract.
3. Run the complete affected lifetime module, lint/format and Backlog guards; review and record outcomes.

ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: test-only correction to existing native borrower ownership; no production ownership or admission change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the test-only independent native cursor correction and explicit managed-cursor closure assertion. Original RED: closed-cursor failure, 1 failed in 1.17s. Complete affected module GREEN: 60 passed in 19.95s, no pytest warnings. Changed-range format passes; full-file Ruff retains the exact same 26 code/message diagnostics as base, not full-file lint-clean. Production/admission untouched; self-review and both Backlog guards pass. ADR126 applies; no new ADR required. Evidence: Docs/QA/task-34659/README.md. Integration/static-debt disposition remains pending; no push or merge performed.

Later integration selection preserved this complete lifetime module without failures; current static comparison still has the exact inherited 26 diagnostics and no additions. The wider eight-module aggregate remains non-green on four other nodes, and TASK34661 descriptor retirement is incomplete. This bounded test correction does not waive those results. Source/evidence preserved locally; no fresh-context review, push or merge.

Final current-source review retains the exact tested cursor source hash and 26 inherited lint findings with no additions; changed-range formatting passes. All original lifetime controls passed in the later affected selection. Separate approved app/AgentRuns ownership changes are TASK34661, not this test-only correction. QA reconciled; task stays In Progress for review/integration and retained static-gate disposition. No production transaction/admission change, source commit, push, merge or other-session control.
<!-- SECTION:NOTES:END -->
