---
id: TASK-32929
title: Contain unexpected agent work recovery failures in Console
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:12'
updated_date: '2026-09-25 16:19'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the Console usable and show an honest owning-conversation failure when a selected earlier-turn worktree recovery cannot start or complete.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An unexpected recovery database or submission error does not terminate the Console.
- [x] #2 The owning view receives a bounded failure message, while a stale view receives none.
- [x] #3 The retained recovery owner drains and its failure receipt remains available without automatically retrying a Git effect.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a UI error-boundary bug fix under existing ADR-155; it changes no storage, authority, or cross-module contract.
1. Reproduce the mounted recovery picker failure on current dev.
2. Contain unexpected helper failures in the disposable UI worker while preserving retained-owner cleanup and cancellation.
3. Verify active and stale view messaging, physical-owner receipt, adjacent lifetime cases, and targeted static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The recovery picker now catches unexpected helper-start failures inside its disposable Textual worker and shows the existing bounded failure copy only when the originating view, session and conversation still match. The retained helper continues to own physical cleanup, receipt persistence and cancellation; no Git effect is retried.

Mounted picker regressions cover database-open failure, executor-submission rejection, stale-view suppression, drained capacity, retained failure receipts and no mutation-engine entry. The original two picker cases failed before the fix; the final three cases plus four adjacent submission/accepted-close cases passed (7 total). Ruff check/format and diff checks passed. The older three-second worker-entry test timed out twice during an earlier broad run, then passed when rerun sequentially on both pristine dev and this branch; it does not exercise the changed UI module. No full-suite or live-provider claim.

Files: tldw_chatbook/UI/Console_Modules/worktree.py; Tests/Chat/test_console_worktree_recovery_ui_errors.py. Existing ADR-155 governs recovery; no new ADR or user-guide change is needed for this failure-path repair.
<!-- SECTION:NOTES:END -->
