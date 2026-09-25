---
id: TASK-32929
title: Contain unexpected agent work recovery failures in Console
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:12'
updated_date: '2026-09-26 16:04'
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
- [x] #4 Unexpected recovery failures leave a bounded diagnostic containing only the exception class, with no exception text or private path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a UI error-boundary bug fix under existing ADR-155; it changes no storage, authority, or cross-module contract.
1. Reproduce the mounted recovery picker failure on current dev.
2. Contain unexpected helper failures in the disposable UI worker while preserving retained-owner cleanup and cancellation.
3. Verify active and stale view messaging, physical-owner receipt, adjacent lifetime cases, and targeted static checks.
4. Keep the UI and receipt failure copy in sync, and record only a failure class at the retained owner; verify the diagnostic inventory.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The recovery picker now catches unexpected helper-start failures inside its disposable Textual worker and shows the existing bounded failure copy only when the originating view, session and conversation still match. The retained helper continues to own physical cleanup, receipt persistence and cancellation; no Git effect is retried.

Mounted picker regressions cover database-open failure, executor-submission rejection, stale-view suppression, drained capacity, retained failure receipts and no mutation-engine entry. The original two picker cases failed before the fix; the final three cases plus four adjacent submission/accepted-close cases passed (7 total). Ruff check/format and diff checks passed. The older three-second worker-entry test timed out twice during an earlier broad run, then passed when rerun sequentially on both pristine dev and this branch; it does not exercise the changed UI module. No full-suite or live-provider claim.

Files: tldw_chatbook/Chat/console_worktree_recovery.py; tldw_chatbook/UI/Console_Modules/worktree.py; Tests/Chat/test_console_worktree_recovery_ui_errors.py; Docs/security/production-diagnostic-inventory.json. Existing ADR-155 governs recovery; no new ADR or user-guide change is needed for this failure-path repair.

PR review follow-up: failure copy now comes from one helper-owned constant, and retained completion records one class-only warning after saving the bounded receipt. The mounted picker regression asserts exactly one warning without injected error text. All seven focused cases, Ruff check/format, diff check, and the persistent diagnostic inventory checker passed. The inventory statement review found one added warning with only the exception class argument; the checked inventory gained exactly one owner/call and no sink change. A second mock-screen test was declined because the mounted picker already exercises the same branch through the real dialog, active/stale view gates, and retained owner.

Latest-dev rebase: regenerated the diagnostic inventory against dev 50f6096. The source-level statement review still finds only the fixed class-only recovery warning. Dev had already carried an internally stale TASK-494 summary (7,535 stored versus 7,537 in its unchanged owner rows); regeneration preserves its rows, adds one TASK-492 owner/call, and corrects that summary without adding TASK-494 calls.

Second latest-dev reconciliation: dev 1b61ee2 added a two-call AppFooterStatus TASK-494 owner row while leaving the stored summary at 7,535. Regeneration now yields the row-consistent 7,539 summary alongside the same single class-only TASK-492 recovery warning; no recovery behavior changed.

Final conflict reconciliation: rebased onto dev c4225b5d38896adb99e992efebeaa206090cdcce. Recovery code and tests are unchanged; the generated inventory retains all current dev entries and adds only the single class-only recovery warning (599 owners, 1,361 TASK-492 calls, 7,542 TASK-494 calls, 14 sinks). All seven focused recovery/lifetime cases, Ruff check/format, diagnostic inventory verification, task-ID guard and diff check passed.
<!-- SECTION:NOTES:END -->
