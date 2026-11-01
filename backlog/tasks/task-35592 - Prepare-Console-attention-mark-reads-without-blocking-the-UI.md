---
id: TASK-35592
title: Prepare Console attention-mark reads without blocking the UI
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19 05:30'
updated_date: '2026-10-19 06:15'
labels: []
dependencies: []
documentation:
  - Docs/Development/evidence/windows-performance-2026-10-18/RESUME_RESULTS.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console resume synchronously re-reads durable unseen receipt marks under native storage admission, blocking interaction on Windows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stock attention refresh reads no longer block Console resume and retain original native checks
- [x] #2 Receipt acknowledgement, notifications, source changes and cancellation retain their original ownership semantics
- [x] #3 Targeted native controls and before/after profiling verify the improvement
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR126/230/231 boundaries apply. Reason: bounded scheduling fix, no authority or storage changes. 1. Trace original attention reads/effects and capture baseline. 2. Separate finite original stock reads from current-loop publication, retaining direct custom/synchronous routes and existing runtime retirement. 3. Add held-read and stale/ack/cancel controls. 4. Profile and update PR3166.

Existing decisions: [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), [ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md), [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md). Validation includes repairing a baseline-proven stale transcript fixture so original receipt-acknowledgement tests reach their assertions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stock attention reads now use coalesced finite runtime preparation with operation-owned connections; original native admission, exact acknowledgement, notifications and synchronous custom API remain intact. New controls cover retained reads, stale source/revision, cancellation, custom affinity, reentrant refresh and disposal. Native paired main-thread read samples fell 79 to 0; whole-resume median 793 to 743 ms, with worse tails and initial regression documented rather than claiming a general speedup. Targeted evidence totals 142 passing checks across runs; two unchanged full-app config-source setup failures reproduce on the baseline. New helper/tests pass Ruff; existing 33 findings unchanged; independent review addressed. See Docs/Development/evidence/windows-performance-2026-10-18/RESUME_RESULTS.md and Docs/Architecture/console.md. No new ADR: existing [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), [ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md) and [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md) apply.
<!-- SECTION:NOTES:END -->
