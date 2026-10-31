---
id: TASK-35593
title: Prepare Console workspace reconciliation reads off the UI thread
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
Console resume and reconciliation synchronously admit workspace registry reads on Windows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Stock workspace reconciliation preserves live storage checks without blocking the UI for its finite reads
- [x] #2 Workspace/session/source changes and custom callback affinity remain correct
- [x] #3 Targeted tests and native before/after evidence verify the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR126/230/231 boundaries apply. Reason: reuse finite worker ownership without changing workspace authority. 1. Trace authoritative reconciliation and original registry reads. 2. Prepare eligible stock reads through existing finite owners and keep state mutation on the UI loop. 3. Verify invalidation, custom callbacks and cancellation. 4. Repeat native profiling and update PR3166.

Existing decisions: [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), [ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md), [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md). Ordinary preparation precedes runtime answerability; already-answerable partial retries keep the original synchronous route.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepare the original registry read after FULL sync and before runtime answerability, retaining native storage admission and finite controller/runtime ownership. A one-use result feeds the original synchronous reconciler; source/session/workspace/visit changes reject it. Custom readers and already-answerable partial retries remain direct. Sixteen native controls cover real held reads, operation retirement, repeated cancellation, teardown, stale sources and attach ordering. Paired main-thread registry samples fell 36 to 0; overall resume median improved 793 to 743 ms, but tails worsened. Draft timing does not measure repaint or completed screen actions; existing screen-message-pump/FULL synchronization remains a measured hotspot. Combined targeted evidence: 142 checks passed across runs, two unchanged baseline config-source setup failures documented. New files pass Ruff; existing findings unchanged. Review findings addressed and architecture updated. See Docs/Development/evidence/windows-performance-2026-10-18/RESUME_RESULTS.md. ADR required: no; existing [ADR-126](../decisions/126-complete-local-backup-and-recovery.md), [ADR-230](../decisions/230-console-send-preparation-and-io-ownership.md), [ADR-231](../decisions/231-console-polling-and-full-state-reconciliation.md) apply.
<!-- SECTION:NOTES:END -->
