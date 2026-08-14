---
id: TASK-15743
title: Reconcile current-dev diagnostic inventory drift
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-14 03:10'
labels:
  - security
  - diagnostics
  - baseline
dependencies:
  - TASK-3070.2
  - TASK-16001
references:
  - Docs/superpowers/specs/2026-08-14-task-15743-current-dev-diagnostic-reconciliation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the ADR-029 production-diagnostic boundary on current dev after the
Console image-controller extraction and ingest-test repair are stacked, without
silently blessing private metadata or exception capture in the regenerated
inventory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every unsafe diagnostic in the reviewed stacked delta is replaced by a fixed event with only ADR-029-approved bounded metadata
- [ ] #2 Reviewed-safe additions and provider-wrapper deletions are recorded without unrelated production edits or a sink-topology change
- [ ] #3 The governed production-diagnostic inventory is regenerated once and equals live extraction on the final integration base
- [ ] #4 Focused security, affected feature, static, privacy, and artifact gates pass with mutation-sensitive evidence; the repository-wide suite is excluded by owner direction
- [ ] #5 Independent spec and code-quality reviews are approved before the ordered PR integration
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed executable plan:
`Docs/superpowers/plans/2026-08-14-task-15743-current-dev-diagnostic-reconciliation.md`

1. Freeze the audited owner/call disposition on top of rebased TASK-3070.2 and TASK-16001.
2. Add failing architecture evidence for every surviving reviewed diagnostic shape.
3. Redact only the audited unsafe production calls while preserving behavior.
4. Regenerate the existing manifest once and verify the complete stack.
5. Complete independent reviews, then rebase and merge TASK-15743 after both dependencies land.

ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: this task enforces an existing privacy boundary and does not change
diagnostic storage, sink ownership, or the allowed metadata policy.
<!-- SECTION:PLAN:END -->
