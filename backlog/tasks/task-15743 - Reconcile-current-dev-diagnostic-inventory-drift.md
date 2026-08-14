---
id: TASK-15743
title: Reconcile current-dev diagnostic inventory drift
status: Done
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
- [x] #1 Every unsafe diagnostic in the reviewed stacked delta is replaced by a fixed event with only ADR-029-approved bounded metadata
- [x] #2 Reviewed-safe additions and provider-wrapper deletions are recorded without unrelated production edits or a sink-topology change
- [x] #3 The governed production-diagnostic inventory is regenerated once and equals live extraction on the final integration base
- [x] #4 Focused security, affected feature, static, privacy, and artifact gates pass with mutation-sensitive evidence; the repository-wide suite is excluded by owner direction
- [x] #5 Independent spec and code-quality reviews are approved before the ordered PR integration
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Reconciled the reviewed current-dev diagnostic delta to ADR-029 fixed-event,
  metadata-only shapes, retained the reviewed-safe additions and deletions, and
  regenerated the governed inventory after the final rebase. The final
  non-write extraction verifies 490 owners, 1,185 TASK-492 calls, 6,942
  TASK-494 calls, and the unchanged six-file sink topology.
- Added mutation-sensitive architecture evidence for the audited repairs and
  latest-dev additions. The focused diagnostic/security gate passed 22 tests;
  the affected Console image-controller gate passed 13 tests after bounding
  remote-image fetch-attempt memory to a 256-entry LRU.
- Rebased onto `origin/dev` at `546e5c4a6070d3b4ebca2274ad17e60442ea8699`,
  reconciled branch-added backlog task-ID collisions, and passed the exact
  duplicate-ID, Ruff, JSON, privacy, artifact, and diff-hygiene checks. The
  repository-wide suite was not run, per owner direction.
- Addressed Qodo's actionable unbounded-memory finding. Three requests to add
  persistent identifiers, configuration values, or exception tracebacks were
  not applied because they conflict with ADR-029; the rationale and governing
  tests were posted on the PR and Qodo acknowledged them as intentional design
  trade-offs requiring no further action.
- ADR required: no new ADR. The implementation enforces existing
  `backlog/decisions/029-local-private-data-boundary.md` without changing sink
  ownership, topology, or the allowed metadata policy.
<!-- SECTION:NOTES:END -->
