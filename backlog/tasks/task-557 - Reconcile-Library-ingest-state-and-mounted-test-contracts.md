---
id: TASK-557
title: Reconcile Library ingest state and mounted test contracts
status: In Progress
assignee: []
created_date: '2026-07-25 17:53'
updated_date: '2026-07-25 17:53'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic Library ingest behavior after persisted type options were added by keeping render derivation side-effect free, treating generic chunk controls as built-in, and aligning mounted tests with scoped config and recompose contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Building Library ingest display state does not mutate the screen-owned form echo
- [ ] #2 Generic chunk-size and overlap controls follow the sibling Chunk toggle instead of optional-package detection
- [ ] #3 Search-history and rail-preference precedence tests reject only calls to their own config sections
- [ ] #4 The different-canvas ingest completion test tolerates the rail recompose while preserving selection
- [ ] #5 The five deterministic RED cases and focused ingest/config suites pass
- [ ] #6 Task notes record RED evidence and ADR applicability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the five deterministic RED failures and trace each to the current form/config/recompose contract.
2. Derive the render-time generic options from a copied form so compose remains side-effect free.
3. Distinguish sibling-field dependencies from optional-feature dependencies for generic chunk controls.
4. Narrow config precedence sentinels to the sections under test and poll across the expected rail recompose.
5. Run the deterministic cases, focused Library ingest/config suites, the full Library shell module, Ruff, formatter, and diff checks.
6. Self-review and document the separate note-conflict flake.

ADR required: no
ADR path: N/A
Reason: These are routine correctness and test-determinism repairs inside the existing Library ingest form and mounted UI contracts; no storage, ownership, service, or cross-module boundary changes.
<!-- SECTION:PLAN:END -->
