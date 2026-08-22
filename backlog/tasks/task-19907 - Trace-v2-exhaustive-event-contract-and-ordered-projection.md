---
id: TASK-19907
title: Trace v2 exhaustive event contract and ordered projection
status: Done
assignee: []
created_date: '2026-08-22 18:26'
updated_date: '2026-08-22 20:41'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - >-
    Docs/superpowers/plans/2026-08-22-task-19907-19910-trace-v2-event-foundation.md
  - >-
    backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the exhaustive observable-event contract and a pure causal projection over existing local owners: messages and trajectory metadata, append-only agent run steps, compaction, retrieval provenance, approvals, and feedback. Preserve legacy trajectory rendering and local-only privacy without duplicating all events into another store.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-080 defines observable-event semantics, causal identifiers, privacy boundaries, and compatibility with ADR-066 and ADR-067
- [x] #2 One pure projection normalizes every documented observable event family from its existing local owner without creating a duplicate all-events database
- [x] #3 Projected events preserve deterministic order, stable identity, parent/source relationships, actor identity, status, timing, missing-data reasons, and structured payload metadata
- [x] #4 Existing v1 trajectory conversations continue to render without backfill
- [x] #5 Contract, adapter, ordering, causal-lineage, privacy, and compatibility tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: this changes the durable cross-module event and collaboration contract. 1. Add the stable causal event envelope and pure multi-owner adapters. 2. Persist agent steps incrementally with explicit-index idempotent recovery. 3. Verify legacy trajectory compatibility, ordering, privacy, and live refresh.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-080 event envelope as a pure, legacy-safe projection over existing message, trajectory, agent-run, retrieval, compaction, approval, and feedback owners. The adapters preserve stable source identity, causal relationships, actor/status/timing metadata, explicit field states, and privacy classifications; deterministic iterative SCC ordering supports large chains, collisions, cycles, concurrent branches, and coherent turn blocks without introducing another event database.

Agent steps now receive UTC creation timestamps and persist incrementally through the existing append-only `agent_run_steps` table. Explicit-index recovery validates payload/index integrity, treats identical retries idempotently, preserves first-writer evidence on conflicts, fills unrelated missing rows before reporting conflicts, and does not mutate lifecycle timestamps. Capture failures remain contained so run/UI finalization continues.

ADR: `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`. Verification covered projection contracts, causal ordering/privacy/legacy compatibility, live UI refresh, runtime/service behavior, SQLite recovery/conflicts, wake-ledger compatibility, Ruff, and diff checks. No generalized lesson entry was added because the discovered SCC and recovery edge cases are encoded directly as focused regressions.
<!-- SECTION:NOTES:END -->
