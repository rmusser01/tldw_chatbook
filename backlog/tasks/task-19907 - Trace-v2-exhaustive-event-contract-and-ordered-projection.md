---
id: TASK-19907
title: Trace v2 exhaustive event contract and ordered projection
status: In Progress
assignee: []
created_date: '2026-08-22 18:26'
updated_date: '2026-08-22 18:47'
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
- [ ] #1 ADR-080 defines observable-event semantics, causal identifiers, privacy boundaries, and compatibility with ADR-066 and ADR-067
- [ ] #2 One pure projection normalizes every documented observable event family from its existing local owner without creating a duplicate all-events database
- [ ] #3 Projected events preserve deterministic order, stable identity, parent/source relationships, actor identity, status, timing, missing-data reasons, and structured payload metadata
- [ ] #4 Existing v1 trajectory conversations continue to render without backfill
- [ ] #5 Contract, adapter, ordering, causal-lineage, privacy, and compatibility tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: this changes the durable cross-module event and collaboration contract. 1. Add the stable causal event envelope and pure multi-owner adapters. 2. Persist agent steps incrementally with explicit-index idempotent recovery. 3. Verify legacy trajectory compatibility, ordering, privacy, and live refresh.
<!-- SECTION:PLAN:END -->
