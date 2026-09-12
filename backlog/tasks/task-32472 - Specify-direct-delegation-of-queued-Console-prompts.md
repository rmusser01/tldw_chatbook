---
id: TASK-32472
title: Specify direct delegation of queued Console prompts
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 18:18'
updated_date: '2026-09-08 18:28'
labels:
  - console
  - agents
  - design
dependencies: []
references:
  - backlog/decisions/137-queued-console-agent-delegation.md
  - >-
    Docs/superpowers/specs/2026-09-08-task-32050-queued-agent-delegation-design.md
---

## Renumbering provenance

Renumbered from TASK-32472 on 2026-09-11: the id collided with a task that
arrived on dev while this branch was in review (owner rule TASK-19601 — the
older arrival keeps the id). No dependencies referenced the old id.


## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define how users can route individual queued Console prompts to named background agents while preserving ordering, request ownership, permissions, result delivery, and recovery. This task produces reviewable design documents only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A detailed design records the confirmed routing, launch-and-continue, task-only context, completion-wake, and FIFO capacity-wait behavior.
- [x] #2 A proposed canonical ADR resolves launch ownership, idempotent acceptance, queue and wake scheduling, capacity, authority, and stop/recovery semantics.
- [x] #3 The design includes an observable acceptance matrix and accounts for existing queue and fleet contracts without claiming the feature is implemented.
- [x] #4 The documents and task are cross-linked and pass a consistency and local-link review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes

ADR path: backlog/decisions/137-queued-console-agent-delegation.md

Reason: Direct user launches change run ownership, acceptance persistence, runtime interfaces, and the existing queue and wake scheduling contracts. The ADR remains Proposed until the written design is reviewed.

1. Reconcile the approved brainstorming choices and design-review findings with ADR-046, ADR-069, ADR-134, ADR-135 and the current queue/fleet source.
2. Write the detailed specification and proposed ADR, resolving the outstanding scheduling policies and stating the scope of the task-only context and memory-only queue guarantees.
3. Self-review lifecycle, identity, cancellation, authority, and recovery cases; check local links, whitespace, and cross-document consistency.
4. Record the design deliverables and evidence in this task and hand off the written proposal for review. Application implementation and its implementation plan are separate work.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Produced the [detailed specification](../../Docs/superpowers/specs/2026-09-08-task-32050-queued-agent-delegation-design.md), [proposed ADR-137](../decisions/137-queued-console-agent-delegation.md), and ADR index entry. Incorporated all seven review findings, plus source-path delivery, provider-history exclusion, ephemeral sessions, usage/change-review ownership, and FIFO acceptance versus physical scheduling.

Resolved the outstanding policies: release unused primary capacity during delegation waits and visibly reacquire it for a later Main entry; Handle ready results pauses the queue for one eligible same-chain automatic wake and leaves it paused. Target selection is available before enqueue and uses stable definition IDs.

Verification: self-reviewed the lifecycle and acceptance matrix; local links, separate task criteria, placeholders, and whitespace checked. No runtime code, application tests, full-suite runs, or live-provider verification were involved. The matrix specifies future implementation evidence. ADR required: yes; ADR path: backlog/decisions/137-queued-console-agent-delegation.md. Both documents remain Proposed for user review; this design-only task does not approve or implement the feature.

Task hygiene: renamed the CLI-assigned 32041 to 32050 after the all-ref/all-worktree scan found a maximum of 32049. This repeats the existing documented CLI collision issue; no new lesson was needed.
<!-- SECTION:NOTES:END -->
