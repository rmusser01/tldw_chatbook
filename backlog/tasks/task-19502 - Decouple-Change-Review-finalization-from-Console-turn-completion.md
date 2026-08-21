---
id: TASK-19502
title: Decouple Change Review finalization from Console turn completion
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-21'
labels:
  - console
  - performance
  - concurrency
dependencies:
  - TASK-19501
priority: critical
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure a completed assistant response releases the Console turn and prompt queue without waiting for filesystem snapshot finalization, while preserving per-root ordering, survivor-window attribution, durable review publication, and safe shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The bridge schedules end-of-turn review work and returns the terminal run outcome without awaiting filesystem or Git finalization
- [ ] #2 One bounded app-owned coordinator orders baseline-to-end windows FIFO per canonical root across conversations without holding UI, store, or database objects in filesystem workers
- [ ] #3 Multi-root admission is all-or-nothing and shared-root survivor windows retain correct successor ordering
- [ ] #4 Review markers derive idempotently from durable assistant anchors and durable change rows for both result-first and anchor-first races and after remount
- [ ] #5 A deterministic mounted three-turn test holds review finalization and proves the third turn starts after the second assistant is durable
- [ ] #6 Shutdown stops admission, bounded-drains pure workers, rejects late-generation results, disposes the coordinator, and only then closes AgentRunsDB
- [ ] #7 Failed and cancelled turns still request review finalization and bounded failures persist honest tracking-error results
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extract synchronous, caller-owned B/E tracker operations without changing the legacy tracker wrapper.
2. Add a bounded app-owned coordinator with all-or-nothing canonical-root lane reservations and per-root FIFO execution.
3. Register B before model execution and schedule E from the terminal bridge path without awaiting it.
4. Move survivor-window lineage into the coordinator so successors and last-child settlement share exact SHA boundaries.
5. Publish change rows independently from assistant anchors and refresh transcript markers through an idempotent durable join.
6. Dispose Console admission, coordinator workers/publisher, and persistence in a tested bounded order.
7. Run deterministic three-turn, concurrency, remount, failure, cancellation, and shutdown verification.

ADR required: no
ADR path: `backlog/decisions/077-change-review-consent-and-asynchronous-finalization.md`
Reason: ADR-077 already defines the cross-root ordering, publication, and lifecycle boundaries implemented by this task.

Detailed plan: `Docs/superpowers/plans/2026-08-21-change-review-nonblocking-finalization.md`
<!-- SECTION:PLAN:END -->
