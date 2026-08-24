---
id: TASK-19502
title: Decouple Change Review finalization from Console turn completion
status: Done
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
- [x] #1 The bridge schedules end-of-turn review work and returns the terminal run outcome without awaiting filesystem or Git finalization
- [x] #2 One bounded app-owned coordinator orders baseline-to-end windows FIFO per canonical root across conversations without holding UI, store, or database objects in filesystem workers
- [x] #3 Multi-root admission is all-or-nothing and shared-root survivor windows retain correct successor ordering
- [x] #4 Review markers derive idempotently from durable assistant anchors and durable change rows for both result-first and anchor-first races and after remount
- [x] #5 A deterministic mounted three-turn test holds review finalization and proves the third turn starts after the second assistant is durable
- [x] #6 Shutdown stops admission, bounded-drains pure workers, rejects late-generation results, disposes the coordinator, and only then closes AgentRunsDB
- [x] #7 Failed and cancelled turns still request review finalization and bounded failures persist honest tracking-error results
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
ADR path: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md`
Reason: ADR-084 already defines the cross-root ordering, publication, and lifecycle boundaries implemented by this task.

Detailed plan: `Docs/superpowers/plans/2026-08-21-change-review-nonblocking-finalization.md`
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a lazy, fixed-worker `ChangeReviewFinalizationCoordinator` with bounded queues, atomic canonical-root admission, dynamically enrolled nested-root lanes, FIFO B/E execution, and bounded generation-safe shutdown. Admitted lane reservations are capped exactly at configured capacity; lightweight rejections reserve a separate bounded error-publication slot, with a typed visible fallback only when that channel is saturated. Filesystem workers receive only tracker/operation/result objects and never persistence/UI owners.
- Extracted caller-owned tracker discovery/B/E operations and event-backed baseline readiness. A timed-out baseline rejects late success, while the legacy `begin_turn` wrapper remains compatible.
- Production Console turns now schedule E from the bridge `finally` and return immediately. Failed/cancelled turns take the same path. Survivor windows conservatively retain their root lanes until the originating turn's last child settles; this differs from the plan's proposed successor-B overlap and avoids attribution gaps or double counting.
- Added transactional per-window snapshot batches plus a payload-free revision/pending signal. The mounted transcript keeps polling while review is pending and injects only durable Change Review rows at the render boundary; ordinary agent-step and diff-feedback rows remain on their existing live/resume paths and cannot be duplicated by a mixed block.
- Publication remains lifecycle-owned until its database callback returns. An atomic batch failure gets exactly one terminal tracking-error attempt. The publisher and runtime close their own thread-local AgentRunsDB connections; a bounded shutdown timeout cannot close a connection still in use by the other thread or append a late overload marker to the store.
- Runtime disposal orders controller shutdown, coordinator shutdown, thread-local AgentRunsDB close, and provider close. A mounted Console regression holds turn two's E, records its durable assistant anchor, and proves turn three reaches the provider before E is released.
- Verification: 92/92 complete coordinator/tracker/execution-context/runtime tests passed; 16/16 focused mounted/projection/state tests passed; the earlier five-file run passed 395 tests and exposed 12 unrelated bare-screen fixture failures now closed by TASK-19507. Ruff and `git diff --check` passed. Independent re-review reported no remaining blocker or Important findings.
- ADR required: no. Existing ADR: `backlog/decisions/084-change-review-consent-and-asynchronous-finalization.md`.
<!-- SECTION:NOTES:END -->
