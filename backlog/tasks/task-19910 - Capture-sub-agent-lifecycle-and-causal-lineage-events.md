---
id: TASK-19910
title: Capture sub-agent lifecycle and causal lineage events
status: Done
assignee: []
created_date: '2026-08-22 18:29'
updated_date: '2026-08-23 03:05'
labels: []
dependencies:
  - TASK-19907
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
Record parent and child agent lifecycle, handoffs, progress, completion, cancellation, and failures as first-class causal Trace events.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent spawn, start, progress, handoff, completion, cancellation, and failure events are captured with stable run and parent identifiers
- [x] #2 Each child run can be traced to the event that created it and the events it produced
- [x] #3 Parallel child runs retain deterministic per-conversation ordering without losing their own sequence
- [x] #4 Safe task summaries are stored without private reasoning content
- [x] #5 Integration tests cover multiple parallel children, failure, cancellation, and parent completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: ADR-080 already approves the AgentRuns v14 spawn_event_id migration and precise child-spawn correlation; this task implements that recorded boundary. 1. Add failing real-seam lineage tests for reserve/spawn/start/progress/handoff/terminal/resume and two parallel children. 2. Add the v14 nullable spawn_event_id migration and thread the preallocated parent spawn event through inline and fleet create_run paths. 3. Project durable run/step lineage with stable parent/source links and safe task summaries, emitting only transitions not recoverable from existing owners. 4. Verify migration history/current schema, deterministic parallel ordering, privacy, continuation, cancellation/failure, and parent completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-080's AgentRuns v14 migration with nullable `spawn_event_id`, including exact-v13 upgrade, fresh bootstrap, reopen, and idempotence coverage. Child runs are precreated before dispatch and retain distinct `parent_run_id`, precise spawning `spawn_event_id`, and continuation `resumed_from_run_id` across native inline/fleet, skill-tool, retained-resume, and thread-start-failure paths.

Agent lifecycle history now uses append-only existing AgentStep ownership for reserve, create, resume, start, progress/steering, completion, failure, cancellation, and supersession. A shared per-run sequence allocator preserves deterministic concurrent order; lifecycle successors link to the actual durable event or diagnostic written. Terminal status/result plus lifecycle insertion is atomic and first-writer/conflict safe, with bounded recovery and truthful incomplete diagnostics. Live and retained steering preserve the parent command cause after reload.

Durable task/result/log summaries withhold hidden reasoning, credentials, PEM material, and local path/file content while retaining useful public output. Process-local fleet handles remain usable in memory but are replaced with durable run IDs in AgentRuns, wake notices, logs, foreign-survivor, and pruned-retention paths. Runtime/Console `max_steps` is capped at 199,999 so control, trace, diagnostic, and lifecycle index bands remain disjoint; the user guide documents this ceiling.

Real temporary-database integration tests cover parallel success, failure, cancellation, supersession, handoff, resume, thread-start faults, capture failure/recovery, parent completion, migration, privacy, handle pruning, deterministic projection, and atomic rollback/reconciliation. ADR: `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`. Three Console wake-safety timeouts were reproduced identically at pre-task commit `e5a354431` before AgentService dispatch and are documented as unrelated baseline failures; no production workaround was added.
<!-- SECTION:NOTES:END -->
