---
id: TASK-19910
title: Capture sub-agent lifecycle and causal lineage events
status: In Progress
assignee: []
created_date: '2026-08-22 18:29'
updated_date: '2026-08-22 23:53'
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
- [ ] #1 Agent spawn, start, progress, handoff, completion, cancellation, and failure events are captured with stable run and parent identifiers
- [ ] #2 Each child run can be traced to the event that created it and the events it produced
- [ ] #3 Parallel child runs retain deterministic per-conversation ordering without losing their own sequence
- [ ] #4 Safe task summaries are stored without private reasoning content
- [ ] #5 Integration tests cover multiple parallel children, failure, cancellation, and parent completion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md. Reason: ADR-080 already approves the AgentRuns v14 spawn_event_id migration and precise child-spawn correlation; this task implements that recorded boundary. 1. Add failing real-seam lineage tests for reserve/spawn/start/progress/handoff/terminal/resume and two parallel children. 2. Add the v14 nullable spawn_event_id migration and thread the preallocated parent spawn event through inline and fleet create_run paths. 3. Project durable run/step lineage with stable parent/source links and safe task summaries, emitting only transitions not recoverable from existing owners. 4. Verify migration history/current schema, deterministic parallel ordering, privacy, continuation, cancellation/failure, and parent completion.
<!-- SECTION:PLAN:END -->
