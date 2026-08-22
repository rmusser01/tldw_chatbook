---
id: TASK-19910
title: Capture sub-agent lifecycle and causal lineage events
status: To Do
assignee: []
created_date: '2026-08-22 18:29'
labels: []
dependencies:
  - TASK-19907
references:
  - Docs/superpowers/specs/2026-08-22-task-19907-trace-v2-exhaustive-collaboration-design.md
  - Docs/superpowers/plans/2026-08-22-task-19907-19910-trace-v2-event-foundation.md
  - backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md
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
