---
id: TASK-34360
title: Own response rule learning and assessment in Console runtime
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 05:55'
updated_date: '2026-10-04 10:53'
labels: []
dependencies:
  - TASK-34359
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 7. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible settled responses are checked before completion release and next queue drain.
- [x] #2 Stop, foreground input and source changes fence late activation or correction.
- [x] #3 App-owned state survives view remount and public editor testing never autoactivates changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: app-owned learning, assessment and completion/queue lifecycle. Follow Task 7 RED-GREEN: real generation boundary checks before completion publication/drain, current source and binding CAS, foreground/Stop cancellation including after generation, fresh manual repair source, editor validation and disposable view projections; targeted runtime/shutdown/ownership checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed app-owned learning, completed-answer assessment, source/binding CAS, editor Test reuse and immutable revisions, physical helper accounting, disposable projections, completion holds, bounded repair and real AgentService budget transfer. ADR-219 owns the architecture. Verified 212 targeted tests, five independently reproduced unsupported external-command exclusions and one inherited expected failure. New runtime, builder/store/model/resource seams and Console controller/store/queue/runtime/cost/status owners were updated; black changed ranges, scoped ruff and package mypy pass. Self-review found and fixed nested generation holds and lost hook-free budgets with actual AgentService RED-GREEN witnesses.
<!-- SECTION:NOTES:END -->
