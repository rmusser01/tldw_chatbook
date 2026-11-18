---
id: TASK-41646
title: "The workflow driver's failure handler leaves never-launched preparations open for reconcile"
status: In Progress
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made reconcile close a retained run's never-launched, unwaved preparation, so the chat accepts a new turn. The driver's own failure handler sees that same preparation first but leaves it open, so the run stays retained until a reconcile happens to run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The driver handler closes a never-launched preparation itself, recording the same review row reconcile records
- [ ] #2 A test fails a run before launch and asserts that the chat accepts a new turn without reconcile
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (the ADR-285 TASK-41320 section already allows closing a never-launched preparation; this moves the same close earlier, into the driver).

1. Red test first in `Tests/Chat/test_workflow_failed_phase_release.py`: the wave call refuses (ordinary exception) after task-1 is prepared and before any wave binds it; assert the run is not retained, the preparation is closed without launch or dispatch, the workflow is the resumable failed-phase state `blocked_phase` reaches, and the chat accepts a new turn with no reconcile. Red on d0486978dd: the open preparation counts as live, so the run is retained.
2. In `_execute`'s TASK-40718 handler, close the run's never-launched, unwaved preparations (the same runtime-state condition and the same `close()` reconcile uses, shared as one helper) before the live check. The existing retirement tail then settles the phase as failed and releases the chat.
3. The TASK-41320 scenarios that need a retained run with an open never-launched preparation now get it from a cancelled wave call (a BaseException the handler deliberately does not treat as an ordinary failure).
4. Mutation check (drop the new close loop: the new test is red), covering workflow files on base and head.
<!-- SECTION:PLAN:END -->
