---
id: TASK-41647
title: "Workflow commands commit and then report failure while the unproven marker is set"
status: In Progress
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
- ux
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** This came from TASK-41320's review (M4). While a run carries the unproven marker, a workflow command commits its change and then fails in the trailing `_publish`/get, so the user sees an error for an action that took effect. Stop is the visible case: the run stops, but the command reports failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A command that committed reports success, or a distinct "applied, view not refreshed" state, never a plain failure
- [ ] #2 Stop keeps working on a run with the unproven marker
- [ ] #3 A test covers Stop under the marker
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, a dated ADR-285 amendment under the TASK-41320 section (what a command that took effect reports when its read-back fails).

1. Red test first (`Tests/Chat/test_workflow_failed_phase_release.py`): with a running phase (held task-1) and the coordinator's unproven-read marker set by another chat's workflow read, Stop takes effect (durable `stopped`, `run.control == "stop"`) but on d0486978dd reports plain `workflow_return_unproven` from its trailing `_publish`.
2. One coordinator helper does every command's trailing publish and read-back. A failure there raises a distinct `workflow_command_applied`, chained from the cause; cancellation still propagates.
3. The Console maps that reason to "<Action> applied · the workflow view could not be refreshed", naming a restart when the cause is the unproven marker, and keeps the action notice when the following refresh fails.
4. ADR-285 amendment and the User Guide Recovery paragraph; mutation check; covering files on base and head.
<!-- SECTION:PLAN:END -->
