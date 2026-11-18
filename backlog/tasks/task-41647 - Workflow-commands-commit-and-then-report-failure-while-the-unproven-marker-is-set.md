---
id: TASK-41647
title: "Workflow commands commit and then report failure while the unproven marker is set"
status: To Do
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
