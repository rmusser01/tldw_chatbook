---
id: TASK-41946
title: "A workflow phase running when the unproven-read marker is set fails at its next wave boundary"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- workflows
- needs-decision
priority: low
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions/285-native-phased-agent-workflows.md
dependencies:
- TASK-41645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41645 kept Stop/Pause/Resume/Accept/Remove working under the marker, but a phase already running when the marker is set still fails at its next wave boundary: the task in flight completes, the mid-phase publish is refused, the phase settles `failed`, and the chat stays held until restart. ADR-285 records this as a known limit. Whether the phase should keep running is a product choice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's choice (keep running, or fail at the boundary as now) is recorded in ADR-285
- [ ] #2 A test pins the chosen behaviour
<!-- AC:END -->
