---
id: TASK-41949
title: "Probe children and import probes still use short fixed waits that time out under load"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- flaky
priority: low
references:
- Tests/Chat/workflow_planning_probe.py
- Tests/Chat
dependencies:
- TASK-41638
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41638 made private-child budgets load-tolerant. Still fixed: the workflow probe children's internal 10 s waits (for example `usage_ready.wait(10)`), which time out under load on dev too, and the import-closure and packaging probes' 60 s `subprocess.run` budgets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed wait uses a load-tolerant bound or the shared CPU-idle-aware wait, or carries a comment saying why not
<!-- AC:END -->
