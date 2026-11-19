---
id: TASK-41945
title: "About 50 workflow tests are red on dev: native waves, native returns, journeys and the Console workflow scheduling file"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- tests
- workflows
priority: medium
references:
- Tests/Chat/test_workflow_phase_wave_native.py
- Tests/Chat/test_workflow_native_return.py
- Tests/Chat/test_console_workflow_scheduling.py
dependencies:
- TASK-41645
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** While verifying TASK-41645 (load about 20), these failed on dev d0486978dd and on the branch alike: 24 native-wave, native-return and journey tests (for example `Tests/Chat/test_workflow_phase_wave_native.py`, `test_workflow_native_return.py`) and 26 of 30 in `Tests/Chat/test_console_workflow_scheduling.py` (for example `assert acquisition[2] is scope` in `enter_other`). Not load: the machine was quiet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing file has a recorded cause per node group
- [ ] #2 Drift is repaired and product defects are fixed red-first; the files pass
<!-- AC:END -->
