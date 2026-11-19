---
id: TASK-41912
title: "The compaction hold flow file's last red test expects a tooltip that no longer says it"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_compaction_hold_flow.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41511 (round 16): its harness fix took `test_console_compaction_hold_flow.py` from 8 red to 1. The last expects "Next send pauses for compaction approval" in a tooltip that no longer says it. The file is otherwise unlaned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test pins the current tooltip copy or the copy is restored, with the reason recorded
- [ ] #2 The file runs on a PR lane or the notes say why not
<!-- AC:END -->
