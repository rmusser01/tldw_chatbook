---
id: TASK-41041
title: "The construct-time import allowlist test is red on dev with 15 unreviewed modules"
status: To Do
created_date: 2026-11-15 02:03
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- performance
- boot
priority: low
references:
- Tests/App/test_construct_runtime_imports.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/App/test_construct_runtime_imports.py::test_construct_time_runtime_imports_stay_within_the_allowlist` fails on dev ecad404bd4 (2026-11-15). `TldwCli.__init__` imports 15 repo modules that are not on its reviewed allowlist, among them:
- `Backup_Recovery.async_file_participants`
- `Canvas.config_policy`
- `Chat.prompt_history`
- `MCP.recovery_activation`
- `Media_Generation`

TASK-36947.2's review found the test red at 5ca7ec6a03 (2026-11-06), with 18 modules. A guard that is always red cannot catch the next unreviewed construct-time import. The file runs on no PR lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed module is either moved off the construct path or added to the allowlist with its reason
- [ ] #2 The test passes on dev
- [ ] #3 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->
