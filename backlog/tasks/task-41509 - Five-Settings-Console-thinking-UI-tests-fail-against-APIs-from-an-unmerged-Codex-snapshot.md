---
id: TASK-41509
title: "Five Settings Console-thinking UI tests fail against APIs from an unmerged Codex snapshot"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- tests
- settings
priority: medium
references:
- Tests/UI/test_settings_console_thinking.py
dependencies:
- TASK-41031
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the trace stream (round 15). `Tests/UI/test_settings_console_thinking.py` has 5 failed nodes that call APIs which exist only in the unmerged Codex snapshot bf364a4bf7. This is separate from the two reasoning tests TASK-41031 already lists (`test_local_qwen_reasoning`, `test_local_reasoning_toggle`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each failing node is repointed at the current API, or deleted with the reason recorded
- [ ] #2 The file runs on a PR lane and passes
<!-- AC:END -->
