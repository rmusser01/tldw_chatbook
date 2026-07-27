---
id: TASK-840
title: Normalize inherited Ruff format drift in Console citation files
status: To Do
assignee: []
created_date: '2026-07-27 04:02'
labels:
  - maintenance
  - formatting
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair pre-existing whole-file Ruff formatting drift in the four Console files surfaced by TASK-553.15 without mixing broad mechanical churn into the citation-repair feature implementation:

- `tldw_chatbook/Chat/console_agent_bridge.py`
- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Widgets/Console/console_transcript.py`

The same four files fail the formatter check on `origin/dev`, so this work is tracked independently of TASK-553.15.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The exact TASK-553.15 eleven-file Ruff format check exits zero
- [ ] #2 Formatting changes are mechanical and behavior-preserving
- [ ] #3 Scoped Console citation and UI tests pass after formatting
- [ ] #4 `git diff --check` passes and formatter-only scope is independently reviewed
<!-- AC:END -->
