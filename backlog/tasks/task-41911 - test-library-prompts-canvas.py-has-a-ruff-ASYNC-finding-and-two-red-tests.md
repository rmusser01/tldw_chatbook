---
id: TASK-41911
title: "test_library_prompts_canvas.py has a ruff ASYNC finding and two red tests"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
priority: low
references:
- Tests/UI/test_library_prompts_canvas.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41502 (round 16), present at dev: ASYNC109 (`asyncio.wait_for` where `asyncio.timeout` is wanted) around line 3240, and the `history_confirmation_geometry` (ConfirmationDialog `Container`) and notes-editor CSS-parity tests fail.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The ruff finding is fixed and both tests pass, or each test's premise is updated with the reason
<!-- AC:END -->
