---
id: TASK-41907
title: "Remaining one-pause press-then-read sites in the approval, media and characterization tests"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- tests
priority: low
references:
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_library_media_render_fixes.py
- Tests/UI/test_library_media_reader_flow.py
- Tests/UI/test_library_media_characterization.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Listed by TASK-41505 and TASK-41530 (round 16), not changed there: the three Alt+A tests and about 65 widget-harness one-pause sites in `test_console_mcp_approval.py`, the keyboard press-pause-assert sites in the media render-fixes and reader-flow files, and Media's "Open original" in `test_library_media_characterization.py` (~280-285), which has the same thread-worker shape TASK-41505 fixed for Collections.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed site waits on its observable condition with a bounded deadline
- [ ] #2 The changed files pass 10 of 10 at `-n 4` with the load recorded
<!-- AC:END -->
