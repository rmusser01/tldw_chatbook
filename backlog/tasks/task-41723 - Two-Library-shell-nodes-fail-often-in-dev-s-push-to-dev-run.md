---
id: TASK-41723
title: "Two Library shell nodes fail often in dev's push-to-dev run"
status: To Do
assignee: []
created_date: 2026-11-18 06:57
updated_date: 2026-11-18 06:57
labels:
- follow-up
- tests
- library
priority: medium
references:
- Tests/UI/test_library_shell.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41532's review (round 16), reading 19 push-to-dev runs: `test_library_shell_ingest_canvas_submit_clears_path_and_title_keeps_metadata` failed 8 times (also red in run 44190585155) and `test_library_media_initial_error_is_unknown_and_retry_is_unique` failed 4 times. TASK-41532 named only four other nodes, so the job will likely stay red without these.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each node's failure shape across those runs is recorded, with its cause
- [ ] #2 Each passes 10 of 10 at `-n 4` at load above 60, or the product defect is fixed with a red-before test
<!-- AC:END -->
