---
id: TASK-41713
title: "test_library_screen_reuse.py cannot run locally and is on no PR lane"
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
- Tests/UI/test_library_screen_reuse.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41500 (round 16). The file's async tests build a `_scratch_env` that trips `RecoveryRequired` locally (red at dev too), and the file is not laned, so TASK-41500's re-pinned `test_suspended_library_gates_ingest_dom_work_until_resume` was reviewed by reading only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every test in the file passes locally through the scratch wrapper
- [ ] #2 The file runs on a PR lane with its measured time recorded
<!-- AC:END -->
