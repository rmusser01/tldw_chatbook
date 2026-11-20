---
id: TASK-42012
title: "The Library is-current helper treats a detached screen as current"
status: To Do
assignee: []
created_date: 2026-11-19 19:39
updated_date: 2026-11-19 19:39
labels:
- follow-up
- ui
- library
priority: low
references:
- tldw_chatbook/UI/Library_Modules/screen_helpers.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41715 (round 17). `screen_helpers._library_screen_is_current` returns True when `app.screen` raises (`except Exception: return True`), so past the exit fence its 11 `library_screen.py` callers treat a detached screen as current.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The helper returns False for a detached or exiting screen
- [ ] #2 A test covers the exit-fence case, red before the fix
<!-- AC:END -->
