---
id: TASK-39102
title: 'get_character_list_for_ui returns an empty list on a DB error'
status: To Do
assignee: []
created_date: '2026-11-06 09:00'
labels:
  - character-chat
  - bug
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`get_character_list_for_ui`, the dropdown loader, returns `[]` on a database error, so a fault looks the same as having no characters. It exists on dev today; this task's PR does not fix it. Found by the Roleplay B2b follow-up burn-down (2026-11-05), read on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A database error in the dropdown loader is distinguishable from an empty character list for its callers
- [ ] #2 A test injects a DB error and asserts the fault is surfaced, not returned as []
<!-- AC:END -->
