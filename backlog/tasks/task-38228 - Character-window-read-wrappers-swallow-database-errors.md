---
id: TASK-38228
title: 'Character window-read wrappers swallow database errors'
status: To Do
assignee: []
created_date: '2026-10-30 18:43'
labels:
  - character-chat
  - bug
dependencies: []
references:
  - tldw_chatbook/Character_Chat/Character_Chat_Lib.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
get_character_page_for_ui and count_character_page log a database error and return [] and 0, so their callers cannot tell a failed read from an empty one. It exists on dev today; Roleplay frame B2b's failure row with Retry therefore cannot show for a database fault inside them (the list reads as "no match"), and B2b does not fix the wrappers. Found during Roleplay frame B2b (TASK-33910.4, plan Docs/superpowers/plans/2026-10-26-roleplay-b2b-toolbar-new-import.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A database error in a character window read reaches its caller as an error, not as an empty result
- [ ] #2 Roleplay shows its failure row with Retry for that error, pinned by a test that injects it
<!-- AC:END -->
