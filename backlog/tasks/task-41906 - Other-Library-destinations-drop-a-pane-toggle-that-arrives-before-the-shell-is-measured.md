---
id: TASK-41906
title: "Other Library destinations drop a pane toggle that arrives before the shell is measured"
status: To Do
assignee: []
created_date: 2026-11-18 20:31
updated_date: 2026-11-18 20:31
labels:
- follow-up
- ui
- library
priority: medium
references:
- tldw_chatbook/UI/Screens/library_screen.py
- tldw_chatbook/UI/Library_Modules/library_conversation_reader_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41503 (round 16), not reproduced. TASK-41503 kept an early Collections toggle until the shell is measured (`_deferred_layout_priority`). The Media, Conversations, Notes, File Notes, Prompts and Skills layout syncs return the same way when the shell is not measured (library_screen.py ~6293 for Notes, library_media_controller.py ~10662, library_prompts_controller.py ~790, library_notes_controller.py ~2690 for File Notes, library_skills_controller.py ~783, library_conversation_reader_controller.py ~845), so they probably drop an early toggle too.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each destination's early pane toggle is applied once measured, or the notes show it cannot arrive early
- [ ] #2 Tests pin the destinations that can, red before the fix, reusing TASK-41503's deferral shape
<!-- AC:END -->
