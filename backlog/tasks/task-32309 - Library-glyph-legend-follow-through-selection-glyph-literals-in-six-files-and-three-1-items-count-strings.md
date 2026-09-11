---
id: TASK-32309
title: >-
  Library glyph legend follow-through: selection glyph literals in six files and
  three '1 items' count strings
status: To Do
assignee: []
created_date: '2026-09-11 00:56'
labels:
  - library
  - ux
  - critique-9
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The critique-9 grammar branch (task-32235, PR #2580) made library_shell_state.py the one home of the state glyphs, but its reviewer found the selection pair still hard-coded as literals in six Library files (library_media_canvas.py:169, library_notes_canvas.py:1178 and 1323, library_prompts_canvas.py:1011, library_conversations_canvas.py:260, UI/Library_Modules/canvas_sync.py:247) and three count strings that still read '1 items' or '1 prompts' (library_export_state.py:325, library_export_controller.py:878 and 882, library_prompts_controller.py:1134). Both were outside that branch's file set.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Library selection glyph is read from the shared constants; a grep for the selection glyph literals under tldw_chatbook/ finds only the constants module
- [ ] #2 Every Library count string pluralises correctly (1 item, 2 items, 1 prompt)
<!-- AC:END -->
