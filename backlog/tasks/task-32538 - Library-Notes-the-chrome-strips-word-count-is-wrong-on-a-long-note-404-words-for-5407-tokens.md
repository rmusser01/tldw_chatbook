---
id: TASK-32538
title: >-
  Library Notes: the chrome strip's word count is wrong on a long note ("404
  words" for 5,407 tokens)
status: To Do
assignee: []
created_date: '2026-09-13 06:46'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor B, persona Alex, Edit workflow on the 35 KB seeded note. D3. New with task-32143 / PR #2615 (the strip did not exist at critique #2).

**What happened.** Open "Very long note — scaling laws digest" (DB: 35,204 chars, 5,407 `\S+` tokens, 363 lines). Chrome strip: "404 words · 1:1", then "404 words · 362:1" after Ctrl+End, then "407 words · 363:22" after typing four words (B 31, 33). 404 is the character length of the list's first row title ("Markdown showcase"). Info's own count was not exercised. Captures: B 31, 33.

**Cause.** INFERRED: `_note_word_count` (`tldw_chatbook/UI/Library_Modules/library_notes_controller.py:3136`, `re.finditer(r"\S+")`) is correct on its input and the strip repaints whatever value it is fed (`tldw_chatbook/Widgets/Library/library_notes_canvas.py:3020-3040`), so the fed value is wrong — the strip is being handed the wrong text on open, and the per-edit delta then accumulates on top of it. Docs contradicted: notes.md's chrome strip "N words · L:C".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The chrome strip's word count equals _note_word_count of the open note's body on load and after every edit, verified on the 35 KB fixture (≈5,400)
- [ ] #2 A regression test opens a multi-thousand-word note through the production open path and asserts the strip value is the body's count, not the length of a list-row title
<!-- AC:END -->
