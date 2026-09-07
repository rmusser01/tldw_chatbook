---
id: TASK-16471
title: 'STTS note/conversation import: four DB helper imports don''t exist, both dialogs unreachable'
status: To Do
assignee: []
created_date: '2026-08-14'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Neither STTS selection dialog can be opened from the app at all: every DB helper the caller imports is missing. `tldw_chatbook/UI/STTS_Window.py:526, 539, 582, 600` import `fetch_all_notes`, `fetch_note_by_id`, `fetch_all_conversations`, and `fetch_messages_by_conversation_id` from `tldw_chatbook.DB.ChaChaNotes_DB` — none of the four exist (probed: all `hasattr` False). Both imports sit inside `try: ... except Exception` blocks (`STTS_Window.py:573-576` / `662-665`), so the ImportError is swallowed into a "Failed to import from notes/conversation: ..." toast and `push_screen` is never reached. The Speech screen's import-source Select (`STTS_Window.py:275-279` → `:455-459`) offers both "Notes" and "Conversation", so this is a live, user-reachable dead end — and it means TASK-15992's dialog fixes have zero user-facing value until this lands. Found by the TASK-15992 review (section B1b, scratchpad `review15992.md`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both import paths (Notes and Conversation) open their selection dialog end-to-end against real, existing DB methods
- [ ] #2 The exception handlers no longer swallow ImportError silently — programming errors are logged with a traceback, not just toasted
- [ ] #3 Born-red test evidence: a test reaching the dialog through the import path fails on current behavior before the fix
<!-- AC:END -->
