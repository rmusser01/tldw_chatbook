---
id: TASK-16471
title: 'STTS note/conversation import: four DB helper imports don''t exist, both dialogs unreachable'
status: Done
assignee:
  - '@zcode'
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
- [x] #1 Both import paths (Notes and Conversation) open their selection dialog end-to-end against real, existing DB methods
- [x] #2 The exception handlers no longer swallow ImportError silently — programming errors are logged with a traceback, not just toasted
- [x] #3 Born-red test evidence: a test reaching the dialog through the import path fails on current behavior before the fix
<!-- AC:END -->

## Implementation Notes

Closed as already-fixed on dev, verified rather than assumed. Task-19576 (merged via PR #1926, "task/19576-burn") rerouted BOTH import paths off the four nonexistent ChaChaNotes_DB helpers before this task was picked up:

- ``_import_from_notes`` now routes through the shared ``notes_scope_service`` seam + ``NoteSelectionDialog`` (worker: ``_import_from_notes_worker``).
- ``_import_from_conversation`` routes through ``chat_conversation_scope_service`` + ``ConversationSelectionDialog``, with a bounded page walk (``get_messages_with_context``, capped at 5000 messages) replacing the old unbounded nonexistent fetch.

Verification on current dev (worktree at fresh origin/dev): the four helper names appear nowhere in STTS_Window.py; ``Tests/UI/test_stts_audiobook_import_scope_services.py`` -- 19576's born-red suite -- passes 5/5, covering exactly this task's ACs: both paths load real note content / real conversation messages end-to-end against real service seams (AC#1), and each path's failure mode notifies instead of crashing when the service is unavailable (the old silent-ImportError handlers were removed wholesale by the reroute, so the swallowing code AC#2 names no longer exists; the remaining handlers log-and-notify rather than silently swallow) and the suite was written red-first against the crash (AC#3, per 19576's own record). No code changed in this task.

Modified: this task file only.
