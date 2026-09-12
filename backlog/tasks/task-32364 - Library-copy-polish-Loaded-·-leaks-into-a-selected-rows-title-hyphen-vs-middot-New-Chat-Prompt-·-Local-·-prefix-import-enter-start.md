---
id: TASK-32364
title: >-
  Library copy polish: 'Loaded ·' leaks into a selected row's title, hyphen vs
  middot, 'New Chat', 'Prompt · Local ·' prefix, import 'enter start'
status: Done
assignee: []
created_date: '2026-09-11 06:19'
updated_date: '2026-09-11 08:00'
labels:
  - library
  - copy
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selecting a media row prefixes its title with 'Loaded ·' (A caps 39/47); conversation rows use '5 messages - 16m' where every other list uses '·' and the untitled conversation reads 'New Chat' (A cap 54); every Prompts row begins 'Prompt · Local ·' and 'System + User' is schema-speak (A cap 57); the import footer says 'enter start' but the first Enter only validates (A caps 07/08). Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Status words never prefix a title
- [x] #2 One separator glyph across lists
- [x] #3 The import footer names what Enter does at each step
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move the Loading/Loaded state word out of the title into the secondary in _media_row_label
2. Conversations separator hyphen -> middot; 'New Chat' -> 'Untitled conversation'
3. Prompts row: drop 'Prompt · ' prefix, 'System + User' -> 'has system and user text'
4. Import footer: state-dependent Enter label
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: _media_row_label_rest moved the Loading/Loaded word off the TITLE and onto the fact line ('Attention Is All You Need' / 'pdf · added 1h ago · loaded'). task-30044's constraint -- the SHORT word, never the old 'Loaded in Reader' prose -- still holds. Two pins updated: test_library_media_toolbar_adapt.py (now asserts every density opens with the title AND ends with the state word) and test_library_media_reader_shell.py.

AC#2: library_conversations_state._secondary_text changed '-' to '·'; three pins in Tests/Library/test_library_conversations_state.py updated. The 'New Chat' half of the task DESCRIPTION is not in library_conversations_state.py as the plan expected -- that file already falls back to 'Untitled conversation'. The 'New Chat' the critique saw is a REAL STORED TITLE written by Chat/chat_conversation_service.py:155 and chat_persistence_service.py:1206 at creation time; renaming it there would rename user data and touch Console, so it is left alone and reported. AC#2 itself ('one separator glyph') is satisfied.

Prompts copy: library_prompts_canvas.py drops the leading 'Prompt · ' only when type_label == 'Prompt' -- Recipe and Template still name themselves, since the canvas title covers only the first. 'System + User' -> 'has system and user text' in library_prompts_state.py (where the lane summary is actually built; the plan expected it in the canvas). The three sibling lane values ('System only' / 'User only' / 'Empty') keep their older phrasing because the task named only this one -- the resulting mixed register is flagged for a follow-up.

AC#3: _library_ingest_shortcuts_for_current_state now derives the Enter label from _build_library_ingest_state().start_enabled -- the SAME gate handle_library_ingest_path_submitted obeys -- so the footer cannot promise an import the keypress will decline: 'enter check this path' while shut, 'enter start import' once open. LIBRARY_INGEST_SHORTCUTS keeps the open-gate wording as its base. Two pins in test_library_ingest_keyboard.py were re-pointed from the raw constant to the method, which IS the shared single source now.

Live-verified 235x52: the conversation row reads '5 messages · 1h · Default · Active'; the media row reads 'pdf · added 1h ago · keyword: notes · loaded' with an unprefixed title; the Import footer reads 'enter check this path'. The open-gate half could NOT be live-verified: local media Import cannot run on this host (POSIX semaphores exhausted -> multiprocessing.Pool [Errno 28]), so the pre-check never settles and the Start gate never opens. Both strings are pinned by a unit test that drives the controller's state directly.

Files: tldw_chatbook/Widgets/Library/library_media_canvas.py; tldw_chatbook/Library/library_conversations_state.py; tldw_chatbook/Widgets/Library/library_prompts_canvas.py; tldw_chatbook/Library/library_prompts_state.py; tldw_chatbook/UI/Library_Modules/library_ingest_controller.py; tldw_chatbook/UI/Screens/library_screen.py (LIBRARY_INGEST_SHORTCUTS only); Tests/UI/test_library_crit10_media_rows.py, test_library_media_toolbar_adapt.py, test_library_media_reader_shell.py, test_library_ingest_keyboard.py, test_library_prompts_canvas.py; Tests/Library/test_library_conversations_state.py, test_library_prompts_state.py; Docs/User_Guide/library/media-and-conversations.md, prompts.md, import-and-export.md.
## Fix round 1 (review response)

AC#3 was met in the pure function but not in the running app: nothing
re-registered the Ingest footer when the Start gate opened. Tracing it under
a real harness narrowed the review's claim -- the common blank-to-valid-path
transition is saved by accident, because filling in an empty `type_groups`
sends `_update_library_ingest_dynamic_regions` down its STRUCTURAL branch and
that recomposes. Every transition that is NOT structural left the footer
naming the previous step's action. `_resync_library_ingest_footer()` now runs
at the pre-flight seam every gate transition passes through; the new app-test
drives a non-structural transition and is red without it.

Also: the Prompts row now keys off `artifact_type` rather than the rendered
`type_label`, and the state word is built lower-case once.

The two halves of the critique this task could not close honestly are now
riders rather than notes inside a closed task: **task-32378** (the prompt
lane summary reads four different ways across Library and Console) and
**task-32379** (the stored "New Chat" title, a cross-surface product call).
<!-- SECTION:NOTES:END -->
