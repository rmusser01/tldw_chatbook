---
id: TASK-32101
title: >-
  Library Conversations: `#library-conversation-use-source` has no compose site;
  its two handlers are dead; hand-off copy and annotation leftovers
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 22:42'
updated_date: '2026-09-10 15:14'
labels:
  - library
  - conversations
  - cleanup
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32056 review (PR #2523): nothing composes `#library-conversation-use-source`, yet `library_conversations_controller.py:1638` and `library_screen.py:35604` handle its press; `_open_console_tooltip()` still says 'Press Link to workspace' in the generic blocked state where the Link button is hidden; `library_conversation_reader_controller.py:150` annotates the injected callable as `Callable[[], str]` (it returns `tuple[str, bool]`); `media-and-conversations.md` describes a 'no active workspace' message-only case the code cannot produce; the blocked state paints 'Open in Console' twice (button label + Static line). Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The dead handlers are removed or the control is composed
- [x] #2 The tooltip never names an action that is not on screen
- [x] #3 The annotation, docstring and guide sentence match the code
- [x] #4 The blocked state paints the action name once
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete the dead `#library-conversation-use-source` handler on the controller + its screen delegator, and the characterization test that pinned it.
2. Extract ONE blocked-sentence helper in library_conversation_reader.py; the tooltip, the inline reason line and the 'c' toast all read it, so the blocked state paints the action name once (marker moves onto the button label).
3. Widen check_action for library_conversation_open_console so 'c' reaches the handoff and toasts that same sentence instead of being silent; keep the footer hint on the ready predicate.
4. Fix the stale media-and-conversations.md sentences (below-the-transcript, blocked copy, 'c' refuses) and stamp.
5. Confirm the reader-controller annotation/docstring already match the tuple return (fixed in a later PR round) and record it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Deleted the dead pair and collapsed the blocked hand-off onto one control.

- **Dead handler**: `use_selected_conversation_as_source` is gone from
  `library_conversations_controller.py` and its `library_screen.py`
  delegator; `#library-conversation-use-source` has no compose site
  anywhere, so nothing could reach it. The task-5 characterization pin that
  recorded the finding went with it, replaced by
  `test_use_as_source_handler_is_gone` (which also greps both modules for
  the id). The browse cluster is 39 names, not 40.
- **One action name**: the non-colour disabled marker moved onto the button
  label ("○ Open in Console"); the Static beneath it now carries the refusal
  SENTENCE instead of a second copy of the action name. Live-verified at
  235x52 on a seeded profile.
- **One sentence, three surfaces**: `library_conversation_block_sentence()`
  (module level in `library_conversation_reader.py`) is the single source
  for the tooltip, the inline line and the 'c' toast.
- **'c' is never silent**: `check_action` for
  `library_conversation_open_console` now gates on "a conversation is open"
  rather than "the hand-off is ready", and the action toasts that same
  sentence when blocked. This deliberately reverses task-32056 fix round 1
  (which stopped the key at the gate to kill a toast naming a workspace with
  nothing on screen to link into): the toast now repeats the visible copy,
  whose remedy IS on screen, and the alternative was a key that did nothing.
  The footer chip reads `_library_conversation_handoff_ready()` directly so
  it still advertises only what works.
- **Annotation**: `library_conversation_workspace_block` was already
  `Callable[[], tuple[str, bool, str]]` with a matching docstring (fixed in
  a later PR #2523 round) -- no change needed.
- **Guide**: `media-and-conversations.md` -- dropped the stale "'Open in
  Console' sits below the transcript" rough edge, replaced the two blocked-copy
  quotes with the shipped sentence, and rewrote the 'c refuses in exactly the
  same cases' claim.

Files: `tldw_chatbook/Widgets/Library/library_conversation_reader.py`,
`tldw_chatbook/Widgets/Library/__init__.py`,
`tldw_chatbook/UI/Library_Modules/library_conversations_controller.py`,
`tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_crit8_conversation_handoff.py`,
`Tests/UI/test_library_conversations_characterization.py`,
`Tests/Architecture/test_library_conversations_wiring.py`,
`Docs/User_Guide/library/media-and-conversations.md`.
<!-- SECTION:NOTES:END -->
