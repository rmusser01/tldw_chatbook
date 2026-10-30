---
id: TASK-37960
title: Closed Console row menus keep claiming Esc
status: Done
assignee: []
created_date: '2026-10-29 12:18'
updated_date: '2026-10-29 14:26'
labels:
  - console
  - keyboard-focus
  - ux-review-2026-09-29
dependencies:
  - TASK-33622.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The conversation- and workspace-row action menus defined a private method named _detach, which shadows the Textual hook that clears a removed widget's parent. A closed row menu therefore kept its parent, the menu registry kept reporting it as open, and the Console's Esc dismissed that ghost instead of returning to the composer. The ghost lasts for as long as anything still references the menu, which for a long-lived object means until a full garbage collection. Found while testing TASK-33622.3's menu fold; it breaks TASK-33622.4's Esc-to-composer rule after any row menu has been used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a conversation-row or workspace-row menu closes, the menu registry no longer reports it, even while the menu object is still referenced
- [x] #2 After a row menu closes, Esc from the transcript returns focus to the composer
- [x] #3 A Pilot test on the real ChatScreen opens and closes each row menu, then presses Esc from the transcript and asserts the composer has focus; it fails on the pre-fix code
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Renamed the conversation- and workspace-row action menus' memoized removal helper from _detach to _remove_once. _detach is Textual's MessagePump hook that clears a removed node's parent (Widget._message_loop_exit calls it); the override returned the AwaitRemove instead, so a closed menu kept its parent, conversation_/workspace_action_menus_on_screen kept reporting it, and ChatScreen's Esc (action_focus_console_composer_home -> _dismiss_console_conversation_action_menus) counted that ghost and returned without focusing the composer. The ghost lives as long as anything references the menu (a long-lived object until a full GC). Found while pinning TASK-33622.3's menu fold.

- Tests: Tests/UI/test_console_key_ownership.py::test_a_closed_row_menu_does_not_eat_a_later_esc[conversation|workspace] (real ChatScreen: mount the menu, Esc closes it, the test keeps a reference, asserts parent None and an empty registry, then F6 to the transcript and Esc expects the composer). Before the rename both failed with "Esc from the transcript left focus on ConsoleTranscript"; on dev 2ee6f1a473 they fail with "the closed menu kept its parent".
- Live (real app, Anthropic claude-haiku-4-5 via the review proxy, isolated profile, 160x45): the Chat 2 row menu opened from its button, Esc closed it, F6 to the transcript, Esc returned to the composer and typing landed in the draft (evidence bd5/U10/r3/14-17). Live cannot show the ghost deterministically, since it ends at the next full GC; the test is the red/green.
- Regression check: Tests/UI/test_console_conversation_action_menu.py (30 passed), test_console_row_menu_action_sweep.py (29 passed), test_console_manual_unread.py (14 passed) on head and dev alike; test_console_workspace_action_menu.py fails 16 at setup locally (RecoveryRequired) on both, and with a scratch bootstrap_profile marker the same 3 ids fail on head and dev.
- Files: tldw_chatbook/Widgets/Console/console_conversation_action_menu.py, tldw_chatbook/Widgets/Console/console_workspace_action_menu.py, Tests/UI/test_console_key_ownership.py.
- Follow-up: TASK-37622 (MeetingsScreen shadows the same hook).

Merge evidence (2026-10-29): branch head rebased onto dev 38c9d2edcf, 0 commits behind. Tests/UI/test_console_key_ownership.py: 24 passed under the plain command on the rebased head; the same file on dev 38c9d2edcf: 23 failed, 1 passed (the Space/text-field guard), each for its stated reason. Touched modules' test files (row menus, selection, composer, transcript, hands-free/realtime, internals, native chat flow and others, 48 files) compared by test id against a detached dev worktree: the only reds new to the branch that were not rotating load flakes were the two composer-collapse tests fixed in this branch (both 8/8 on head and dev afterwards); every other head-only red passed on a solo rerun or failed at the same rate on dev. Module-size ratchet: the same 8 failures as dev (composer and transcript rows raised with dated owner-rule comments); screen-size ratchet green (ChatScreen 24337/737 exact); ui-ready module census and boot CSS byte budget green; ./scripts/preflight.sh passes (UI gate census 1148/1148).
<!-- SECTION:NOTES:END -->
