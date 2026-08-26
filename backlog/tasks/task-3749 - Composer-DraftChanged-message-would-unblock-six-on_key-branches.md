---
id: TASK-3749
title: Composer DraftChanged message would unblock six on_key branches
status: Done
assignee: []
created_date: '2026-08-08 21:06'
updated_date: '2026-08-08 23:37'
labels:
  - refactor
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 task 1 moved 7 of on_key's composer branches into ConsoleComposerBar. Eleven more could not move because they call screen methods after editing the draft: six of those call _sync_console_workbench_actions_from_draft (workbench state + slash-command popup) and _dismiss_console_guidance (repaints the transcript). If the composer posted a DraftChanged message that the screen subscribed to, those six branches would become composer-only and could move, taking the keymap with them. This is a DESIGN change, not an extraction, which is why wave 5 did not do it: an extraction that also changes how components communicate gives a regression two candidate causes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The composer notifies the screen of draft changes rather than the screen calling back after each edit
- [x] #2 The six blocked branches move to ConsoleComposerBar.handle_console_key
- [x] #3 No behaviour change: the workbench actions, slash-command popup and guidance repaint still update on every draft edit
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`ConsoleComposerBar` now posts a `DraftChanged` message; `ChatScreen._handle_console_composer_draft_edit` subscribes and does the Workbench resync (and, for insertions, the guidance dismissal) the six branches used to do inline. Those six -- backspace/ctrl+h, delete, ctrl+w, shift+enter/ctrl+j, ctrl+u, printable fallthrough -- moved into `handle_console_key`, taking `_is_modified_chord` with them. `chat_screen.py` sheds ~85 lines (the six branch bodies plus the `_is_modified_chord` block) and gains a ~53-line subscriber: net -32 lines, 18,595 -> 18,563. The commit message for 87a9d4bca says "~90", which was the gross removal, not the net; `on_key` keeps only the branches that genuinely reach past the composer (clipboard, send, paging, undo/redo's store persistence).

Three design decisions, each settled from measured baseline behaviour:

1. **One message with an `is_insertion` flag**, not two messages and not a bare notification. Guidance is dismissed at baseline by the two keys that ADD text and by none of the four that remove it; "dismiss on every DraftChanged" would have silently changed backspace/delete/ctrl+w/ctrl+u. The flag names the cause (the edit added text) rather than the effect, so the "insertions retire the first-run guidance" policy stays on the screen.
2. **Posted from the moved branches only**, never from `insert_text`/`delete_left`/`clear_draft`. Verified those helpers have other callers -- `Console_Modules/dictation.py`, `Console_Modules/session.py`, `/prompt`, paste -- each with its own different follow-up (dictation also re-persists to the chat store; the session restore deliberately does not dismiss guidance), so posting from the helpers would have changed all of them.
3. **Ordering: async delivery is a real, measured difference** -- scoped, not papered over. No existing test and no human-speed input depends on the old synchronous timing (471 composer/popup/dictation tests green). A key arriving in the SAME driver read as the slash that opens the popup does: two-arm A/B (this code vs. the baseline callback restored synchronously in its place) shows batched slash+Down used to highlight the second entry and now ignores the Down, and batched slash+Enter used to accept the highlighted suggestion and now ignores the Enter. In both arms the popup still opens, the draft is untouched and nothing is sent -- one lost keystroke under programmatic input (tmux send-keys, macros), never a wrong send. Kept as a strict xfail that will fail loudly when fixed, and filed as TASK-3790. AC #3 therefore holds for every drained observation and for all human-speed input, with that one batched-input exception documented rather than hidden.

Defect found and fixed en route, caught by the existing suite: the new handler was first named `_on_console_composer_draft_changed`, which is ALREADY a `ChatScreen` method (`@on(Input.Changed, "#console-command-input")`, disarming the unknown-command Enter escape). The later definition silently replaced the earlier one and killed that subscription -- two `test_console_command_composer.py` tests caught it. Renamed to `_handle_console_composer_draft_edit`, with both docstrings now cross-referencing each other and explaining why the broader `Input.Changed` signal is deliberately NOT reused here (it fires on load_draft/paste/dictation/session-restore, which do not sync the Workbench or dismiss guidance today).

Verification: 21 characterisation tests written and committed GREEN before the change (`Tests/UI/test_console_composer_draft_changed.py`, real pilot key presses on the real Console screen, asserting end state). After: 471 passed + 1 xfailed across the composer, popup, command-composer, dictation and decomposition suites; 411 passed across the workbench/guidance/send-state suites. Two `test_console_native_chat_flow.py` toast failures were A/B'd against a pre-change worktree and are pre-existing on dev. pyflakes unchanged (26 = 26, identical message set) on `chat_screen.py`, 0 on the composer and the new test. AST duplicate-method audit clean. No hand-built `ChatScreen` fixtures (`ChatScreen.__new__`/`spec=ChatScreen`) exist. Full-suite `--collect-only`: 33,536 tests, no import breakage.

Files: `tldw_chatbook/Widgets/Console/console_composer_bar.py`, `tldw_chatbook/UI/Screens/chat_screen.py`, `Tests/UI/test_console_composer_draft_changed.py` (new).
<!-- SECTION:NOTES:END -->
