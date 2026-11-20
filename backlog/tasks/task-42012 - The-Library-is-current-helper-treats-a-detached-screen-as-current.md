---
id: TASK-42012
title: "The Library is-current helper treats a detached screen as current"
status: Done
assignee: ['@claude']
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 16:32
labels:
- follow-up
- ui
- library
priority: low
references:
- tldw_chatbook/UI/Library_Modules/screen_helpers.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41715 (round 17). `screen_helpers._library_screen_is_current` returns True when `app.screen` raises (`except Exception: return True`), so past the exit fence its 11 `library_screen.py` callers treat a detached screen as current.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The helper returns False for a detached or exiting screen
- [x] #2 A test covers the exit-fence case, red before the fix
<!-- AC:END -->

## Implementation Plan

1. Read every `_library_screen_is_current` caller in `library_screen.py` and check that a False answer at the exit fence (or for a detached screen) skips only work that is pointless or wrong once the screen is gone.
2. Make the helper require `screen.is_attached` (False once `App._exit` is set or the screen has left the DOM) and return False, not True, when reading `app.screen` raises.
3. Give the fake-screen tests that call these handlers a fake that is attached and current, so they keep testing what they tested.
4. Add a pilot test: a real Textual screen, the helper True while current, False once `app._exit` is set and once the stack is empty; red before the fix.

## Implementation Notes

`_library_screen_is_current` (`UI/Library_Modules/screen_helpers.py`) is now `screen.is_attached and screen.app.screen is screen`, and False when either read raises. `is_attached` is False once `App._exit` is set (shutdown sets it first, while the stack still holds the screen) and for a screen that has left the DOM; an empty stack makes `app.screen` raise `ScreenStackError`. Before, the fence case returned True (the stack still held the screen) and the raise case returned True from the `except`.

**The 11 callers, and why False is right for each** (all in `library_screen.py`; the import coordinator they drive is app-owned and keeps its draft in memory only, and a quit flushes a dirty skill editor through `flush_pending_work(quitting=True)`, not through any of these):

| Caller | Work skipped on False | Needed at exit? |
|---|---|---|
| `handle_library_skills_import` | opens the import draft, flushing the editor save first, and syncs the canvas | No: UI only; the quit path flushes the editor |
| `handle_library_skills_import_cancel` | dismisses the draft | No: in-memory draft dies with the app |
| `handle_library_skills_import_browse` | pushes `FileOpen` | No: pushing a modal onto an exiting app is wrong |
| its `browse_callback` | writes the picked path into the draft | No: in-memory |
| `handle_library_skills_import_browse_folder` | pushes `SelectDirectory` | No, as above |
| its `browse_callback` | writes the picked folder into the draft | No: in-memory |
| `handle_library_skills_import_review` | flushes the editor save, opens the imported skill's editor | No: the quit path flushes the editor |
| `handle_library_skills_import_path_changed` | copies typed text into the draft | No: in-memory |
| `handle_library_skills_import_retry` | claims the retry and starts an import worker | No: starting a write during shutdown is wrong |
| `_start_library_skills_import` | claims and starts an import worker | No, as above |
| `resolve` (candidate-choice modal callback) | claims or cancels the pending candidate | No: in-memory; a replaced screen already got False |

A replaced or suspended screen got False before (another screen is on top) and still does.

- **Test.** `Tests/UI/test_screen_derivation_helpers.py::test_library_is_current_is_false_past_the_exit_fence` (whole-file UI census entry): a real Textual screen, True while current, then read with `app._exit` set and with `app.screen` raising `ScreenStackError`. Before the fix: `assert (True, True) == (False, False)`. After: passes.
- **Fakes.** Six `test_library_skills_canvas.py` handler tests built a `SimpleNamespace` screen with no `is_attached` or `app.screen`, and passed only because the old helper said True on the error. They now opt in through `_as_current_screen(...)`; their assertions are unchanged.
- **Covering runs** (`-n 4`, load 19-26): the helper, skills-canvas, skills-import, modal-dismissal, trust-journey and crit8-wait files. Head and base fail the same 82 nodes, all but one on the local `RecoveryRequired` trap. The only head-only failures were the six fakes above, which now pass. The three real-screen import journeys in `Tests/Skills/test_skills_import.py` (`single_flight_across_every_route` x4, `routed_library_replacement_observes...`, `completed_import_receipt_survives...`) pass 6 of 6 under a temporary `@private_profile_test` marker (removed afterwards).
- **Size.** `screen_helpers.py` 401 -> 403; its `test_module_size_ratchet.py` row is raised by those 2 lines, with a dated comment.

Files: `tldw_chatbook/UI/Library_Modules/screen_helpers.py`, `Tests/UI/test_screen_derivation_helpers.py`, `Tests/UI/test_library_skills_canvas.py`, `Tests/Architecture/test_module_size_ratchet.py`.
