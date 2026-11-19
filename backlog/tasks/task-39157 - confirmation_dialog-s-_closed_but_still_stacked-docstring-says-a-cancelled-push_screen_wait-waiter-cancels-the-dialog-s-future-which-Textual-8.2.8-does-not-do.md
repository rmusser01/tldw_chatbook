---
id: TASK-39157
title: "confirmation_dialog's _closed_but_still_stacked docstring says a cancelled push_screen_wait waiter cancels the dialog's future, which Textual 8.2.8 does not do"
status: Done
assignee: ['@claude']
created_date: 2026-11-08 06:54
updated_date: 2026-11-10 16:20
labels:
- follow-up
- ui
- docs
- textual
priority: low
references:
- tldw_chatbook/Widgets/confirmation_dialog.py
dependencies:
- TASK-39146
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (cprod stream). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08, against the installed Textual 8.2.8. Line numbers are dev's; the module is the same on the stack.

- `_closed_but_still_stacked` (`Widgets/confirmation_dialog.py:361-374`) treats a resolved result future, not a cancelled one, as "closed but still stacked". Its docstring explains the `cancelled()` exclusion as what "a cancelled `push_screen_wait` waiter does to a screen still legitimately open".
- Textual 8.2.8's `App.push_screen_wait` awaits `asyncio.shield(self.push_screen(screen, wait_for_dismiss=True, mode=mode))` (`textual/app.py:2995-2999`). Its comment reads "The shield prevents the cancellation of the current task from canceling the push_screen awaitable". Cancelling its waiter leaves the future pending.
- Only a bare `await push_screen(..., wait_for_dismiss=True)` has no shield. TASK-39146 (round 12) moved the nine such waits onto `push_screen_wait` and added a guard, `test_no_unshielded_dismiss_waits`, that fails on a new one. It also corrected the same claim in `personas_screen.py`'s `_io_dialog_active` comment, but not this docstring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The docstring says what Textual 8.2.8 does, and when a cancelled future can still reach this check, if it can
- [x] #2 If no path can leave a cancelled future on a stacked screen any more, the notes say so; the check stays as it is, or is simplified with a test that shows `_finish_interrupted_closes` still pops a screen whose own close went astray
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read Textual 8.2.8's `push_screen` / `push_screen_wait` / `ResultCallback` and grep the package for every `wait_for_dismiss=True` future and every `_result_callbacks` reader, to find any path that can still cancel a stacked screen's result future.
2. Rewrite the `_closed_but_still_stacked` docstring to say what Textual does and when a cancelled future can still reach the check.
3. Keep the check unchanged (no behaviour change, so no new test); run the quit-prompt tests and the unshielded-wait guard.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docstring-only change in `tldw_chatbook/Widgets/confirmation_dialog.py` (`_closed_but_still_stacked`).

- What Textual 8.2.8 does (read in the installed `textual/app.py:2920-3000` and `textual/screen.py:107-142`): `push_screen` creates the future and hands it to the pushed screen's `ResultCallback`; only that screen's `dismiss` resolves it (`ResultCallback.__call__` -> `set_result`), and no Textual code cancels it. `push_screen_wait` awaits it through `asyncio.shield`, so cancelling its waiter leaves the future pending.
- When a cancelled future can still reach the check: only when a caller awaits a bare `push_screen(..., wait_for_dismiss=True)` future (or wraps it in `asyncio.wait_for`) and is cancelled. The package has no such wait now: the only `wait_for_dismiss=True` push is `await_quit_prompt`, which waits with `asyncio.wait` (never cancels; allowlisted in `UNSHIELDED_DISMISS_WAIT_ALLOWLIST`), and `test_no_unshielded_dismiss_waits` fails on any new one. No package code calls `.cancel()` on a `_result_callbacks` future (`app_keep_alive._release_pending_result`, `app_destinations`, `app_navigation` only read or restore them).
- AC #2: no path can leave a cancelled future on a stacked screen today. The check stays as it is: the `not future.cancelled()` exclusion is harmless and keeps a never-dismissed screen from being popped if a bare wait ever comes back, so the docstring now calls it a guard, not a live path. No behaviour changed, so no new test.
- Verified: `Tests/UI/test_quit_prompt_vanish.py` + `Tests/Architecture/test_worker_exclusive_group_inventory.py -k 'vanish or dismiss'`, 14 passed (`-n 4`, load ~13).
<!-- SECTION:NOTES:END -->
