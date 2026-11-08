---
id: TASK-39157
title: "confirmation_dialog's _closed_but_still_stacked docstring says a cancelled push_screen_wait waiter cancels the dialog's future, which Textual 8.2.8 does not do"
status: To Do
created_date: 2026-11-08 06:54
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
- [ ] #1 The docstring says what Textual 8.2.8 does, and when a cancelled future can still reach this check, if it can
- [ ] #2 If no path can leave a cancelled future on a stacked screen any more, the notes say so; the check stays as it is, or is simplified with a test that shows `_finish_interrupted_closes` still pops a screen whose own close went astray
<!-- AC:END -->
