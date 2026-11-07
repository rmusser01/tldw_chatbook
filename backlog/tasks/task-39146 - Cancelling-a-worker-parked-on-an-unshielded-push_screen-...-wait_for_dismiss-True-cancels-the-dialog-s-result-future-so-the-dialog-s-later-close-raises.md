---
id: TASK-39146
title: "Cancelling a worker parked on an unshielded push_screen(..., wait_for_dismiss=True) cancels the dialog's result future, so the dialog's later close raises InvalidStateError and leaves it on the stack"
status: To Do
created_date: 2026-11-07 02:28
labels:
- follow-up
- ui
- textual
priority: low
references:
- tldw_chatbook/UI/ChatbookExportManagementWindow.py
- tldw_chatbook/UI/STTS_Window.py
- tldw_chatbook/UI/Chatbooks_Window_Improved.py
- tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py
- tldw_chatbook/Widgets/Media/media_viewer_panel.py
- tldw_chatbook/UI/Screens/personas_screen.py
- Tests/Architecture/test_quit_flow_prompt_choke_point.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down, while checking a modals follow-up; checked on dev 2684e5453f with Textual 8.2.8, 2026-11-07. Line numbers are dev's.

**Where it goes wrong.**
- `App.push_screen_wait` awaits `asyncio.shield(self.push_screen(..., wait_for_dismiss=True))`, so cancelling its worker leaves the dismiss future pending, and a later close resolves it normally.
- An unshielded `await app.push_screen(screen, wait_for_dismiss=True)` has no shield. Cancelling the waiting worker cancels the screen's result future.
- A Textual probe on dev shows the result: the dialog's later dismiss raises `InvalidStateError: invalid state`, and the dialog stays on the screen stack (`['Screen', 'Editor']`). The shielded `push_screen_wait` raised nothing in the same probe.

**The 9 unshielded waits on dev:**
- `UI/ChatbookExportManagementWindow.py:1093`;
- `UI/STTS_Window.py:675` and `:784`;
- `UI/Chatbooks_Window_Improved.py:825`, `:876`, `:885` and `:899`;
- `UI/CCP_Modules/ccp_character_handler.py:1065`;
- `Widgets/Media/media_viewer_panel.py:2054`.

**Not affected.** `Widgets/confirmation_dialog.py:339` waits through `asyncio.wait`, which never cancels the future.

**Related records.**
- The comment at `UI/Screens/personas_screen.py:960-964` attributes this error to `push_screen_wait` "on Textual 8.2.7". Under 8.2.8 that does not hold.
- TASK-33621.25 (To Do) covers wait pushes outside a worker (W003), not cancellation.

**Why it matters.** A worker cancelled at shutdown or by an exclusive restart leaves a dialog the user cannot close cleanly.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 For each of the 9 sites, cancelling its waiting worker and then closing the dialog raises nothing and leaves the dialog off the stack; a test is red on dev
- [ ] #2 A guard fails when a new unshielded `push_screen(..., wait_for_dismiss=True)` await is added, shown red by a negative control
- [ ] #3 The `personas_screen.py` comment states what holds under the pinned Textual version
<!-- AC:END -->
