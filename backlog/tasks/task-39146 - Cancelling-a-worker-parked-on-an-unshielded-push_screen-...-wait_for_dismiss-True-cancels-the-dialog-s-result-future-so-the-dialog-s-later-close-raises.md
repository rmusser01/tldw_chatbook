---
id: TASK-39146
title: "Cancelling a worker parked on an unshielded push_screen(..., wait_for_dismiss=True) cancels the dialog's result future, so the dialog's later close raises InvalidStateError and leaves it on the stack"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-07 06:09'
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
- [x] #1 For each of the 9 sites, cancelling its waiting worker and then closing the dialog raises nothing and leaves the dialog off the stack; a test is red on dev
- [x] #2 A guard fails when a new unshielded `push_screen(..., wait_for_dismiss=True)` await is added, shown red by a negative control
- [x] #3 The `personas_screen.py` comment states what holds under the pinned Textual version
<!-- AC:END -->

## Implementation Plan

1. The shared helper already exists: Textual's `App.push_screen_wait` awaits the same `push_screen(..., wait_for_dismiss=True)` future through `asyncio.shield` (170 call sites use it). Move the 9 bare awaits onto it; no new wrapper.
2. One parametrized test drives each of the 9 sites in a worker (dialog classes swapped for a plain modal), cancels the worker while the dialog is open, then closes the dialog; red on base with `InvalidStateError`.
3. Guard: an AST scan in `Tests/Architecture/test_worker_exclusive_group_inventory.py` (architecture-guards lane) fails on any `await <x>.push_screen(..., wait_for_dismiss=<not False>)`; a synthetic-source test pins the flagged forms, and the guard is shown red on base and under a reverted site.
4. Rewrite the `personas_screen.py` `_io_dialog_active` comment for Textual 8.2.8.

## Implementation Notes

The shared helper is Textual's own `App.push_screen_wait`, which awaits the same `push_screen(..., wait_for_dismiss=True)` future through `asyncio.shield` and is already used at 170 sites. All 9 bare awaits now use it; no new wrapper was written.

- **Sites.** `ChatbookExportManagementWindow._delete_selected`; `AudioBookGenerationWidget._import_from_notes_worker` and `_import_from_conversation_worker` (`STTS_Window.py`); `ChatbooksWindowImproved.action_create_chatbook`, `action_browse_templates` (both waits), `action_manage_exports`; `CCPCharacterHandler.handle_import`; `MediaViewerPanel._run_delete_confirmation`.
- **Behaviour test.** `Tests/UI/test_cancelled_dialog_waits.py` drives each site in a worker (its dialog classes swapped for a plain `ModalScreen`; the subject is the wait), cancels the worker while the dialog is open, then closes it, and asserts the close raises nothing and the dialog leaves the stack. The template-wizard case answers the templates dialog first to reach the second wait. Before the fix all 9 cases failed with `InvalidStateError: invalid state` from `Screen.dismiss`; after it, 9 passed (5.2-5.6 s serial wall, 3 runs, load 26-37). Appended to the UI census tail.
- **Guard.** `Tests/Architecture/test_worker_exclusive_group_inventory.py::test_no_unshielded_dismiss_waits` (architecture-guards lane) flags any `await <x>.push_screen(...)` whose `wait_for_dismiss` (keyword, or third positional) is not literal `False`. `confirmation_dialog.await_quit_prompt` keeps its future and waits with `asyncio.wait`, so it is not flagged. `test_unshielded_dismiss_wait_forms_are_flagged` pins the flagged and unflagged forms on synthetic source. Negative controls: the guard's scanner run over the base tree lists exactly the 9 sites; reverting `_delete_selected` to the bare await on head turned the guard red (`ChatbookExportManagementWindow.py:1093`), restored afterwards.
- **Comment.** The `_io_dialog_active` comment in `personas_screen.py` now says what holds on Textual 8.2.8: `push_screen_wait` shields the future, so a cancelled worker's dialog stays open and closes cleanly, but nothing reads its answer, which is why a second request is still refused.
- **Mock seams.** Three tests that faked `push_screen(..., wait_for_dismiss=...)` now fake `push_screen_wait` (`test_chatbook_action_recovery_tooltips.py`, `test_chatbook_management_server_jobs.py`, `test_ccp_handlers.py`). The tooltip test's fake app has no `push_screen` at all, so a bare wait would now fail it. The first two are plain tests that hit the local RecoveryRequired trap; both passed under a temporary `@private_profile_test` marker, which was then removed.

**Runs.** Covering files (the new file, the three seam files, Chatbooks screen server actions, STTS import scope services, file-picker tooltips, tier-2 media guards, quit-flow choke point, worker-contract script tests): 432 passed, 20 failed on head; the same 20 fail on base 8b762b52fa with the same command (RecoveryRequired in the Chatbooks files, `test_display_messages_batches_mount`). `test_worker_exclusive_group_inventory.py`: 25 passed. The W003 wait census counts `push_screen_wait` and `wait_for_dismiss=True` alike, so its rows are unchanged.

**Lane cost** (serial, 3 runs, load 43-58): `test_no_unshielded_dismiss_waits` 4.2-6.8 s (it parses the package files that name `push_screen`), `test_unshielded_dismiss_wait_forms_are_flagged` 0.7-1.2 s. The Architecture guard contracts step ran 403 passed in 408 s at load 45-60.


**Review r12p-rev1 M4 (2026-11-08).** The guard flagged only a `push_screen(..., wait_for_dismiss=...)` awaited directly. It missed:
- a stored future awaited later (`fut = app.push_screen(...); await fut`);
- `await asyncio.wait_for(app.push_screen(..., wait_for_dismiss=True), 5)`;
- a `**{"wait_for_dismiss": True}` splat.

It now flags any `push_screen` call that may pass a non-False `wait_for_dismiss` (keyword, third positional argument, a `**` dict literal, or an unprovable `**` splat, which fails closed), unless the call is the direct argument of `asyncio.shield(...)`. `UNSHIELDED_DISMISS_WAIT_ALLOWLIST` holds the one deliberate site, `confirmation_dialog.py::await_quit_prompt` (it waits with `asyncio.wait`), with a staleness check.
- The package scan still finds no other production site.
- The forms test flags lines `[2, 3, 7, 8, 10, 11, 12]`; cebb6681b1's scanner flagged only `[2, 3, 7]` on the same source.
- Negative control: with the allowlist entry renamed, the guard went red on `confirmation_dialog.py:339`.


**Review r12p-rev2 M3 (2026-11-08).** A `*args` splat could carry the third positional `wait_for_dismiss`. `_waits_for_dismiss` now treats any starred argument as unprovable and fails closed, as it does for a `**` splat. The forms test adds `await app.push_screen(*args)`, so the expected lines are now `[2, 3, 7, 8, 10, 11, 12, 13]`. With the new check removed, the forms test goes red. The package scan still finds only the allowlisted `await_quit_prompt`.

Files: the six product modules above, `UI/Screens/personas_screen.py`, the four test files, `Tests/Architecture/test_worker_exclusive_group_inventory.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
