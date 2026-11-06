---
id: TASK-38980
title: "Response-rules Promote runs inside its confirmation dialog with the manager not busy, so Ctrl+Q quits mid-write and a queued Escape closes the manager"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- bug
- console
- response-rules
- quit
priority: low
references:
- tldw_chatbook/Widgets/Console/response_rules_modal.py
- tldw_chatbook/Widgets/confirmation_dialog.py
- Tests/UI/test_response_rules_modal.py
- Tests/UI/test_modal_quit_in_flight_hooks.py
- Tests/Architecture/test_guarded_modal_quit_census.py
dependencies:
- TASK-38747
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b on 2026-11-05; the Ctrl+Q part was read from the TASK-38747 branch, which merges first.

**How Promote runs.**
- Promote pushes a ConfirmationDialog with `confirm_callback=commit` (`tldw_chatbook/Widgets/Console/response_rules_modal.py:556-587`).
- `ConfirmationDialog.on_button_pressed` awaits that callback before `dismiss(True)` (`tldw_chatbook/Widgets/confirmation_dialog.py:153-155`).
- So the promote write runs inside the dialog's button handler. `_busy = True` is set only for the manager's other actions (`:589`).

**Probe on dev, with the promote write held open.**
- `_busy` was False. The screen stack was `['Screen','ResponseRulesModal','ConfirmationDialog']`.
- An Escape pressed during the write was still unprocessed after 3 s, because the dialog's handler was blocked.
- Once the write was released, the stack went straight to `['Screen']`: the queued Escape closed the manager as soon as the promote landed. The binding was written (revision 1).

**Ctrl+Q after TASK-38747 merges (from reading its branch).**
- TASK-38747 adds `ResponseRulesModal.confirm_quit` (`response_rules_modal.py:712-738` on that branch). It refuses only when `_busy` is set; otherwise it asks only about unsaved edits.
- ConfirmationDialog has no `confirm_quit`.
- So Ctrl+Q during a promote write quits without asking.
- TASK-38747 covers Test, Save and the pin changes (Disable/Enable, Exclude, Delete local pin), but not Promote.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While a confirmed Promote write is in flight, Ctrl+Q answers "still working" as it does for Test, Save and pin changes; the test is red against the current Promote flow
- [ ] #2 While the write is in flight the manager shows that the promote is running, its action buttons are disabled, and Escape does not close the manager before the write settles
- [ ] #3 After the write, the manager shows the promote's outcome (promoted, refused or failed), including when Escape was pressed during the write
- [ ] #4 The guarded-modal quit census and Tests/UI/test_modal_quit_in_flight_hooks.py stay green
<!-- AC:END -->
