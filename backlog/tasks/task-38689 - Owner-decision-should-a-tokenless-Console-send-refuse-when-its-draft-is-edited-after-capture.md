---
id: TASK-38689
title: "Owner decision: should a tokenless Console send refuse when its draft is edited after capture?"
status: To Do
created_date: 2026-11-03 11:40
labels:
- follow-up
- console
- needs-owner-decision
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/wiring.py
- tldw_chatbook/UI/Console_Modules/hooks.py
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/UI/test_console_send_draft_snapshot.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in the round-5 wave of the 2026-10-23 burn-down (TASK-36520, PR #3376, 2026-11-03). Checked on dev 9bbd5c2515.

Two rules on the Console send path disagree about text typed after a send captures its draft.
- **hooks.py:430-440 (TASK-340).** After the hook snapshot read, "text typed after the capture belongs to the next draft ..., not a reason to refuse this send". hooks.py refuses an edited draft only when a hook review actually ran.
- **wiring.py:370-375.** `receive_console_visible_intent` arrived with owner commit 3e618d7119 ("wip: checkpoint reviewed Console preparation integration", 2026-10-15). When the send carries no captured inputs, it compares the live composer's generation and edit serial with the captured stash, and it refuses any difference with "Draft changed; Send again.".

**Which sends reach the wiring check.**
- Enter does not. It carries a pending-send token, and the token's received inputs go through as `_captured_inputs` (chat_screen.py:18320-18341, :18486-18489).
- Sends without a token do: a mouse or Workbench Send (chat_screen.py:18318, "Mouse/Workbench sends have no token"), and the voice command "Console, send.", which calls `handle_console_send_message` (chat_screen.py:10379).

When the stash is captured just before the check (composer_run_controls.py:1031-1039), no edit can land in between. When it is captured before an await, typing during that await is refused. `Tests/UI/test_console_send_draft_snapshot.py::test_console_armed_unknown_mouse_send_snapshots_before_skill_await` is such a case: its docstring says later typing is preserved and the send proceeds. On dev it fails with `['Draft changed; Send again.']` (2026-11-03, `-n 4`). PR #3376 isolates the test from the wiring check and leaves product code as it is.

No test asserts the "Draft changed; Send again." refusal.

**Question for the owner.** When a tokenless send's draft is edited after the send captured it, which should happen?
- (a) The send is refused with "Draft changed; Send again." (wiring.py today).
- (b) The captured text is sent, and the later typing stays as the next draft (hooks.py's TASK-340 rule).
- (c) Something else.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's answer is recorded in this task
- [ ] #2 Tokenless sends (button, Workbench and voice) follow that answer, and a test pins it, including the refusal copy if refusal stays
<!-- AC:END -->
