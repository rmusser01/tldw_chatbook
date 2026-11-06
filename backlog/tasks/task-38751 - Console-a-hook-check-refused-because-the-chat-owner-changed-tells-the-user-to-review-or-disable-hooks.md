---
id: TASK-38751
title: "Console a hook check refused because the chat owner changed tells the user to review or disable hooks"
status: Done
assignee:
- '@claude'
created_date: 2026-11-04 06:34
updated_date: '2026-11-06 00:02'
labels:
- follow-up
- console
- ux
- hooks
priority: medium
references:
- tldw_chatbook/Chat/console_automatic_submission.py
- tldw_chatbook/Chat/console_chat_controller.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

`read_hook_admission_result` (`Chat/console_automatic_submission.py:449`) runs the hook preparation read for a send. After the read, it checks that the hook admission owner has not changed: the chat session and its binding revision, the runtime, and its hook permissions. If any of them moved, it raises `RuntimeError("Console hook admission owner changed.")` (:546).

The function's catch-all (:559-562) turns every exception into "Hooks unavailable; review or disable hooks before sending." So a send refused because the user switched chats during the read, or because the Console runtime was disposed, tells them to review or disable their hooks. Nothing is wrong with the hooks, and disabling them is the wrong advice. The same copy is also the generic fallback of `hook_admission_reason` (:400).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A send refused because the hook admission owner changed shows copy that says what happened (the chat changed during the hook check) and what to do (send again), with no advice about hooks
- [x] #2 A real failure of the hooks read still shows the review-or-disable copy
- [x] #3 A test changes the owner during the preparation read and checks the copy; it is red before the change
- [x] #4 Any User Guide page that describes hook send refusals matches the new copy
- [x] #5 A send whose v2 hook preparation is voided because the chat changed shows the same send-again copy, while a real hook initialization failure keeps "Required hook initialization failed."; a test is red before the change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a test that runs the stock admission read for a bound session and closes the chat inside the permission read; show it red on the old copy.
2. Give the owner-change check its own exception type and map it to new copy in `read_hook_admission_result`, keeping every other failure on the review-or-disable copy.
3. Add a test for a real hooks-read failure, and describe both refusals in the guide's hooks section.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- `Chat/console_hook_preparation.py` defines `HookCheckOwnerChanged`, a `RuntimeError` subclass that keeps its old messages. Two owner checks raise it: the admission check in `read_hook_admission_result` ("Console hook admission owner changed.") and the v2 source check `require_hook_preparation_source` ("Console hook preparation owner changed.").
- `Chat/console_automatic_submission.py` maps it in both places to "Chat changed during the hook check; send again.", which says what happened and what to do and gives no hooks advice. Everything else keeps its copy: the admission read's other failures and `hook_admission_reason`'s own fallback still return "Hooks unavailable; review or disable hooks before sending." (now one module constant), and every other v2 preparation failure still returns "Required hook initialization failed."
- The owner check also runs after the callback returns (`run_preparation_read` calls `require_current` after `callback()`). So a chat that changes during the inner authority read gets the new copy too, even when the callback swallowed the inner workspace check. This follows from reading the code; no separate test covers it. The workspace check inside the v2 context-key read ("Console hook workspace ownership changed.") is unchanged and still shows the initialization copy.
- Tests are in `Tests/Chat/test_console_viewless_hooks.py`, which runs whole in derived-artifacts' admission-sensitive step:
  - `test_a_chat_closed_during_the_hook_check_says_send_again` closes the session through `ConsoleChatStore.close_session` during the admission read. Red before the fix (`'Hooks unavailable; review or disable hooks before sending.' == 'Chat changed during the hook check; send again.'`), green after.
  - `test_a_failed_hooks_read_still_says_review_or_disable_hooks` makes the accessor raise `OSError`. Green before and after (AC#2).
  - `test_a_chat_closed_during_v2_hook_preparation_says_send_again` (AC#5) pauses `HookPermissions.v2_configuration` inside `prepare_submission_hooks`, closes the chat on the event loop, then releases the read. Red before the mapping (`'Required hook initialization failed.' == 'Chat changed…'`), green after. Negative control: with the v2 raise turned back into a plain `RuntimeError`, the test goes red again.
  - Real initialization failures: `test_console_turn_resend.py::test_resend_runs_granted_required_hooks_before_clear[SessionStart-*]` (a required hook exits 1) still asserts "Required hook initialization failed." and passes.
- AC#4: no User Guide page quoted either copy. The hooks section of `console/agent-runs-and-tools.md` now quotes both, and names closing the chat as the trigger. Saving a temporary chat would also change the session identity, but that was not tested, so the guide does not claim it.
- Covering run (`-n 4`, load ~16): 307 passed, 5 failed across viewless hooks, hook preparation lifetime/reads, both read-sharing suites, preparation scheduling, turn resend, prompt mix-in admission, agent swap, received custody, the pre-import payload budget and first-send pre-import. The 5 failures also fail at base 29592ecc9b with the same command: agent swap's 2 stop/cancel nodes and turn resend's 3 refused-echo/vision-gate nodes.
<!-- SECTION:NOTES:END -->
