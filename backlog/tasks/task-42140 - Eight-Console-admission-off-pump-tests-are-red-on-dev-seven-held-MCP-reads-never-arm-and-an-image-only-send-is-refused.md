---
id: TASK-42140
title: "Eight Console admission off-pump tests are red on dev: seven held MCP reads never arm, and an image-only send is refused"
status: To Do
assignee: []
created_date: 2026-11-20 20:06
updated_date: 2026-11-20 20:45
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_send_admission_off_pump.py
- Tests/UI/test_console_send_acknowledgement.py
- tldw_chatbook/Chat/console_configuration_capture.py
- tldw_chatbook/UI/Console_Modules/wiring.py
- tldw_chatbook/UI/Console_Modules/composer_run_controls.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the consoletests stream (round 18). At d85e90bc9e eight nodes of `Tests/UI/test_console_send_admission_off_pump.py` fail locally (two runs, `-n 4`, through the test wrapper, load 11-20). They have two causes.

**Seven `HeldMcpRead` nodes.** Six functions create a `HeldMcpRead` (lines 120, 218, 344, 383, 476 and 591), seven nodes in all: `test_a_key_typed_while_admission_reads_mcp_is_handled_before_the_read_returns`, `test_enter_during_admission_is_replayed_after_it_as_the_held_pump_did`, `test_an_enter_deferred_behind_a_send_never_sends_another_tab_draft`, `test_leaving_a_tab_while_its_send_is_admitted_still_sends_it_there`, `test_an_enter_made_while_the_sent_draft_still_shows_never_takes_later_keys` and `test_a_bare_second_enter_never_sends_a_committed_draft_again[alone|then_y]`. Each times out in `until` (`Tests/UI/test_console_send_acknowledgement.py:278`) waiting for `hold.entered`: the held MCP read is never entered. The likely cause: `HeldMcpRead` (line 56) holds `get_kill_switch` only when the calling frame is named `capture_mcp_definition_maximum`. At the base that function (`tldw_chatbook/Chat/console_configuration_capture.py:271`) delegates to `_capture_mcp_definition_selection`, which makes the call (`:326`), so the name never matches. The split looks like 11d71e5db5 (2026-11-11).

**The image-only node.** `test_an_image_only_enter_never_takes_text_typed_after_it` (line 652) uses no `HeldMcpRead`. It stages a fake attachment (`console._console_pending_image_attachment = lambda: object()`), replaces `console._hooks.dispatch` with a stub, and fails after about 20 s at line 684 on its own `until(lambda: bool(dispatched) and composer.draft_text() == "hi")`. The Enter takes the received-intent route: `receive_console_visible_intent` (`tldw_chatbook/UI/Console_Modules/wiring.py:389`), with `awaiting_review` set at `UI/Console_Modules/composer_run_controls.py:1055`. That route never calls `self._hooks.dispatch` (`:1116`), so the stub would stay uncalled even if the submit were accepted. Here it is not: the log shows `phase=ui_dispatch status=awaiting_review`, then `console_chat_controller.submit_draft` refusing it ("Console submit requires attention | accepted=False status=refused"; `phase=controller_submit status=refused`). Whether the fake `object()` attachment causes the refusal is not yet known.

Only `test_an_enter_deferred_behind_a_send_sends_its_own_capture_only` of this file is in the UI census, and it passes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The cause of the seven `HeldMcpRead` failures is named and fixed, so the hold is entered on admission's MCP read and the seven nodes pass locally; the notes record the runs and the load
- [ ] #2 If admission no longer makes the read a hold waits for, the test fails naming that read, not only with a timeout
- [ ] #3 The image-only node's cause is named, both the route its Enter takes (which bypasses the stubbed `_hooks.dispatch`) and the check that refuses its submit, and it is fixed in the test or the product, whichever the cause shows, so the node passes
- [ ] #4 The eight nodes run in a PR lane
<!-- AC:END -->
