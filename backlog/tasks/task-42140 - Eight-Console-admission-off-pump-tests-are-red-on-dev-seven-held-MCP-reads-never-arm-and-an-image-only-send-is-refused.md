---
id: TASK-42140
title: "Eight Console admission off-pump tests are red on dev: seven held MCP reads never arm, and an image-only send is refused"
status: Done
assignee:
- '@claude'
created_date: 2026-11-20 20:06
updated_date: 2026-11-21 19:12
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
- [x] #1 The cause of the seven `HeldMcpRead` failures is named and fixed, so the hold is entered on admission's MCP read and the seven nodes pass locally; the notes record the runs and the load
- [x] #2 If admission no longer makes the read a hold waits for, the test fails naming that read, not only with a timeout
- [x] #3 The image-only node's cause is named, both the route its Enter takes (which bypasses the stubbed `_hooks.dispatch`) and the check that refuses its submit, and it is fixed in the test or the product, whichever the cause shows, so the node passes
- [x] #4 The eight nodes run in a PR lane
<!-- AC:END -->

## Implementation Plan

1. Probe which frame calls `get_kill_switch` during admission now, and name the cause of the seven `HeldMcpRead` timeouts.
2. Fix `HeldMcpRead` to hold admission's read where it is now made, and give it a wait that fails naming that read (and the reads it did see) instead of a bare timeout.
3. Trace the image-only node: the route its Enter takes and the check that refuses its submit; fix the test or the product as the cause shows.
4. Lane the whole file (replace its single-node census entry) and measure it with the UI lane's flags.

## Implementation Notes

**Cause of the seven `HeldMcpRead` failures.** The stale frame name (the call moved from `capture_mcp_definition_maximum` into `_capture_mcp_definition_selection`, 11d71e5db5) was not the whole story: with the name corrected, a probe saw no `get_kill_switch` call at all ("reads seen from: []"). `HeldMcpRead` replaces `get_kill_switch` on the service instance, which makes the MCP service non-stock: `standard_console_sources(service)` is False (the source check requires the original bound method, since 3e618d7119). A non-stock source leaves the owned capture for the custom-capture fallback, `turn_admission.capture_authority` (`UI/Console_Modules/turn_admission.py:189`), which since 11d71e5db5 refuses any app with a managed Console runtime. So the send was refused with "Console preparation source unavailable; Send again." about 176 ms after Enter (`phase=ui_dispatch status=refused`), before any MCP read. A probe that tagged each of the five sites raising that copy showed the notice came from turn_admission.py:189. The product is right (a replaced reader must not run); the hold was wrong.

**Fix (tests).** The six functions hold `HeldNativeMcpRead` instead: the MCP permission read admission now makes (`_CapturedSources.read_permission_payload` for the app's service, patched on the class so the service stays stock), which `test_console_send_resend_guard.py` already uses (064d5cb014). Its new `wait_entered()` replaces the bare `until(hold.entered.is_set)`: if admission stops making that read, the test fails with "admission never made the held MCP read (_CapturedSources.read_permission_payload for the app's MCP service) within 30.0 s". Negative control: the hold's service match changed to `capture.service is None` (never true) turned `test_a_key_typed_...` red with exactly that message; reverted. `HeldMcpRead` stays (only `test_console_fallback_pressed_draft.py` uses it now), with a docstring note on why it no longer reaches the read.

**A race the working hold exposed.** With the hold entered, `test_a_bare_second_enter_never_sends_a_committed_draft_again[then_y]` failed 3 of 3 serially with `assert 'yz' == 'z'`. It waited for `len(_sent_or_queued(...)) >= 2`, which the "y" turn's saved user row meets before its run starts and reloads the composer; the log showed that turn's durable commit finishing during teardown (and then "blocked settlement_failed" as the app closed). It now waits until the provider has the second call and the session is idle. Then 2 of 2 serial runs passed (load 42-44).

**Cause of the image-only failure.** Route: the Enter is received by `receive_console_visible_intent` (`UI/Console_Modules/wiring.py:389`, `phase=ui_dispatch status=awaiting_review`), which never calls `self._hooks.dispatch`, so the stub could never be called. Refusal: that route takes the turn's attachments from the store's session inputs, never from `ChatScreen._console_pending_image_attachment`, which the test had replaced with `lambda: object()`. The turn carried an empty draft and no attachment, and the controller's draft check (`_validated_draft(draft, allow_empty=has_pending_attachment)`, copy "Type a message before sending.") refused it: `Console submit requires attention | accepted=False status=refused`. Test fault, not product. The test now stages a real image `PendingAttachment` in the store, makes the model vision-capable at both capability reads (`attachment_core.is_vision_capable`, `console_configuration_capture.is_vision_capable`), releases provider validation, and asserts the saved user turn is `("", b"\x89PNG")`, one provider call, and the composer keeps "hi".

**Runs.** Base dc371f0842, `-n 4`, load ~12: the eight nodes fail (7 timeouts on `hold.entered`, the image node on its own `until`). After the fix: `-n 4` whole file 24 passed in 71 s, 54 s and 67 s (three runs, load 35-55); the serial lane run is below.

**Lane.** Seven node ids (the six held functions, `[alone|then_y]` as one function id, and the image node) appended at the UI census tail; `MINIMUM_FILES` 2441 -> 2448. Local serial times with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`, load 46-51, call + teardown): `test_a_key_typed_while_admission_reads_mcp_is_handled_before_the_read_returns` 22.2 s; `test_enter_during_admission_is_replayed_after_it_as_the_held_pump_did` 17.0 s; `test_an_enter_deferred_behind_a_send_never_sends_another_tab_draft` 20.0 s; `test_leaving_a_tab_while_its_send_is_admitted_still_sends_it_there` 16.3 s; `test_an_enter_made_while_the_sent_draft_still_shows_never_takes_later_keys` 16.0 s; `test_a_bare_second_enter_never_sends_a_committed_draft_again` (both cases) 39.9 s; `test_an_image_only_enter_never_takes_text_typed_after_it` 17.4 s. The run: 8 passed in 159.4 s.

**Lesson.** `backlog/docs/lessons-testing-evidence.md`: a hold made by replacing a stock service's method never arms, because the Console refuses the send first; the filed diagnosis (the frame name) was wrong.

Not fixed here: `Tests/UI/test_console_fallback_pressed_draft.py` (no lane) still uses `HeldMcpRead` on purpose, to exercise the custom-capture path, and is red at base for the same refusal (both cases time out on `hold.entered`).

Files: `Tests/UI/test_console_send_admission_off_pump.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`, `backlog/docs/lessons-testing-evidence.md`.
