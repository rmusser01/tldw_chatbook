---
id: TASK-41904
title: "The first-send onboarding write can race an in-flight Send"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 20:31
updated_date: 2026-11-19 18:17
labels:
- follow-up
- console
priority: medium
references:
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41511's harness (round 16). The first-ever reply writes the first-send onboarding flag to config (`ChatScreen._record_console_first_send` and `_save_console_onboarding_flag`, chat_screen.py ~11859-11888, a thread worker calling `save_setting_to_cli_config`); a Send in flight at that moment (most plausibly a queued prompt admitted right after the first reply) fails with "Draft or chat changed; Send again.". The test harness now marks onboarding done to avoid it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Send pressed while the first-send onboarding write runs waits for the write and is then sent, not refused with "Draft or chat changed; Send again." (A queued prompt cannot be admitted during the write: the save starts only with nothing queued or in custody, and queue admission and queued dispatch already wait on the same fence.)
- [x] #2 A test reproduces the race and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce through the mounted Send button: hold the first reply's onboarding save (`apply_settings_mutation_to_cli_config`) open and press Send meanwhile; record notices and provider calls.
2. Trace where the refusal comes from (the save's source fence, `ConsoleRuntime._first_send_source_scope`, entered synchronously by `receive_console_visible_intent`) and make the visible Send wait for the save as queued dispatch already does (`_first_send_source_scope_async`), instead of refusing.
3. Regression test beside the first-send persistence tests in `Tests/UI/test_console_runtime_ownership.py` (admission-sensitive lane), red at base; covering send suites base vs head; ratchets and preflight.

## Implementation Notes

**What races.** `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag`, which the Description names, have no production callers (only `Tests/UI/test_console_native_chat_flow.py::test_console_record_first_send_repairs_corrupt_config_value` calls `_record_console_first_send` directly); the live save is `ConsoleRuntime._record_successful_first_send` -> `_persist_first_send_when_quiescent` (`Chat/console_received_dispatch.py`). It already waits until no capture holds the first-send fence, no turn is in custody and no prompt is queued, then sets `_first_send_publishing` and writes off-thread. A queue drains inside the custody of the turn it follows, so a queued prompt cannot be admitted during the write. What can is the next Send pressed while the write runs: its synchronous intake, `receive_console_visible_intent`, enters `_first_send_source_scope()`, which raises "Console configuration is being saved; Send again." while publishing, and `_dispatch_console_draft_send` shows that as "Draft or chat changed; Send again.". Reproduced through the mounted Send button with the save held open: that notice, and the provider never saw the second message. This is the Send TASK-41511's harness saw (its second `_send` right after the first reply).

**Fix (`UI/Console_Modules/composer_run_controls.py`, +4).** `_dispatch_console_draft_send` enters `_first_send_source_scope_async()` around the intake, the scope queued dispatch and the controller already use: it waits for the save and then holds the fence while the synchronous intake runs. With no save running it does not yield, so the Send's timing is unchanged. The save runs at most once per runtime, so the wait happens at most once. Raw `!` commands keep their own refusal (`test_raw_publication_refusal_keeps_original_live_draft`).

**Test.** `Tests/UI/test_console_runtime_ownership.py::test_a_send_pressed_during_the_first_send_save_waits_and_is_sent` (admission-sensitive lane). It holds `apply_settings_mutation_to_cli_config` open for the onboarding mutation, presses Send during it, then releases it, and requires no notice, both messages at the provider, both replies saved and the flag set. The harness wires `local_chat_conversation_service`, as the app does; without it the second turn ends with "Local conversation storage is unavailable." (found while probing).
- Red at base: `['Draft or chat changed; Send again.'] == []`.

**AC#1 reworded (2026-11-19, review r17cn-rev1).** As filed it read "A queued Send admitted during the first-send onboarding write is not refused". No queued prompt can be admitted during the write (above), so it was ticked on a reading the criterion did not say. The coordinator replaced it with the criterion now above, which names the Send that was refused: one pressed while the write runs. Docs: `Docs/User_Guide/console.md` (first-run section) says the save happens once Console is idle and that a Send pressed meanwhile waits for it.

**Runs.** The covering run is the one in TASK-41900's notes (send acknowledgement, draft snapshot, received intent feedback, raw CLI send, admission off pump, resume send guard, fallback input boundaries, runtime ownership): no node fails only at head except a pre-existing load flake shown 2 of 4 red on both. `test_console_runtime_ownership.py` runs in `derived-artifacts.yml`'s admission-sensitive step, so no census edit. `Tests/Architecture/test_module_size_ratchet.py`: the `composer_run_controls.py` row rises 1567 -> 1571, the change's own lines, with a dated note (owner rule). Preflight passed.

Files: `tldw_chatbook/UI/Console_Modules/composer_run_controls.py`, `Tests/UI/test_console_runtime_ownership.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/console.md`.

**Review r17cn-rev1 (fix round 1).**
- The test now pins the wait: while the save is held it requires exactly one provider call, and it holds the save open for 5 s (breaking early on a second call or a notice). With the save's fence opened in both scopes (temporary), a Send that does not wait reached the provider 2.0-2.5 s after the press at load 20, so a 1 s window could not see it. Red with the fence opened: "the Send did not wait" (`2 == 1`), 2 of 2; green at head 2 of 2. Removing only the intake's two fences did not reach that assertion: the Send met a later fence on its path and was refused ("Draft or chat changed"), failing the notice check instead.
- It waits for the second reply to be saved (two complete assistant rows) before reading them, and then for its terminal receipt to be seen (no unseen marks). Without that last wait the receipt stayed unseen in the shared bootstrap profile's database, and the next app in the same process opened with "A Console turn completed while hidden.": run first, it failed `test_console_send_without_mcp_store.py::test_unbound_store_with_a_replaced_mcp_reader_is_still_refused` 1 of 1 serially, and that node failed in 2 of 3 combined `-n 4` runs. With the wait the same ordered pair passed 2 of 2 and the combined `-n 4` run (this test, TASK-41903's, and all of `test_console_send_without_mcp_store.py`) passed 3 of 3, 22 passed each, at load 14-17. Preflight passed.
- User Guide: a `!` command still asks you to press Send again during the save (`test_raw_publication_refusal_keeps_original_live_draft`), and the line now says so.
- AC#1 reworded (above) and the "no callers" claim corrected (it has one test caller).

**Review r17cn-rev2 (fix round 2).**
- The unseen-receipt wait and its assertion now cover only this test's own chats (`persisted_conversation_id` of its sessions), since the bootstrap profile's database is shared and an earlier test's leaked receipt must not fail this one.
- While the save is held, the test also requires that no turn was admitted (`runtime._turn_custody` empty; the save itself starts only when it is). It breaks out of the 5 s hold as soon as one is. With the save's fence opened in both scopes (temporary, reverted): red 2 of 2 with "the Send was admitted during the save", 6.2-6.4 s per call. At head, serially: 3 of 3 passed, 15.1-15.3 s each, load 12. With TASK-41903's node and all of `test_console_send_without_mcp_store.py` at `-n 4`: 22 passed. Preflight passed.

**Review r17cn-rev3 (fix round 3).**
- **Leaked receipts from earlier tests.** Bootstrap-profile tests share one ChaChaNotes database, and an app announces any unseen terminal receipt there ("A Console turn completed while hidden."), which landed in this test's `notices`. The leaker was TASK-41900's new `[*-agent]` ordinary sends: a probe listing unseen marks after each test of `test_console_send_without_mcp_store.py` showed one left from `[bound-agent]` on. The fix does not filter notice text. Before mounting, the test now marks every unseen receipt already in the shared database seen (`_acknowledge_leaked_receipts`; the test has no chats yet, so none is its own), and at the end it waits for its own receipts through the shared `_wait_for_own_receipts_seen`.
- **Red before, green after.** The coordinator's order (`[non-default-data-dir-gateway]` then this test) passed 8 of 8 at the old test, because that send leaks nothing. With the real leaker first (`[bound-agent]`, then this test, serially, old test): 1 of 6 failed with the reviewer's error, and 3 of 44 in diagnostic loops without the acknowledgement. Deterministic control: a completed, unseen receipt planted for another conversation before mounting (temporary). Without the acknowledgement the test failed 2 of 2 with that notice; with it, it passed 2 of 2. After the fix, `[bound-agent]` then this test passed 12 of 12 (four concurrent loops, load 14-17), and the coordinator's order 2 of 2. The combined `-n 4` set (TASK-41903's test, this one, all of `test_console_send_without_mcp_store.py`) passed 5 of 5, 22 passed each, at load 12-14. Before the fix it failed 1 of 5, at `test_unbound_store_with_a_replaced_mcp_reader_is_still_refused`, from the same leak.
- **The dropped second Send seen once by the reviewer: (a), the bound.** After `release.set()` the path is: the held writer resumes; the save task clears `_first_send_publishing` in its `finally`; the waiting `_first_send_source_scope_async` re-checks and enters; then the synchronous intake runs. Every exit from the intake either sends, notifies, writes a transcript refusal row, or records a recovery. Its one quiet `""` is a capture another send already took. The quiet exit in `observed_send` (`superseded`) runs before the wait. So no path drops a waited Send without a trace. I saw the same failure once (1 of 7, leaker first, old 200-pause bound, load about 20). It did not recur in 68 later runs, including 44 with a dump of every transcript row, custody, recoveries and the composer on failure. The reply wait is now a 30 s monotonic deadline that ends as soon as the second reply is saved. The final assertions show every transcript row, so a refusal row would be visible if this ever recurs.
- `test_unbound_store_with_a_replaced_mcp_reader_is_still_refused` takes the same acknowledgement before mounting: its first notice must be the refusal. Preflight passed.

