---
id: TASK-41904
title: "The first-send onboarding write can race an in-flight Send"
status: Done
assignee:
- '@claude'
created_date: 2026-11-18 20:31
updated_date: 2026-11-19 16:13
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
- [x] #1 A queued Send admitted during the first-send onboarding write is not refused
- [x] #2 A test reproduces the race and is red before the fix
<!-- AC:END -->

## Implementation Plan

1. Reproduce through the mounted Send button: hold the first reply's onboarding save (`apply_settings_mutation_to_cli_config`) open and press Send meanwhile; record notices and provider calls.
2. Trace where the refusal comes from (the save's source fence, `ConsoleRuntime._first_send_source_scope`, entered synchronously by `receive_console_visible_intent`) and make the visible Send wait for the save as queued dispatch already does (`_first_send_source_scope_async`), instead of refusing.
3. Regression test beside the first-send persistence tests in `Tests/UI/test_console_runtime_ownership.py` (admission-sensitive lane), red at base; covering send suites base vs head; ratchets and preflight.

## Implementation Notes

**What races.** `ChatScreen._record_console_first_send` and `_save_console_onboarding_flag`, which the Description names, have no callers; the live save is `ConsoleRuntime._record_successful_first_send` -> `_persist_first_send_when_quiescent` (`Chat/console_received_dispatch.py`). It already waits until no capture holds the first-send fence, no turn is in custody and no prompt is queued, then sets `_first_send_publishing` and writes off-thread. A queue drains inside the custody of the turn it follows, so a queued prompt cannot be admitted during the write. What can is the next Send pressed while the write runs: its synchronous intake, `receive_console_visible_intent`, enters `_first_send_source_scope()`, which raises "Console configuration is being saved; Send again." while publishing, and `_dispatch_console_draft_send` shows that as "Draft or chat changed; Send again.". Reproduced through the mounted Send button with the save held open: that notice, and the provider never saw the second message. This is the Send TASK-41511's harness saw (its second `_send` right after the first reply).

**Fix (`UI/Console_Modules/composer_run_controls.py`, +4).** `_dispatch_console_draft_send` enters `_first_send_source_scope_async()` around the intake, the scope queued dispatch and the controller already use: it waits for the save and then holds the fence while the synchronous intake runs. With no save running it does not yield, so the Send's timing is unchanged. The save runs at most once per runtime, so the wait happens at most once. Raw `!` commands keep their own refusal (`test_raw_publication_refusal_keeps_original_live_draft`).

**Test.** `Tests/UI/test_console_runtime_ownership.py::test_a_send_pressed_during_the_first_send_save_waits_and_is_sent` (admission-sensitive lane). It holds `apply_settings_mutation_to_cli_config` open for the onboarding mutation, presses Send during it, then releases it, and requires no notice, both messages at the provider, both replies saved and the flag set. The harness wires `local_chat_conversation_service`, as the app does; without it the second turn ends with "Local conversation storage is unavailable." (found while probing).
- Red at base: `['Draft or chat changed; Send again.'] == []`.

**AC#1 wording.** No queued prompt can be admitted during the write (above); the Send that was refused is the one pressed during it, which the fix and the test cover. Docs: `Docs/User_Guide/console.md` (first-run section) says the save happens once Console is idle and that a Send pressed meanwhile waits for it.

**Runs.** The covering run is the one in TASK-41900's notes (send acknowledgement, draft snapshot, received intent feedback, raw CLI send, admission off pump, resume send guard, fallback input boundaries, runtime ownership): no node fails only at head except a pre-existing load flake shown 2 of 4 red on both. `test_console_runtime_ownership.py` runs in `derived-artifacts.yml`'s admission-sensitive step, so no census edit. `Tests/Architecture/test_module_size_ratchet.py`: the `composer_run_controls.py` row rises 1567 -> 1571, the change's own lines, with a dated note (owner rule). Preflight passed.

Files: `tldw_chatbook/UI/Console_Modules/composer_run_controls.py`, `Tests/UI/test_console_runtime_ownership.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/console.md`.
