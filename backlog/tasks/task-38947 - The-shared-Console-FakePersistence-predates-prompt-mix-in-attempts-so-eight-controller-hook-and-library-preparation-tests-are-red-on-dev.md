---
id: TASK-38947
title: "The shared Console FakePersistence predates prompt mix-in attempts, so eight controller, hook and library-preparation tests are red on dev"
status: To Do
created_date: 2026-11-05 21:35
labels:
- follow-up
- console
- tests
priority: medium
references:
- Tests/Chat/test_console_chat_controller.py
- Tests/Chat/test_console_run_hooks_regressions.py
- Tests/Chat/test_console_automatic_library_preparation.py
- Tests/Chat/test_console_hook_admission.py
- tldw_chatbook/Chat/console_prompt_mixin_service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 35-60).

`FakePersistence` (`Tests/Chat/test_console_chat_controller.py:520`) has no `save_conversation_mixin_attempt`, no `get_conversation_mixin_attempt` and no `db`. Since 7970fbcada (2026-11-01), `tldw_chatbook/Chat/console_prompt_mixin_service.py:563-571` refuses a persisted chat's turn with `mixin_persistence_unavailable` when `save_conversation_mixin_attempt` is not callable or `db` is missing. The missing getter matters later, in the retry path (see the probe below).

**Users of the shared fake:**
- `test_console_chat_controller.py`: 6 tests;
- `test_console_run_hooks_regressions.py`: 3 tests;
- `test_console_automatic_library_preparation.py:1614`: one test with two params (the file's other `FakePersistence` imports, at `:1908` and `:3403`, come from `test_console_chat_store`, a different fake);
- `test_console_hook_admission.py:238`: already worked around by TASK-38718's local `MixinFakePersistence` subclass;
- `test_console_fleet_wake_safety.py:43`: imports the fake but never uses it.

**Red nodes.** Running those 13 nodes plus the fleet file at `-n 4` gave 8 failed and 13 passed, in 610 s:
- `test_blocked_send_persists_no_durable_record`, `test_submit_draft_marks_assistant_failed_when_stream_errors` and `test_retry_failed_message_streams_replacement_from_original_turn` fail with `KeyError: 'Unknown Console chat session: None'`.
- `test_accepted_send_persists_the_deferred_user_echo` fails with `assert 'user' in []`.
- `test_durable_hook_context_reaches_provider_and_audit` and `test_durable_postcommit_retry_keeps_hook_context_and_single_audit` fail with `TypeError: 'NoneType' object is not iterable`, because the gateway never received the messages.
- `test_queued_reclaim_thinking_refusal_returns_owner_to_recoverable_head[retry]` and `[bypass]` run until the 300 s pytest timeout.

**Probe.** A probe patched the class in-process only. With `db` and `save_conversation_mixin_attempt` added, 7 of the 8 pass, including the two that hung (now done in seconds). The retry test then fails on the missing `get_conversation_mixin_attempt`, and a getter that returns None is not enough (`'NoneType' object has no attribute 'status'`). It needs a fake that returns the saved attempt.

TASK-38718's notes say it did not check the fake's other users. No lane runs these files.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The shared fake saves and returns mix-in attempts; the eight nodes pass 10 of 10 runs alone and in their files at -n 4, and TASK-38718's local MixinFakePersistence subclass is folded into the shared fake
- [ ] #2 With the save removed from the shared fake, the eight go red again, and that negative control is recorded in the notes
- [ ] #3 The notes record why the library-preparation case hangs until the 300 s timeout instead of failing when the mix-in save is refused, and whether a refused save can leave a queued reclaim waiting in the product
<!-- AC:END -->
