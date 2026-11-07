---
id: TASK-38947
title: "The shared Console FakePersistence predates prompt mix-in attempts, so eight controller, hook and library-preparation tests are red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-05 21:35
updated_date: 2026-11-06 05:27
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
- [x] #1 The shared fake saves and returns mix-in attempts; the eight nodes pass 10 of 10 runs alone and in their files at -n 4, and TASK-38718's local MixinFakePersistence subclass is folded into the shared fake
- [x] #2 With the save removed from the shared fake, the six fake-dependent nodes go red again, and that negative control is recorded in the notes
- [x] #3 The notes record why the library-preparation case hangs until the 300 s timeout instead of failing when the mix-in save is refused, and whether a refused save can leave a queued reclaim waiting in the product
<!-- AC:END -->

## Implementation Plan

1. Re-run the 13 fake-user nodes at base e8eeae7db9 to confirm which are red and why.
2. Fold TASK-38718's `MixinFakePersistence` into the shared `FakePersistence`: an in-memory `db` marker, a `save_conversation_mixin_attempt` that records each attempt, and a `get_conversation_mixin_attempt` that returns the recorded attempt as a `MixinReplayRead`; drop the local subclass.
3. Run the eight nodes 10x alone and 10x in their files at -n 4 (with uptime), plus the other users of the fake.
4. Negative control: remove the save from the shared fake, show the eight red.
5. Trace why the library-preparation reclaim case hangs instead of failing when the save is refused, and whether the product can leave a queued reclaim waiting; record it.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fake.** The shared `FakePersistence` now carries TASK-38718's memory `db` marker, a `save_conversation_mixin_attempt` that records each attempt (conversation, native assistant id, snapshot), and a `get_conversation_mixin_attempt` that returns the recorded attempt as a `MixinReplayRead`. An unknown attempt id reads "unavailable"; an unmarked turn (no attempt id) reads "legacy". TASK-38718's local `MixinFakePersistence` subclass, and its now-unused import, are gone from `test_console_hook_admission.py`.

**Base check (e8eeae7db9).** The 13 nodes plus the fleet file at `-n 4` gave 10 failed and 37 passed. The extra two reds are `TestUserPromptSubmitHooks::test_block_rejects_send_without_assistant_row` and `test_stdout_injected_as_hook_origin_system_row`. Their whole class is red at base, including the three tests that do not use the fake: hook admission's `read_hooks_config_snapshot()` raises `RecoveryRequired('raw_source_selection_changed')` in the per-test sandbox (shown by printing the swallowed exception), the TASK-38111 class. They are out of scope.

**Two of the eight never touch the fake's mix-in surface.** The same `RecoveryRequired` refusal at hook admission ("Hooks unavailable; review or disable hooks before sending.") is the whole base red of `test_blocked_send_persists_no_durable_record` and `test_accepted_send_persists_the_deferred_user_echo`. Under a temporary `@private_profile_test` marker, both pass with the base fake: the blocked send stops before acceptance, and the accepted send's echo is written at durable acceptance, before the mix-in save. The other two controller nodes (`..._marks_assistant_failed_when_stream_errors`, `test_retry_failed_message_streams_replacement_from_original_turn`) need both the private profile and the fake: they are red under the marker with the base fake, and green with the new one.

**AC#1.** The four plain controller nodes ran under a temporary `@private_profile_test` marker (BRIEF-RULES' recipe for the RecoveryRequired trap), removed before commit:
- Alone, the eight nodes at `-n 4`: 10 of 10 runs passed (8 passed each, 12-15 s), at load 18-27.
- In their files (`test_console_chat_controller.py`, `test_console_run_hooks_regressions.py`, `test_console_automatic_library_preparation.py`, `test_console_hook_admission.py`, `test_console_fleet_wake_safety.py`), whole at `-n 4`: none of the eight, and no hook-admission node, failed in 10 of 10 runs (5.5-9.5 min each), at load 15-63. Eight runs failed the same 207 nodes: base's 215 minus the eight, all of them the files' other plain tests, which hit the same RecoveryRequired. Two runs each added one load flake that does not touch the fake. In run 8, `test_console_fleet_wake_safety.py::test_second_wake_progresses_while_first_waits_for_real_approval` hung 315 s at its `gather(*wake._delivery_tasks)` (the file imports the fake but never builds one). In run 10, `test_a_wedged_bootstrap_does_not_park_a_second_worker` measured 0.011 s against its 10 ms wall-clock bound.
- Without the marker, the four controller nodes are still red locally, from RecoveryRequired only (TASK-38111).

**AC#2.** The coordinator amended this criterion on 2026-11-06, from "the eight" to the six fake-dependent nodes. With the save removed from the shared fake (the method renamed), all six went red:
- both hook-regression nodes;
- both library-preparation params, each at the 60 s `--timeout` before the bounded wait below;
- the stream-error and retry controller nodes (`_last_failed_assistant` found no failed row).

The other two filed nodes, `test_blocked_send_persists_no_durable_record` and `test_accepted_send_persists_the_deferred_user_echo`, never reach the mix-in save, and they stayed green. Their filed red was TASK-38111's `RecoveryRequired` at hook admission; it is recorded in TASK-38111's notes.

**Bounded wait (coordinator ruling, 2026-11-06).** `_paused_queued_send` now waits for the provider start or the end of the owner turn, whichever comes first, for at most 30 s. A turn that ends before the provider fails at once with its result: "the owner turn ended before the provider: ConsoleSubmitResult(... 'Accepted turn is retained for recovery.' ...)". With the save removed from the fake (a temporary edit), both library-preparation params failed that way in 2.31 s each, where they used to hang to the 300 s timeout; with the save restored, 2 passed. The whole library-preparation file at `-n 4` then had 86 passed and 4 failed. The 4 are base's other four reds, none of them a caller of the helper. The helper's seven callers all pass. The file's eight other bare `await gateway.started.wait()` sites are left as they were. For a final check, the eight nodes ran once more with the temporary private-profile marker, at `-n 4` and load 31: 8 passed in 11 s. The marker was removed afterwards.

**AC#3. Why the library-preparation case hangs.** `_paused_queued_send` starts `run_prompt_chain("owner")` and then awaits `gateway.started.wait()` with no bound. Only the gateway's `stream_chat` sets that event. With the save refused, the owner turn is retained before provider entry (`mixin_persistence_unavailable`, then `_block_context_preflight`). The chain task finishes, but nothing awaits it, so the test waits until pytest-timeout. A probe that bounded `Event.wait` at 5 s put the TimeoutError at `_paused_queued_send` line 649, `await gateway.started.wait()`, with 0 provider calls.

**AC#3. Can a refused save leave a queued reclaim waiting in the product?** Yes, by design until the user acts, and not stranded. A probe on real SQLite (`ChatPersistenceService`, the save raising once during `retry_library_preparation`) found:
- The reclaim returns "Accepted turn is retained for recovery." (BLOCKED). Its assistant row stays pending, and a "Mix-in attempt could not be saved" system row is added.
- The queue is PAUSED, with the later entry waiting. The session sits in `_dispatch_recoveries`, so `accepted_live_turn` is True, and `resume_prompt_queue` refuses with "Finish or discard the pending response first."
- The recovery card offers Retry response and Discard. `retry_dispatch_recovery` completed the turn, settled the queue owner and drained the later entry (3 provider calls, queue empty, no live turn).
- One trap for anyone probing this: calling `resume_durable_postcommit` directly completes the turn but skips `_settle_recovered_queue_owner`. The queue then stays paused behind a stale recovery owner. Only the card's own action settles it.

Files: `Tests/Chat/test_console_chat_controller.py`, `Tests/Chat/test_console_hook_admission.py`, `Tests/Chat/test_console_automatic_library_preparation.py`.
<!-- SECTION:NOTES:END -->
