---
id: TASK-39138
title: "Re-pin the 34 Console controller test nodes in test_console_chat_controller.py that the RecoveryRequired wall hid, or name the product defects among them"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-08 01:01'
created_date: 2026-11-07 02:03
labels:
- follow-up
- tests
- console
priority: medium
references:
- Tests/Chat/test_console_chat_controller.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/Backup_Recovery/raw_participants.py
dependencies:
- TASK-38111
- TASK-39045
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38111's notes (round-11 branch fu/r11-recovery) list them. Line numbers are dev's (`Tests/Chat/test_console_chat_controller.py`).

**How they surfaced.**
- With TASK-38111's profile marker, the file has 330 passed and 35 failed.
- The same 35 are red on dev under a marker plugin, and red alone: 38 nodes across three files, 38 failed at `-n 4` (load 49).
- One of the 35, `TestUserPromptSubmitHooks::test_wake_notice_fires_nothing`, is TASK-39045's.

**The other 34, by cause:**
- Test fakes refuse new keyword arguments (5):
  - `invocation_source`: `:1143`, `:7431`, `:8110[changed-history]`;
  - `validated`: `:9805`;
  - `expected_continuation`: `:11182`.
- Named-profile fakes have the wrong shape (2): `:9562` and `:9629`.
- Images and vision (8): `:2824`, `:2863`, `:2881`, `:2935`, `:3122`, `:3195`, `:3283`, `:3367`.
  - Three are sends refused with "Console send blocked: vision-model can't a…".
  - Five are media counts or history-part shapes.
- Session close and dispatch gap (3): `:1911` and `:1951` read "Accepted turn ownership changed before settlement." instead of "Session closed.", and `:1993` raises `KeyError 'Unknown Console chat session'`.
- Canvas and agent composition (9):
  - `test_real_canvas_controller_allows_exact_failed_assistant_retry` (`:6000`, 6 cases) raises `StopIteration`;
  - `:5764` and `:5825` read "Couldn't save the prepared turn";
  - `:8776` ends with its agent call state ERROR.
- Usage and the stop path (4):
  - `:7072` has no signals object, and `:7272` reads `'complete' == 'failed'`;
  - `:7138` and `:7209` time out at the 1 s provider-start wait (`:7172`, `:7248`), even run alone, so the provider never starts.
- Character emote metadata (2): `:1444` and `:1484`; the metadata now carries a `mixin_attempt_id`.
- The prompt-history stub (1): `test_accepted_send_survives_prompt_history_maintenance_refusal` (`:9783`) raises `RecoveryRequired` from its own `raw._raw_participant(history)` call, because its `default_prompt_history_path` stub cannot satisfy the guard's selected-path check.

**Why this needs care.** Some of these may be product defects rather than drift: vision-capable image sends are refused, a closed session reports ownership-changed copy, and canvas turns cannot save. A blind re-pin would hide them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 34 nodes passes against current production behaviour under the file's plain one-file command at `-n 4`, or is named in the notes as a product defect with a follow-up proposed, rather than re-pinned to the defect
- [x] #2 No node is deleted or loosened unless the notes name the commit that changed the behaviour it pinned
- [x] #3 The two stop-path usage nodes fail, if they fail, on their real assertion rather than on the provider-start wait; the before evidence is the `TimeoutError`
- [x] #4 The prompt-history node still proves the send is accepted while history persistence is refused: it fails under a temporary edit that lets the history write through
- [x] #5 The whole file passes 3 times at `-n 4` with the load recorded, apart from TASK-39045's node if that task has not landed
<!-- AC:END -->

## Implementation Plan

1. Base run of the whole file through run.sh at `-n 4`: 35 failed / 330 passed (34 in scope plus TASK-39045's `test_wake_notice_fires_nothing`).
2. For each cause group, read the node's traceback and the production code it drives, and find the commit that changed the behaviour (`git log -S`). Classify each node as test drift (name the commit) or product defect.
3. Drift: update the fake or expectation to the current contract without loosening what it proves. Defect: fix production with a test that is red before the fix.
4. Stop-path usage nodes: replace the 1 s provider-start wait with a deterministic one, so they fail, if at all, on their own assertion.
5. Prompt-history node: re-point its stub at the path the guard selects; negative control by letting the history write through.
6. Whole file 3 times at `-n 4` on head (and base for comparison), load recorded; covering tests for any production change; Tests/CI and preflight.

## Implementation Notes

**Result.** Base 8b762b52fa, whole file, run.sh `-n 4`: 35 failed / 330 passed. Head, three runs at load 42-50: 1 failed / 364 passed each. The one red is TASK-39045's `TestUserPromptSubmitHooks::test_wake_notice_fires_nothing` (`PermissionError: agent-wake sends require coordinator-issued wake authority`); that task has not landed. All 34 nodes in scope pass. 31 were test drift and are re-pinned below with the commit that moved the code. 3 were product defects, fixed with red-first evidence.

**Product defects (3 nodes, fixed).**
- *Close mid-stream reports ownership changed* (`test_close_streaming_session_stops_run_without_key_error`, `test_close_streaming_session_result_does_not_set_dispatch_gap_toast_flag`).
  - e788e2bc6f (2026-11-01) checks `AcceptedStopAuthority.is_current` before a stopped stream settles. A session the user closed in place fails that check, because its id is gone from the same store. So a close while streaming returned "Accepted turn ownership changed before settlement." with `BLOCKED`/`OWNER_CHANGED`, logged "Console submit requires attention", and never set the closed session's run state to `STOPPED`.
  - That commit's plan only targets successors: a replaced store or persistence, or a session with the same id.
  - Fix: a new `AcceptedStopAuthority.closed_in_place` (same store, persistence and database, and no session left with that id) returns `_session_closed_result`. Every other ownership change keeps the owner-changed result.
  - Evidence: red at base; with the new branch temporarily removed, both nodes fail (`'Accepted tur...e settlement.' == 'Session closed.'`).
- *A send to a closed chat never reaches its dispatch-gap result* (`test_submit_draft_dispatch_gap_session_closed_sets_toast_flag_with_informative_copy`). This took two commits:
  - 7970fbcada (2026-11-01): the prompt mix-in capture reads `store.session_input_snapshot` before the closed-session check, so a send captured before its tab closed raised `KeyError: 'Unknown Console chat session'`.
  - 3176bebc22 (2026-10-03): behind that, the draft-custody check refused the send with "Draft custody is unconfirmed." (not accepted, no `session_closed` toast flag), because `drain_agent_handoff` is False for a missing session.
  - Fix: both checks skip a session the store no longer holds, so the existing dispatch-gap result is reached ("Console session closed before your message could send.", `session_closed=True`).
  - Evidence: red at base (`KeyError`). Reverting only the capture guard turns it red again (`KeyError`). Reverting only the custody guard does too (`assert False is True`).

**Test drift (31 nodes, re-pinned).**
- Fakes refusing new keyword arguments (5). The fakes accept or forward the keywords; the wrapper around the real substitution passes them through.
  - `test_skill_refuse_after_preparation_removes_transient_echo`, `test_baseline_ignores_dispatch_time_substitution_and_stays_comparable` and `test_two_saved_turns_keep_history_references_through_production_trace[changed-history]`: 0327cb5cf9 (2026-10-15) passes `invocation_source`/`reference_offsets` to `_apply_skill_substitution`.
  - `test_direct_stream_pairs_typed_thinking_with_visible_answer`: 940a2e82b8 (2026-09-27) passes `validated=True` to `replace_message_thinking`.
  - `test_agent_cancellation_before_worker_start_retires_and_rejects_handoff`: bf1886c95c (2026-10-28; the call moved to `console_agent_reply.py` in b8826f8635) passes `expected_continuation` to `begin_generation_attempt`.
- Named-profile fakes (2).
  - `test_mcp_provider_session_and_persistent_paths_use_named_profile`: cd84c6b9b3 (2026-09-30) projects the executed result with `dataclasses.replace`. The fake now returns a real `ToolResult(ok=True)`.
  - `test_agent_provider_composition_captures_one_named_profile_for_mcp_and_builtin`: the context double gains `skill_context_maximum={}` (read since f3316f7781, 2026-10-01) and `mcp_definition_capture="captured"` (read since 077e5bfa09, 2026-10-31). Both are the production defaults.
- Images and vision (8): `test_submit_draft_sends_image_parts_when_vision_capable`, `test_image_only_draft_is_sendable`, `test_submit_stages_all_pendings_and_clears`, `test_history_images_capped_to_most_recent`, `test_image_budget_counts_images_newest_first`, `test_history_image_with_empty_mime_type_falls_back_to_default_mime`, and the two `test_provider_messages_for_next_send_estimate_*` nodes.
  - 3e618d7119 (2026-10-15) moved the turn's frozen capability capture, which holds the send's vision gate and the history-image budget, into `console_configuration_capture`, with its own `is_vision_capable`/`max_history_images` bindings. A patch on `controller_module` no longer reached the gate, so a real vision-capable send was never refused; only the patch missed the moved binding.
  - A new `_patch_capability` helper patches both bindings. The controller still reads its own for regenerate and for the next-send estimate without a turn context.
  - The controller's gate comment still called its own binding "the documented monkeypatch seam". It now names the frozen capture (same line count).
- Character emote metadata (2).
  - Since 7970fbcada every attempt stamps its prompt mix-in marker on the reply.
  - `test_generic_dispatch_does_not_arm_character_emote_protocol` now asserts the metadata is empty apart from that marker, which is stricter than "is None" ever was beside it.
  - `test_character_retry_without_chunks_preserves_prior_emote_metadata` compares every field except the retry's own attempt id.
- Canvas and agent composition (9).
  - b7dd8e53f2 (2026-09-08) wired the runtime store's `default_session_settings`. Sessions made by `_arm_session` therefore arrive with config-derived settings, whose provider is blank in this harness. The first durable commit's generation snapshot refuses a blank provider (`ValueError` -> "Couldn't save the prepared turn"). `_arm_session` now sets `llama_cpp` when the provider is blank, as it already did when settings were `None`. This is not a product defect: the test gateways resolve a provider whatever the session holds, and a real blank provider is refused at readiness.
  - That fix clears `test_real_canvas_controller_allows_exact_failed_assistant_retry` (6 cases) and `test_deferred_canvas_provider_uses_the_runtime_restart_latch`.
  - `test_real_agent_composition_advertises_and_invokes_shared_canvas_owner` also expects `canvas_guide` in the catalog (added by 2aed6c384d, 2026-09-10).
  - `test_production_trace_factory_keeps_canvas_tool_loops_on_their_saved_turns`: 0077b872f1 (2026-09-16) raised an unknown openai model's window from 4,096 to 32,000 tokens, so the first request now discloses every schema directly instead of naming the Canvas tools in the find_tools hint. The adapter's precondition (`"use find_tools, then load_tools" in system_message`) raised inside the provider call, and each run's first trace call settled `ERROR`. The precondition now asserts that `canvas_create` and `canvas_update` are offered in the first request. All the trace-ownership assertions are unchanged: 8 calls `COMPLETE`, routes, turn ids and revisions.
- Usage and the stop path (4).
  - `_UsageRecordingPersistence` lacked the seam 7970fbcada requires before provider entry for a saved chat (`save_conversation_mixin_attempt` and a `db`). Every send raised `MixinSnapshotUnavailableError('mixin_persistence_unavailable')` and ended "Accepted turn is retained for recovery." (`DISPATCH_SETTLEMENT`). The fake now records attempts and carries a memory `db` marker. No active-leaf or other write warning appears in these runs.
  - AC#3: `test_stopped_stream_persists_partial_input_usage` and `test_stop_path_usage_attach_survives_a_persistence_exception` failed at base with `TimeoutError` at the 1 s provider-start wait. They now use `until_provider_starts` (TASK-39143's helper; `await task` replaces the second 1 s bound). With the old fake they then failed on that helper's assertion, which printed the real result ("ended before the provider started: ... 'Accepted turn is retained for recovery.'"). With the fake fixed they pass.
  - Fixing the fake also cleared `test_agent_path_attaches_and_persists_usage` (no signals object) and `test_billed_turn_without_visible_content_still_records_usage` (`'complete' == 'failed'`).
- Prompt-history stub (1): `test_accepted_send_survives_prompt_history_maintenance_refusal`.
  - The raw guard checks the history path against the bound config's data directory (`raw_participants._participant_state`). That check has been in place since b5251e9a6e (2026-09-16), the commit that added this test; 2bce8ffa0f, named by TASK-38111, did not change the block. A stubbed `default_prompt_history_path` cannot satisfy it.
  - The test now uses the path production selects (the sandbox bootstrap profile's, under the test config root). "Nothing written" is now asserted as the file's bytes being unchanged, since that shared profile may already hold a history file.
  - AC#4: with `participant.close_admission()` temporarily removed, so the write goes through, the test fails (`assert 1 == 0` on `history.size`).

**Size waiver.** Owner decision (2026-11-06): a fix that pushes a module past its size ratchet raises the row to the measured size. `console_chat_controller.py` 29,180 -> 29,183. `console_pre_provider_stop.py` has no row.

**Covering runs** (run.sh, `-n 4`, head and base run concurrently at load 44-60, about 16 min each).
- The set was 23 files: the agent-swap, chat-start, dispatch-recovery rounds 1/4, durable-turn acceptance and fix round 1, local citation boundary, prompt mix-in admission, queue coordinator, received-intent inputs, run hooks, run markers, run state per session, runtime shutdown, accepted setup native and stop guards, native commit integration and configuration capture suites, plus Tests/UI `test_console_accepted_setup_stop`, `test_console_mcp_approval`, `test_console_runtime_ownership` and `test_skill_install_concurrent_confirms`.
- Result: head 51 failed / 980 passed, base 52 / 979. No node is red only on head. One MCP approval node was red only on base (`test_request_mcp_approvals_marks_human_input_wait_while_round_armed`).
- A message-by-message comparison of the shared reds found two differences:
  - a queue snapshot id that differs every run;
  - `test_console_run_state_per_session.py::test_submit_draft_closed_session_id_fails_closed_without_touching_active`, which pins the same dispatch-gap contract. It moved from base's `KeyError: 'Unknown Console chat session'` to "Hooks unavailable" (that file has no profile marker). Under a temporary `bootstrap_profile` marker (reverted) it passes on head. That is a second, independent pin agreeing with the dispatch-gap fix.
- e788e2bc6f's successor tests (`test_console_accepted_setup_native.py`, `test_console_accepted_stop_guards.py`, `Tests/UI/test_console_accepted_setup_stop.py`) pass on both trees, so a real ownership change still returns the owner-changed result.
- Module size ratchet: 201 passed.

**Review r12c-rev1 fixes (2026-11-07).**
- New `test_open_session_unconfirmed_draft_custody_is_refused`, parametrised over review required, a failed handoff write and an unsynced handoff write. It pins the open-session half of the relaxed custody check: the refusal copy, no provider call and no user row. All 3 cases are green. Each of two temporary regressions turned all 3 red (`assert True is False` on `accepted`): inverting the membership test to `target_id not in self.store._sessions`, and dropping the refusal.
- The closed-session comment moved above the custody `if`.
- `AcceptedStopAuthority.is_current` and `closed_in_place` now share `_same_owners` for the store, persistence and database checks. e788e2bc6f's successor suites still pass (58).
- Size row for `console_chat_controller.py`: 29,183 -> 29,184 (the moved comment), under the owner waiver.
- Whole controller file at `-n 4`, load 46-51: 1 failed (TASK-39045's node) / 367 passed. Module size ratchet: 201 passed. Preflight: passed.
