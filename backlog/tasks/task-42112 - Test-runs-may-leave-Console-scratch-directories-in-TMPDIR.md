---
id: TASK-42112
title: "Test runs may leave Console scratch directories in TMPDIR"
status: Done
assignee: ['@claude']
created_date: 2026-11-20 10:36
updated_date: 2026-11-20 20:52
labels:
- follow-up
- tests
- console
priority: low
references:
- tldw_chatbook/Chat/console_scratch_space.py
- Tests/conftest.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41708 (round 17). After its full guarded sweep of the test suite (2026-11-19/20, every test file through the scratch wrapper, whose pytest basetemp is deleted after each run), the implementer reported about 45,000 directories, about 16 GB, left in the system `$TMPDIR`. About 28,000 of them were named `tldw-console-*`, the prefix `ConsoleScratchSpaceManager` passes to `tempfile.mkdtemp` (`Chat/console_scratch_space.py`). Its `dispose()` and `close()` remove them; a test that builds a Console runtime and never disposes it would leave one behind.

By 2026-11-20 10:36 UTC no `tldw-console-*` directory remained in `$TMPDIR`, and what removed them is not known. So the leak is reported, not yet reproduced. Free disk on the machine fell from about 223 GB to 30 GB during the sweep, while the wrapper's own basetemp stayed under 100 MB.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A local run of the Console UI and Chat test files is measured for `tldw-console-*` directories left in `$TMPDIR` afterwards, with the count recorded
- [x] #2 If any are left, the tests or fixtures that leave them are named and fixed so a run leaves none, or each remaining source has its own task
<!-- AC:END -->

## Implementation Plan

1. Measure in chunks: Tests/Chat, then Tests/UI/test_console_* split by first letter (a-c, d-l, m-r, s, t-z).
   - Each chunk runs with `-n 4` through the scratch wrapper, with a private per-worker `$TMPDIR`, because other sessions add `tldw-console-*` directories to the shared one all the time.
   - A throwaway probe plugin (never committed) records the test during which each surviving directory appeared.
2. Name the leaking tests and fixtures per file. Where a rig is simple to dispose of properly (for example the `test_console_agent_swap._controller` users), fix the rig.
3. Add a small session-scoped autouse fixture in `Tests/conftest.py`. It gives `ConsoleScratchSpaceManager` a pytest-owned default `temp_parent` only when the caller passes none, so any remaining leak lands in pytest's basetemp, which the run deletes.
4. Pin it in a laned test (`Tests/UI/test_console_runtime_ownership.py`): a disposed runtime still removes its own directory, so the redirect cannot hide a product-side leak; and an explicit `temp_parent` is still honoured.
5. Re-measure one chunk with the fix and record the count.

## Implementation Notes

The leak is real: a Console controller built in a test without `scratch_spaces`, and never shut down, leaves the `tldw-console-*` directory it allocated in `$TMPDIR`.

**Fix**
- `Tests/conftest.py` gains a session-scoped autouse fixture, `console_scratch_defaults_to_basetemp`. It wraps `ConsoleScratchSpaceManager.__init__` so that a manager built without `temp_parent` gets a `tmp_path_factory.mktemp("console-scratch")` parent. An explicit `temp_parent` is passed through unchanged.
  - Any directory that is still leaked now lands in pytest's basetemp instead of `$TMPDIR`. The scratch wrapper deletes basetemp after each run; a bare pytest keeps the last three basetemps, so the leaked directories are bounded and kept for up to three runs. No product code changed.
  - The wrapper is decorated with `functools.wraps(real_init)`.
- The shared rig `Tests/Chat/test_console_project_instruction_traces.py::_project_console` never shuts its controller down. It now builds a `ConsoleScratchSpaceManager(temp_parent=tmp_path)` and disposes it in its `finally`.
- `test_console_agent_swap._controller` is left as it was. Review rev1 found that many of its callers do shut down (for example `test_console_chat_start.py` and `test_console_agent_progress_identity.py`). An external manager makes `_owns_scratch_spaces` False, so `shutdown()` would no longer dispose it: with that change `test_native_start_uses_both_receipts_and_literal_machine_request[*]` left 3 directories, against 0 without it. The conftest default covers this rig's callers that do not shut down.
- Pins (in `Tests/UI/test_console_runtime_ownership.py`, laned in derived-artifacts.yml):
  - `test_a_disposed_runtime_still_removes_its_scratch_under_the_suite_default`: an argument-less runtime's scratch lies under basetemp, and `runtime.dispose()` still removes it. The redirect therefore cannot hide a product-side leak.
  - `test_an_explicit_scratch_parent_still_wins_over_the_suite_default`.

**Red evidence** (each under a temporary edit, then reverted):
- With the fixture's `setattr` removed, the first pin fails: `assert .../bt-r18cr-pin in <PosixPath.parents>`.
- With `shutil.rmtree` in `_cleanup_record` replaced by `pass`, both pins fail: `assert not True`.
- With the fixture ignoring an explicit parent, the second pin fails: `console-scratch0 == test_an_explicit_scratch_paren0`.

**Measurement (AC#1).** Each chunk ran `-n 4` through the scratch wrapper, with a private per-worker `$TMPDIR`, because other sessions keep adding `tldw-console-*` directories to the shared one (35,537 there by 19:17Z). A throwaway probe plugin, never committed, recorded the test during which each surviving directory appeared.

Before the fix, 2026-11-20:

| Chunk | Time (Z) | Load | Result | Directories left |
|---|---|---|---|---|
| `Tests/Chat` | 16:17-18:51 | 10-28 | 825 failed, 21,847 passed | **1,650** (from 1,443 tests in 111 files) |
| `Tests/UI/test_console_[a-c]*` | | 9-17 | 389 failed, 1,847 passed | **0** |
| `[d-l]*` | | | 440 failed, 1,282 passed | **1** |
| `[m-r]*` | | | 590 failed, 2,514 passed | **18** |
| `s*` | | | 188 failed, 1,795 passed | **2** |
| `[t-z]*` | | | 371 failed, 1,168 passed | **6** |

- That is 1,677 directories in total.
- The failures are the local profile trap (`RecoveryRequired`, `Hooks unavailable`) and are not touched here.
- At the end of the Chat run one xdist worker (gw1) sat idle and the controller never finished. A SIGINT to that worker let it write its probe data and end the run, so all four workers' data is complete.

After the fix:
- **UI `[m-r]*`** (19:39-20:06Z): **0** directories in `$TMPDIR` (18 before). 18 are contained in basetemp.
  - 16 test outcomes differed between the two m-r runs, all delete/undo, parallel-run and rewind UI tests. Re-running exactly those 16 at load 10-12 failed 11 of them with the fixture and 10 without it. They are local-trap and timing flakes, not caused by the fixture.
- **The 32 most-leaking Chat files** (19:39-19:51Z):
  - An unfixed re-run of the same files (19:52-20:05Z) left 1,225 directories in `$TMPDIR`.
  - The fixed run left **0** in `$TMPDIR`; 1,030 are contained in basetemp. The 195 fewer come from the rig changes.
  - This run still had the `_controller` change that review rev1 reverted. The `$TMPDIR` count does not depend on it, because the conftest default covers that rig.
  - Both runs had the same 46 failing tests.
- Spot check: `test_console_prompt_mixin_retry.py` and `test_console_rewind_summarize.py` left 160 directories in `$TMPDIR` before, and 160 in basetemp after.

**Sources (AC#2)**, per test file, from the probe. Each entry is directories left / tests that left them. In every case the cause is a `ConsoleChatController` built in the test, or by a shared rig, without `scratch_spaces`, and never shut down. The shared rigs are:
- `test_console_project_instruction_traces._project_console` (fixed; also used by `test_console_trace_failed_tool_run.py` and `test_console_trace_discarded_tool_run.py`)
- `test_console_agent_swap._controller`, through callers that never shut the controller down, for example `test_console_fleet_wake_safety.py` and `test_automatic_wake_dispatch.py` (unchanged; covered by the conftest default)
- `test_console_trace_discarded_tool_run._reload_console`
- `test_console_rewind_summarize._durable_controller`

In `test_console_chat_controller.py`, 210 direct constructions leak one directory each.

Tests/Chat:
test_console_chat_controller.py (219/219), test_console_trace_failed_tool_run.py (161/51), test_console_prompt_mixin_retry.py (102/95), test_console_local_citation_boundary.py (95/94), test_console_trace_discarded_tool_run.py (81/31), test_console_project_instruction_traces.py (67/67), test_console_automatic_library_preparation.py (62/59), test_console_rewind_summarize.py (58/58), test_console_turn_resend.py (43/39), test_console_agent_swap.py (38/37), test_console_failure_copy_typed.py (35/34), test_console_shared_tool_preparation.py (29/29), test_console_prompt_queue_coordinator.py (29/28), test_console_compaction_live_session.py (26/26), test_console_canvas_profile_preview.py (25/25), test_console_fleet_wake.py (22/20), test_console_async_mcp_snapshot.py (19/19), test_self_hosted_retry_accounting.py (19/19), test_console_first_send_atomicity.py (19/19), test_progress_wakes.py (18/18), test_response_rule_corrections.py (17/17), test_console_dispatch_recovery_relaunch.py (17/10), test_console_compaction_ask_hold.py (16/16), test_console_durable_commit_offload.py (15/9), test_console_trace_rewritten_history.py (14/14), test_console_provider_continuation.py (14/14), test_console_trace_current_turn_transforms.py (14/14), test_console_provider_failure_copy.py (13/13), test_console_trace_first_send_atomicity.py (11/10), test_trajectory_capture.py (10/10), test_console_fork_fresh_lineage.py (10/10), test_console_turn_execution_context.py (9/9), test_automatic_wake_scheduling.py (9/5), test_automatic_wake_dispatch.py (9/9), test_console_agent_project_instructions.py (9/9), test_console_run_hooks_regressions.py (8/8), test_console_trace_surface_identity.py (8/6), test_workflow_failed_phase_release.py (8/8), test_console_terminal_citation_persistence.py (8/8), test_console_shutdown_after_done_finalization.py (7/7), test_console_capture_send_triggers.py (7/7), test_console_ssh_custody.py (7/7), test_workflow_activity_admission.py (7/7), test_console_prompt_mixin_journey.py (7/7), test_automatic_wake_recovery.py (7/7), test_console_message_delete_persistence.py (7/5), test_console_edit_resend.py (7/7), test_hooks_v2_continuations.py (7/7), test_console_skill_substitution.py (6/6), test_console_chat_controller_exchanges.py (6/6), test_console_shared_tool_cache.py (6/6), test_console_send_gate_queue_race.py (6/6), test_console_send_diagnostics.py (6/6), test_console_regenerate_branching.py (6/6), test_console_attachment_riders.py (6/6), test_custom_endpoint_trace_surface.py (6/6), test_console_dispatch_recovery_fix_round2.py (6/6), test_console_run_state_per_session.py (5/4), test_console_turn_library_authority.py (5/5), test_console_review_owner_lazy.py (5/5), test_console_submit_blocker_code.py (5/5), test_console_raw_shell_revocation.py (5/5), test_console_personal_context_snapshot.py (5/5), test_console_prompt_mixin_forks.py (5/3), test_automatic_wake_budget.py (5/5), test_console_fleet_wake_safety.py (5/4), test_console_capacity_refusal_copy.py (4/4), test_workflow_phase_wave_native.py (4/4), test_console_trace_request_missing_pause.py (4/4), test_console_trace_model_retries.py (4/4), test_console_pre_run_cleanup_stop.py (4/4), test_console_note_span_actions.py (4/4), test_console_variant_stream.py (4/4), test_console_durable_turn_fix_round2.py (4/4), test_console_dispatch_recovery.py (4/4), test_console_agent_bridge.py (4/4), test_console_durable_commit_window.py (4/4), test_workflow_native_return.py (3/3), test_console_close_during_durable_postcommit.py (3/3), test_console_raw_cli_custody.py (3/3), test_console_workspace_guidance.py (3/2), test_workflow_settlement_delivery.py (3/3), test_console_hook_admission.py (3/3), test_console_dispatch_continuation_handoff.py (3/3), test_console_skill_script_confirm.py (2/2), test_console_chat_create_integration.py (2/2), test_console_trace_system_prompt_send.py (2/2), test_console_prompt_transform_single_collection.py (2/2), test_console_committed_send_recovery.py (2/2), test_console_staged_evidence_dispatch_pin.py (2/2), test_console_plugin_tests.py (2/2), test_workflow_planning_native.py (2/2), test_console_durable_commit_diagnostics.py (2/2), test_console_context_presentation_reads.py (2/2), test_console_approval_scope_journeys.py (1/1), test_console_runtime_lifetime.py (1/1), test_fleet_budget_boundary_probes.py (1/1), test_console_chat_store.py (1/1), test_console_staged_evidence_trace_export.py (1/1), test_response_rules_evidence.py (1/1), test_workflow_pause_partial_native.py (1/1), test_console_stop_reliability.py (1/1), test_console_voice_preflight.py (1/1), test_console_project_instruction_threads.py (1/1), test_console_named_workspace_first_send.py (1/1), test_console_ordinary_tree.py (1/1), test_console_dispatch_recovery_fix_round3.py (1/1), test_console_continuation_review_fixes.py (1/1), test_console_dispatch_retry_cancel_during_cas_t22502.py (1/1), test_hooks_v2_teardown.py (1/1), test_console_first_send_survivor_native.py (1/1)

Tests/UI:
- `test_console_project_instructions.py` (17/17): the `test_captured_context_*`, `test_context_preview_*`, `test_controller_preview_*` and `test_preview_revalidates_authority_after_awaited_composition[*]` tests, `test_context_snapshot_stays_on_captured_session_after_active_switch`, `test_parked_session_preview_does_not_publish_global_mcp_counts`, and `test_preview_fails_closed_when_run_log_binding_is_uncertain`.
- `test_console_unreserved_commit_failure.py` (4/4): `test_a_commit_another_owner_holds_is_left_committing[reservation in flight]`, `[commit recorded]`, `test_a_commit_that_fails_before_its_reservation_pauses_for_persistence`, and `test_a_newer_attempt_of_the_failed_send_is_left_alone`.
- `test_console_workflow_controls.py` (2/2): `test_native_workflow_control_boundaries[metadata]` and `[unknown-task]`.
- `test_console_staged_evidence_sessions.py` (2/2): `test_accepted_sources_are_attached_when_prerequisite_refuses_before_contact[saved]` and `[temporary]`.
- `test_console_dictionary_send_integration.py` (1/1): `test_native_send_applies_conversation_dictionary_agent_branch`.
- `test_console_resume_active_path.py` (1/1): `test_resumed_untrusted_handoff_keeps_provider_sequence`.

No source is left writing to `$TMPDIR`. The rigs and tests other than `_project_console` still do not shut their controllers down; their directories now go to basetemp.

**Checks:**

| Run | Result |
|---|---|
| `test_console_runtime_ownership.py` `-n 4` | 96 passed, 1 xfailed |
| agent_swap, project_instruction_traces, trace_failed/discarded_tool_run and scratch_space `-n 4` | 201 passed, 2 failed |
| Tests that read `Tests/conftest.py` | 71 passed, 2 skipped |
| ruff F | clean |
| `ruff_gate.py` | RATCHET OK |
| `preflight.sh` | all checks passed |
| After the review rev1 fixes (20:50Z, load 4.8): `test_console_agent_swap.py` `-n 4` | 44 passed, 2 failed |
| After the review rev1 fixes: `test_console_runtime_ownership.py` `-n 4` | 96 passed, 1 xfailed |
| After the review rev1 fixes: ruff F, `preflight.sh` | clean, all checks passed |

The 2 failures in both agent_swap runs are `test_stop_cancels_tree_and_persists_cancelled` and `test_stop_before_first_token_persists_cancelled_no_agent_run_failed`. They also fail in both unfixed runs.

**Files:**
- `Tests/conftest.py`
- `Tests/UI/test_console_runtime_ownership.py`
- `Tests/Chat/test_console_project_instruction_traces.py`
