---
id: TASK-41138
title: "Console and Buddy test leftovers: a two-MCP-rounds payload flake, a cold-home Buddy child deadline, test_buddy_conversation_modal.py on no lane, and test_fleet_wake_retry.py without a profile marker"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-17 02:54
assignee:
- '@claude'
labels:
- follow-up
- console
- buddy
- tests
priority: low
references:
- Tests/UI/test_console_mcp_approval.py
- Tests/UI/test_buddy_conversation_modal.py
- Tests/Chat/test_fleet_wake_retry.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Each one either fails intermittently on a laned file or leaves behaviour unguarded.

**Evidence.**
- `test_console_mcp_approval.py::test_two_mcp_rounds_for_the_same_session_the_earlier_ones_teardown_does_not_evict_the_newer_ones_payload` (`:3236`) was reported flaky by the round-14 Console stream; the file is a whole UI census entry, so a flake reaches the required check. Not reproduced at closeout.
- `test_buddy_conversation_modal.py::test_cold_home_saved_buddy_bootstraps_only_after_explicit_open` fails with the base modal too: a child process's 5 s `until` deadline under load (TASK-40315's notes, load 62).
- `Tests/UI/test_buddy_conversation_modal.py` is in no workflow and not in the census.
- `Tests/Chat/test_fleet_wake_retry.py`: its three nodes fail the same with the base service; "that file has no profile marker" (TASK-40313's notes).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The payload flake is reproduced under load and fixed, or the notes record it could not be reproduced in 30 runs at load 40 or more
- [x] #2 The cold-home Buddy test's deadline comes from measured times with 2x headroom, or waits on its condition
- [x] #3 test_fleet_wake_retry.py gets the profile marker its config reads need and its three nodes pass
- [x] #4 test_buddy_conversation_modal.py's green nodes run on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Reproduce the payload flake under load (30 runs at load 40 or more), find its cause, fix it in the test, and rerun 30.
2. Measure the cold-home Buddy test's waits under load and derive its deadline (or wait on its condition).
3. Give test_fleet_wake_retry.py the profile marker its config reads need, then make its three nodes pass.
4. Run test_buddy_conversation_modal.py serially with the UI lane's flags and append its green tests to the UI census tail with their times.

## Implementation Notes

- **AC#1, reproduced and fixed.** `Tests/UI/test_console_mcp_approval.py::test_two_mcp_rounds_for_the_same_session_the_earlier_ones_teardown_does_not_evict_the_newer_ones_payload`, 30 runs in one serial session (a temporary plugin repeating the item; removed) at load 78-88 on 18 CPUs: `1 failed, 29 passed`, `TypeError: 'NoneType' object is not subscriptable`. Cause: the test slept a fixed 0.1 s after starting each worker thread and then read the session's head payload; under load the worker had not armed its round yet. The failed run also left both workers blocked: an armed round's clock counts only while its card is shown, so they never time out, and the pytest process sat in `Py_FinalizeEx` -> `wait_for_thread_shutdown` for 25 h until killed (the same trap as TASK-40314's lesson). Fix (test only): `_wait_for_parked_rounds(controller, session_id, count)` polls the parked payloads with a 10 s deadline in place of each sleep, the worker joins allow 10 s, and a `finally` calls `controller.begin_shutdown()` and joins both workers. Red/green under a temporary 0.3 s delay at the start of each worker (reverted): the old test failed with the same `TypeError` and then hung at interpreter exit until killed; the new test passed. After the fix, 30 runs in one session at load ~58: `30 passed in 15.40s`, and the process exited.
- **AC#2 (first pass; superseded by the review round below).** The deadline is not what fails. Measured inside the private child with a 30-120 s ceiling (three runs, load 80-101): the cold bootstrap wait took 2.47 s and the send-enabled wait 0.07 s, and the third wait (`gateway.started`) never completed in 120 s. The send is refused: a temporary `RecoveryRequired.__init__` spy (reverted) caught `RecoveryRequired('console_capture_source_not_ready')` from `_ready_attribute` (`tldw_chatbook/Chat/console_configuration_preparation.py:145`) for `LocalSkillsService.trust_service`, whose backing `_trust_service` is still None on a cold Home, reached from `resolve_turn_configuration_snapshot` <- `_submit_draft_body`. The diagnostics log records `console_send_stage ... exception_type=RecoveryRequired phase=controller_submit status=failed`. No deadline fixes that; the owner has to say whether a cold-Home Buddy send should materialize the trust service (product fix) or the test should expect the refusal. The 5 s `until` is unchanged; the node stays off the lane.
- **AC#3, met.** `Tests/Chat/test_fleet_wake_retry.py`'s three tests are `@private_profile_test` (a wake turn is a real Console submit, which reads config). Without the marker they fail even with the other changes (`assert False` at the first settle: no resolve within 10 s). The marker alone was not enough: under load the settles timed out (measured 1.2-1.5 s to the first resolve, 3.6 s to the first payload, 4.1 s from "provider ready" to delivery, against 5 s and 3 s deadlines), and the fairness test raced: its `routed` set one shared `gateway.ready` per submit, the two wake attempts interleave, and the first conversation was once delivered first. The settles now use 10 s (over 2x the worst measured), and readiness is per submit through a ContextVar set in `routed`. Red: 3 failed plainly (`:36`, `:78`, `:111`), 3 failed with the marker only. Green: 3 passed in each of three runs at load ~105.
- **AC#4, met.** `Tests/UI/test_buddy_conversation_modal.py` (module `bootstrap_profile`) serially with the UI lane's flags at load 68-77: `1 failed, 47 passed in 389.57s`; the failure is the cold-home node above. Its 21 green test functions are appended at the census tail by function (all their parameters), `MINIMUM_FILES` 2149 -> 2170 (checker OK; the 21 entries collect 47 nodes). Per-entry serial times (setup+call+teardown, s): rendered_typed_head 20.2, completed_release 3.1, typed_mixed_head 4.9, clock_claim 15.0, send_targets_bound_session 3.0, drafts_and_approval 3.4, closed_owner 2.7, existing_decision_cards 7.8, retains_headless_decisions 9.5, draft_follows_canonical 2.9, saved_load_revalidates 5.2, failed_cold_bootstrap 2.2, concurrent_saved_loader 2.8, dictation_finishes 2.0, closing_buddy_discards 2.0, refuses_unseen_input 10.9, late_voice_completion 4.2, real_console_send_preserves_drafts 155.9, recorder_buffer_callback 1.3, saved_bulk_reads 55.6, deferred_transcript_scroll 2.4; about 317 s in all.
- **Other runs.** `Tests/UI/test_console_mcp_approval.py` whole at `-n 4` (load ~90): 123 passed, 1 failed, `test_finishing_card_is_not_counted_and_keyboard_focuses_the_card` (`_console_attach_reconciled` False), which this task did not touch and which passed alone twice (47.0 s, 42.3 s). Intermittent under load; see Proposed follow-ups in the round report.
- Files: `Tests/UI/test_console_mcp_approval.py`, `Tests/Chat/test_fleet_wake_retry.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
- **Review r15ti-rev1 (fix round 1), with the coordinator's ruling on AC#2: fix the product.**
  - **AC#2, met: the cold-Home Buddy send is no longer refused.** `tldw_chatbook/UI/Navigation/buddy_conversation.py`'s `_submit` now awaits `_publish_skill_trust(app)` before `controller.submit_draft`. A mounted Console builds the skill trust service during its skill discovery; a cold Home never had, and capture (`_ready_attribute` in `console_configuration_preparation.py`) reads `LocalSkillsService._trust_service` without building it. The helper reads the app's existing slots, does nothing when the local service already holds its trust service, and otherwise builds it through `app.ensure_local_skill_trust_service()` (the off-loop builder Library uses; keyring discovery runs in `run_preparation_read`) and stores it in `local.trust_service`. Measured: that build took 0.05-0.08 s; the rest of the first cold send (personal-context bootstrap and storage evidence reads on worker threads) took 9.7-13.9 s from press to provider at load 55-75. The test keeps its assertions; its waits now use `COLD_HOME_SEND_SECONDS = 30.0` (over 2x the 13.9 s worst). Red with the product change reverted: the send wait expired at 30 s and the log showed the `console_send_stage ... status=failed` refusal; green with it.
  - **`test_late_voice_completion_after_close_cannot_change_the_draft`:** waits `until(lambda: voice.discards)`, because the discard runs in the modal's `on_unmount`, after the pop.
  - **`test_real_console_buddy_send_preserves_both_durable_console_drafts`:** its waits use `REAL_CONSOLE_WAIT_SECONDS = 15.0` (worst measured 2.2 s serially, 3.3 s at `-n 3`, load ~58). The stability loop then found a second flake, not a deadline: in 2 of about 16 runs of `[False]`, `target.draft` read "Unrelated visible composer". The test typed into the composer right after `new_session()` and `_sync_native_console_chat_ui()`, before `_console_visible_draft_session_id` had moved to the new session, so leaving Console banked that text into `target`. It now waits for the composer to belong to `other` before typing; 8 of 8 loops (16 node runs) passed after that. The census entry is split by parameter.
  - **Stability of the three fixed entries plus the cold-home node** (5 nodes): 5 serial runs (load 47-76) and 3 at `-n 4` (load ~40): 40 of 40 node runs passed. Serial per node, s: late_voice[starting] 4.4-10.0, [transcribing] 0.9-2.2; real_console[False] 42.5-98.6, [True] 49.1-102.1; cold_home 35.0-99.5. The whole file at `-n 4`: 48 passed, twice.
  - **Census.** The real-Console entry is now two parameter entries, and the cold-home node is appended at the tail; `MINIMUM_FILES` 2179 -> 2181.
  - **The same flake class in `test_console_mcp_approval.py`.** Every remaining fixed sleep before reading a round now waits for the condition it reads: `_wait_for_mounted_round(mounted)` or `_wait_for_parked_rounds(controller, session, n[, kind])` (both on the file's existing `_wait_until`) in the five two-thread tests, and `_wait_until(lambda: received and received[-1] is not None)` in the four resolver threads; their joins allow 10 s. The two remaining sleeps are deliberate negative windows. Red/green under a temporary 0.3 s delay at the start of each worker: the five tests failed with the old waits and passed with the new; the resolver case (`test_request_mcp_approvals_preserves_native_call_ids_as_verdict_keys`, which failed once at `-n 4`) went from timeout verdicts after the 30 s round to a pass. The whole file at `-n 4`: 124 passed, twice.
  - The Buddy guide (`Docs/User_Guide/buddies.md`) already says a reply is typed and sent "without changing the underlying screen"; the fix makes that true from a cold Home, so no doc change.
  - `Tests/Persona_Buddy` at `-n 4`: the same 19 failures on this branch and on the base worktree (identical sets), none touching this change.
  - Files: `tldw_chatbook/UI/Navigation/buddy_conversation.py`, `Tests/UI/test_buddy_conversation_modal.py`, `Tests/UI/test_console_mcp_approval.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
