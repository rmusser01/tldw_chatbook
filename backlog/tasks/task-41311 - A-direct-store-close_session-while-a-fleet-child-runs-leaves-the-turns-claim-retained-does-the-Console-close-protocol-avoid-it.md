---
id: TASK-41311
title: "A direct store.close_session while a fleet child runs leaves the turn's claim retained: does the Console's close protocol avoid it?"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
priority: low
references:
- tldw_chatbook/Chat/console_chat_store.py
- tldw_chatbook/Chat/console_runtime.py
- tldw_chatbook/Chat/console_chat_controller.py
- tldw_chatbook/Agents/ordinary_tree.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715 found it while testing (PR #3548 lists it under "Follow-ups to be filed"). Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/`.

**Why.** While testing TASK-40715, closing a session with a direct `store.close_session` while a fleet child was running left the turn's claim retained, so the turn never retired. A retained claim that has widened to OPAQUE blocks every other chat's Send ("Workspace activity is busy") until restart. The user-facing close does not call the store directly, so this may only be a test-harness shape, but nobody has checked.

**Evidence.**
- The user-facing close goes through `UI/Console_Modules/session.py:3328`, then `ConsoleRuntime.close_session` (`Chat/console_runtime.py:5886`, which drains bounded publication and turns within a grace period), then the controller's ticketed close (`Chat/console_chat_controller.py:14900-14915`). That close refuses while a native preparation or an ordinary save still owns the session, and only then calls `store.close_session`.
- `ConsoleChatStore.close_session` (`Chat/console_chat_store.py:5081`) itself does none of that.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test closes a session through the Console's own close path while a fleet child runs, and records whether the turn's claim is released once the child returns
- [x] #2 If the claim stays retained on that path, it is released once the child returns, pinned by that test (red on dev); if not, the task notes say which step of the close protocol prevents it, and any test that closes through the store directly is changed to use the real close path
<!-- AC:END -->

## Implementation Plan

ADR required: no. It extends the existing disposal carve-out for a retained tree's storage check to a closed session; no new decision.

1. Add a test that closes a chat through the Console's own close path (the tab's impact, then `ConsoleRuntime.close_session`) while a fleet child runs, both when the child observes cancellation and when it outlives the close grace, and records whether the claim is released once the child returns.
2. If it stays retained, find the step that keeps it and fix that step; show the test red on base and the mutation check.

## Implementation Notes

Outcome: the real close path retains the claim too, so this is a product bug; fixed.

- **Test.** `Tests/Chat/test_console_ordinary_tree_widening.py::test_closing_a_chat_while_its_fleet_child_runs_releases_the_claim`. A turn spawns a fleet child and finishes (sealed, OPAQUE). While the child is in its provider call, the test closes the chat the way the tab's confirmed close does (`UI/Console_Modules/session.py` `_complete_close`): `session_close_impact(...)`, then `ConsoleRuntime.close_session(id, expected_revision=impact.lifecycle.revision)`, which runs the controller's ticketed close. The close finishes (2 s grace) while the child still runs. Then the child is released and emits a `run_skill_script` call. The test asserts: `tree.check()` refuses the closed chat; the script never runs; the child run ends `cancelled`; the turn retires; another chat's Send is accepted.
- **A fleet child's cancel never cuts an in-flight provider call.** The close does cancel it (`cancel_all_subagents`, logged "cancelled 1 sub-agent(s)"), but `_cancel_fleet_handles` only sets the child's cancel Event, which the child reads at its next call boundary. (Review round) A first version of the test had a second mode meant to let the child "observe the cancel" during the grace. Review found the child's wait returned only after the test released it (instrumented: close done, then the wait returned with release set), so both modes tested the same ordering. The stream's `signals` kwarg carries no cancel flag, so there was nothing real to watch, and the mode was dropped.
- **Result on the unfixed code: retained** ("the closed chat's claim outlived its child"). The child member stayed `host_returned=False`, `drain_state() == "live"`. Cause: `finalize_session_close` calls `store.close_session`, which removes the session; the child's `_finish_execution` (`Agents/agent_native_execution.py`) then enters `tree.source.activate()`, and `_AgentSources._check_original_bindings` (`Agents/native_source_storage.py`) raised "original agent storage source changed" because the session was no longer in the store. `host_returned` and `owner.finish_root()` never ran, and `run_child_owned` swallowed the error. No step of the close protocol prevents this: its fleet drain (`await_fleet_terminal`) is bounded by the grace and returns once the child's run is terminal, which is before the child's host return, and the cancel is cooperative. A direct `store.close_session` fails the same way.
- **Fix.** In `_check_original_bindings`, an ordinary tree with admitted members (`tree.has_admitted_members()`; the workflow-preparation branch of the disposal carve-out is not used for this) passes the session-presence test after its session left the store. A session replaced under the same id still refuses, and so does a tree with no admitted members, so admission is unchanged. This check does not gate effects after the close; each effect's own check does. `OrdinaryTurn.check()` (via `check_local_sources`, `widen`, `borrow`) refuses a closed session. The host-return readback still has to prove the child's row, so an unproven child still keeps its claim (TASK-40716).
- **Mutations** (each run alone on head, one change at a time):
  - Storage fix reverted: fails, "the closed chat's claim outlived its child". On base the test also needs TASK-41312's gateway fields, so this was run on head.
  - `OrdinaryTurn.check` accepting a missing session: fails at `tree.check()` ("DID NOT RAISE").
  - Close without `cancel_all_subagents`: fails at the status assertion (`error`, not `cancelled`). The script still does not run: the run log shows `tool_execution_started` then `tool_failed`, then the child's next model call errors.
  - Both of the above (with the `tree.check()` assertion removed): the script still does not run (status `error`). Another layer of the skill-script path refuses the closed chat too, so no single-gate mutation makes the "nothing executed" assertion fail; the gate itself is pinned by the `tree.check()` assertion.
- No test in the tree closes a session through the store directly while a fleet child runs (the TASK-40715 probe that found this was never committed), so there was nothing to move to the real close path.
- **Checks (both TASK-41311 and TASK-41312, load 50-150).** Covering files: the 13 Chat files that use `_RecordingGateway` or the storage source, the 14 `Tests/Plugins` files on the `native_console` fixture, `Tests/Chat/test_console_ordinary_tree.py` and `Tests/Agents/test_agent_service.py` (about 1560 nodes, `-n 4`). After the review fixes, head had 100 failures; 97 of them also fail on base fbf1c9ea74. The rest, each re-run alone: `test_late_subagent_stop_retains_original_tree_until_physical_return` (its child pytest timed out under load; passes alone twice on head, fails on base); `test_real_received_cold_skill_receipt_and_custom_pending_mcp[queued|mounted_queue_before]` (a 70 s child journey budget: `mounted_queue_before` passed 2 of 3 alone on head and 0 of 3 on base; `queued` passed twice alone on head); `test_installed_owned_mcp_post_compaction[cold]` (passed twice alone on head). `test_context_resolution_does_not_observe_destination_before_dispatch[never-allowed]` fails alone on head and on base. 31 nodes red on base pass on head. `Tests/Agents/test_agent_service.py`: 156/156 on base, 157/157 on head. `./scripts/preflight.sh` passes. Serial `Tests/Architecture` on base: 3 failed, 1418 passed (the diagnostic inventory review and two timer-path inventory tests); head (load 55): 5 failed, 1416 passed; the 3 base failures plus two 300 s pytest-timeout cases of the inventory file (`..._unchanged`, `..._canonical_schema...`), which pass on head when that file runs alone with `--timeout=1200` (79 passed, 1 failed on both base and head: the same `test_reviewed_diagnostic_changes_are_metadata_only`). Review round (verified): the storage change uses `tree.has_admitted_members()` only, with no workflow branch; the `consent_decision` move is behaviour-preserving and was needed to hold `agent_service.py` under its size ratchet.
- Files: `tldw_chatbook/Agents/native_source_storage.py`, `Tests/Chat/test_console_ordinary_tree_widening.py`.
