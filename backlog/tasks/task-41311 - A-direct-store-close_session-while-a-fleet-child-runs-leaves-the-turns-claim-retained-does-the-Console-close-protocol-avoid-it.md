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

- **Test.** `Tests/Chat/test_console_ordinary_tree_widening.py::test_closing_a_chat_while_its_fleet_child_runs_releases_the_claim[cancellable|outlives_grace]`. A turn spawns a fleet child and finishes (sealed, OPAQUE). While the child is in its provider call, the test closes the chat the way the tab's confirmed close does (`UI/Console_Modules/session.py` `_complete_close`): `session_close_impact(...)`, then `ConsoleRuntime.close_session(id, expected_revision=impact.lifecycle.revision)`, which runs the controller's ticketed close. `cancellable` holds the child in an awaitable wait that the close's cancel can interrupt; `outlives_grace` blocks the child's thread, so the close finishes (2 s grace) while the child still runs. Then the child is released, and the test asserts the turn retires and another chat's Send is accepted. It also asserts `tree.check()` still refuses the closed chat, so tool effects stay gated.
- **Result on the unfixed code: retained in both cases** ("the closed chat's claim outlived its child"). The child member stayed `host_returned=False`, `drain_state() == "live"`. Cause: `finalize_session_close` calls `store.close_session`, which removes the session; the child's `_finish_execution` (`Agents/agent_native_execution.py`) then enters `tree.source.activate()`, and `_AgentSources._check_original_bindings` (`Agents/native_source_storage.py`) raised "original agent storage source changed" because the session was no longer in the store. `host_returned` and `owner.finish_root()` never ran, and `run_child_owned` swallowed the error. No step of the close protocol prevents this: its fleet drain (`await_fleet_terminal`) is bounded by the grace and returns when the handle goes terminal, which is before the child's host return, and `cancel_all_subagents` is cooperative. A direct `store.close_session` fails the same way.
- **Fix.** `_check_original_bindings` lets a retained tree (one with admitted members, the same `retained` test as the existing disposal carve-out) activate after its session left the store. A session replaced under the same id still refuses, and so does a tree with no admitted members, so admission is unchanged. Tool calls still go through `OrdinaryTurn.check()` (`check_local_sources`, `widen`, `borrow`), which refuses a closed session. The host-return readback still has to prove the child's row, so an unproven child still keeps its claim (TASK-40716).
- **Mutation.** Reverting only the `native_source_storage.py` change fails both params with "the closed chat's claim outlived its child". On base the test also needs TASK-41312's gateway fields (the child's resolution), so the check was run on head with only the fix reverted.
- No test in the tree closes a session through the store directly while a fleet child runs (the TASK-40715 probe that found this was never committed), so there was nothing to move to the real close path.
- **Checks (both TASK-41311 and TASK-41312, load 30-75).** Covering files: the 13 Chat files that use `_RecordingGateway` or the storage source, plus the 14 `Tests/Plugins` files on the `native_console` fixture (1330 nodes, `-n 4`). Base and head each had 97 failures, the same set with two exceptions. `test_original_console_mcp_context_reaches_child_model[workspace_custom_reader-1]` is red on base (twice alone) and green on head. `test_real_received_cold_skill_receipt_and_custom_pending_mcp[mounted_new_queued]` failed once in the loaded head run and passed twice alone. `./scripts/preflight.sh` passes. Serial `Tests/Architecture`: 3 failed, 1418 passed on both base and head (the same 3: the diagnostic inventory review and two timer-path inventory tests).
- Files: `tldw_chatbook/Agents/native_source_storage.py`, `Tests/Chat/test_console_ordinary_tree_widening.py`.
