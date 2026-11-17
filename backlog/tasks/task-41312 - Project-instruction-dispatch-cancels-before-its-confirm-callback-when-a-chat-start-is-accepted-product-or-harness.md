---
id: TASK-41312
title: "Project-instruction dispatch cancels before its confirm callback when a chat start is accepted: product or harness?"
status: Done
created_date: 2026-11-16 19:55
labels:
- follow-up
- console
- agents
- tests
priority: low
references:
- tldw_chatbook/Chat/console_project_dispatch.py
- tldw_chatbook/Chat/console_chat_start.py
- Tests/Chat/test_console_ordinary_tree.py
- https://github.com/rmusser01/tldw_chatbook/pull/3548
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40715's notes list it as "found, not fixed". Checked on dev 5e3c8b2bd4 (2026-11-16) by reading the code; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/`.

**Why.** In the ordinary-tree test host, agent runs are cancelled before they reach the model. When project instructions need a confirmation and the session's chat start has been accepted, the dispatch returns "cancel" without ever asking. If a real Console can reach that state, an agent run there is cancelled with no prompt shown. If only the test host can reach it, the host is hiding everything after that point in the tests that use it. Either way, part of TASK-41028's red `Tests/Chat/test_console_ordinary_tree.py` nodes may come from this.

**Evidence.**
- `Chat/console_project_dispatch.py:111-123` (`confirm_project_dispatch`): when the first commit returns "prompt", `:114` checks `controller._chat_start.is_accepted(session_id)` and returns "cancel" at `:115`. The confirm callback (`controller._confirm_project_instruction_dispatch`, `:116-122`) is never reached.
- `Chat/console_chat_start.py:104-107`: `is_accepted` is true while the runtime still owns an accepted target start for that session.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The task notes say whether a real Console session can reach this branch with a pending project-instruction prompt, with the evidence (a mounted reproduction, or the code path that rules it out)
- [x] #2 If it can, the user is asked (or told why the run stopped) instead of the run being silently cancelled, pinned by a test that is red on dev; if it cannot, the ordinary-tree test host is fixed so its agent runs reach the model
<!-- AC:END -->

## Implementation Plan

ADR required: no. A test-harness fix, plus findings recorded in these notes; no product behaviour changes.

1. Trace why agent runs in the ordinary-tree host never reach the model (probe the dispatch callback on the base tree).
2. Find whether a real Console session can reach `is_accepted` with a pending consent prompt (agent chat start path), and record the code path and pinning test.
3. If unreachable, fix the shared test double so host agent runs reach the model; show red-on-base nodes go green, and the mutation check.

## Implementation Notes

Outcome: harness, not product. A real Console session cannot reach `console_project_dispatch.py:114-115` with a pending prompt, and the ordinary-tree host's cancellations never came from that branch.

- **Why the host's agent runs stopped.** A probe on base fbf1c9ea74 (`test_original_unknown_library_factory_is_opaque_before_construction[notes_read]`) logged `is_accepted(...) == False`. The cancel came one step earlier: `confirm_project_dispatch` reads `active_resolution.execution_key` (`:76`), the shared `_RecordingGateway` double (`Tests/Chat/test_console_skill_substitution.py`) returned a resolution without that field, and the `AttributeError` was swallowed by `AgentService._run_one` (`Agents/agent_service.py`, the `except Exception: decision = "cancel"` around the consent call), which persisted the run `cancelled` with no steps. TASK-40715's note that a child run also needs `continuation_protocol` is the same double.
- **Why a real session cannot reach the branch.** `is_accepted` is true only for an accepted agent chat start (`Chat/console_chat_start.py`; `create_chat` mode `start`). That start is accepted only in `_controller_accept_durable_turn` (`Chat/console_native_commit.py`, `self._chat_start.accept(...)`), which first calls `_native_start_project_required_action` (`Chat/console_chat_controller.py`). That check refuses the start before acceptance with `project_consent_required` whenever the session's stored notice key differs from the one for its binding and resolution, and with `project_binding_required` when no binding can be selected. The user sees "Background start needs project setup." and the launch status `not_started`. The dispatch then derives the same key from the same selection and resolution, so `commit_and_record(None)` returns `"proceed"`, not `"prompt"`. A binding or state that changes in between returns `"cancel"` from the authority/state check, not `"prompt"`. The only other caller, review delivery (`Chat/console_review_delivery.py`), passes a review destination's resolution; there a `"cancel"` refuses that destination with `destination_consent_required` and the run goes on. Pinned already by `Tests/Chat/test_console_chat_start.py::test_native_project_decision_refuses_before_both_fences[1|2]` (passes on base: not started, reason `project_consent_required`/`project_binding_required`, the consent callback never called). The `:114` branch stays as a fail-closed guard.
- **Fix (test double only).** `_RecordingGateway.resolve_for_send` now returns the routing fields every real `ConsoleProviderResolution` has (`execution_key`, `continuation_protocol`, `api_mode`). The fleet-child test in `test_console_ordinary_tree_widening.py` no longer needs its own resolution override, so it is gone.
- **Evidence.** `Tests/Chat/test_console_ordinary_tree.py` at `-n 4 --timeout=150`, `[True-True]` deselected (load 40-75): base 40 passed, 31 failed, 3 never reported (the run hung; a watchdog killed it). Head 72 passed, 2 failed, and the process exited. Mutation: reverting only the double gives results identical to base (same 31 failures, same 3 hangs). The TASK-41028 hang node `test_original_stop_hook_holds_same_ordinary_tree[True-True]` alone on head: passed in 9 s and exited.
- **Still red on head, not this cause (TASK-41028's):** `test_skill_tool_child_joins_the_original_ordinary_claim` (the run is refused before the model: "Agent run failed: unexpected provider error (ordinary provider maximum widened)") and `test_actual_received_turn_with_native_killed_mcp_is_bounded[True]` (`DID NOT RAISE RuntimeError` at `:1384`). Both fail on base too.
- **Checks (both TASK-41311 and TASK-41312, load 30-75).** Covering files: the 13 Chat files that use `_RecordingGateway` or the storage source, plus the 14 `Tests/Plugins` files on the `native_console` fixture (1330 nodes, `-n 4`). Base and head each had 97 failures, the same set with two exceptions. `test_original_console_mcp_context_reaches_child_model[workspace_custom_reader-1]` is red on base (twice alone) and green on head. `test_real_received_cold_skill_receipt_and_custom_pending_mcp[mounted_new_queued]` failed once in the loaded head run and passed twice alone. `./scripts/preflight.sh` passes. Serial `Tests/Architecture`: 3 failed, 1418 passed on both base and head (the same 3: the diagnostic inventory review and two timer-path inventory tests).
- Files: `Tests/Chat/test_console_skill_substitution.py`, `Tests/Chat/test_console_ordinary_tree_widening.py`.
