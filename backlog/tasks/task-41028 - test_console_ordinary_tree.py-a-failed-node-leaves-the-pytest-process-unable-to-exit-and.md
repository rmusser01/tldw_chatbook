---
id: TASK-41028
title: "test_console_ordinary_tree.py: a failed node leaves the pytest process unable to exit, and 32 of its 75 nodes are red on dev"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:03
updated_date: '2026-11-24 00:00'
labels:
- follow-up
- architecture-review-2026-10-25
- tests
- console
- ci
priority: high
references:
- Tests/Chat/test_console_ordinary_tree.py
- tldw_chatbook/Chat/console_ordinary_tree.py
- tldw_chatbook/Agents/ordinary_tree.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36950.1's covering run in Phase 4 of the 2026-10-25 architecture review: on base 26b2e677dd the run hung for 4 h at 97% in `Tests/Chat/test_console_ordinary_tree.py`, with the workers at 0% CPU. pytest-timeout never fired, and SIGINT did not stop the workers.

Re-checked on dev ecad404bd4 (2026-11-15):
- `-n 4 --timeout=120 --timeout-method=thread` on the file alone stops at 74 of 75 nodes, with all four workers at 0% CPU, until a 420 s watchdog kills it. The node that never reports is `test_original_stop_hook_holds_same_ordinary_tree[True-True]`.
- Run serially on its own, the node reports FAILED about 10 s into the test, and then the pytest process never exits (still alive at 300 s; killed by a watchdog). The hang comes after the test body, where pytest-timeout no longer runs, which is why the 120 s timeout never fires. The review of this filing saw the same shape in `Tests/Chat/test_native_v2_mcp_events.py`.
- The same run fails 32 nodes in 19 tests, none of them through the local `RecoveryRequired` trap. Among them are all 7 params of `test_original_unknown_library_factory_is_opaque_before_construction`, the nodes TASK-36950.1 saw hang. On this base they fail fast instead: six fail at `:507` because no provider call was made (`h.gateway.payloads` is empty), and `[swapped_native]` fails at `:505`.

Why it matters: a process that cannot exit after a failed test is a hang pytest-timeout cannot interrupt. It loses every other result in the run, and an xdist run never ends. The file came with #3480 (TASK-39147) and runs on no PR lane, so CI sees neither problem.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pytest run of Tests/Chat/test_console_ordinary_tree.py alone, serially and at `-n 4`, and of `test_original_stop_hook_holds_same_ordinary_tree[True-True]` by itself, runs to completion and the pytest process exits on its own within the run's timeout, with no worker or interpreter left waiting
- [x] #2 The task notes name the cause of the hang (a product deadlock or a test await) and why pytest-timeout could not interrupt it, and whether the same cause stops `Tests/Chat/test_native_v2_mcp_events.py::test_accepted_context_uses_positive_original_drain_without_reviving_dispatch[negative_drain]` from exiting
- [x] #3 Every node of Tests/Chat/test_console_ordinary_tree.py passes on dev at `-n 4` (a node may first be deleted or retargeted, with its reason in the task notes); each red node first gets a named cause
- [x] #4 The file runs on a PR lane, or the task notes say why not
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-check the hang and the red nodes on current dev.
2. Name each remaining red node's cause and fix it in the product or the test.
3. Compare the code-review and ordinary-tree suites with dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Picked up after the light CI tier (TASK-42521) stopped #3688 on this file. Checked on dev 44e7a47e4f.

**AC #1 and #2, the hang.** It no longer happens. TASK-41135 named the cause: pytest-timeout 2.4 cancels its timer in `pytest_exception_interact`, so a failed call left the teardown untimed. It fixed that in `Tests/private_profile.py`. TASK-41312 fixed the harness double that made the stop-hook nodes fail in the first place.
- On this branch the file passes serially (75 passed, 205 s) and at `-n 4` (75 passed, 64 s), and both processes exit on their own.
- `test_original_stop_hook_holds_same_ordinary_tree` alone: 4 passed, and the process exits.
- `Tests/Chat/test_native_v2_mcp_events.py::test_accepted_context_uses_positive_original_drain_without_reviving_dispatch[negative_drain]` also exits (27 s). It still fails, on dev too, for its own reason; TASK-41061 tracks it.

**AC #3, the last two red nodes.** Both came with #3480 and were never green on dev.

- `test_skill_tool_child_joins_the_original_ordinary_claim`: **a product bug.**
  - A turn the multi-model review classifies as a review composes `CodeReviewToolProvider` after the turn's maximum is captured. That covers any code-review wording, with the `[tools] code_review_enabled` gate on (its default) and one bound local root.
  - The provider's `code_review` source was never in the ordinary tree's `other_maximum`, so `bind_catalog` refused the whole run: "ordinary provider maximum widened". In the app that reads "Agent run failed: unexpected provider error".
  - **Fix.** `_capture_other_providers` (`Chat/console_ordinary_tree.py`) projects the review tool names the catalog permits into the turn's maximum. Their availability alone stays bounded, like sub-agents.
    - The projection does not read the review gate. Reading it would load the review service on every send, which TASK-33620.15.53 keeps off ordinary turns. With the gate off, the service lists nothing anyway.
    - Any other review call (`prepare`, `read`, `submit`) widens the turn's claim to OPAQUE in `OrdinaryTurn.check_catalog_call` (`Agents/ordinary_tree.py`), as a skill install does (TASK-40715). Prepare and read run git in the bound root, which can read a linked worktree's objects outside it, and every review then spawns reviewers, which widen anyway.
    - `code_review_guide` (guidance text) and `code_review_release` (always refused with `native_discard_required`) do not widen.
    - While another turn holds the workspace, the call is refused with the existing `widen_refusal` text, and the turn carries on.
  - **Test change.** The test's script also predated on-demand schema disclosure (2d9504cfd9), so it now calls `load_tools` before the skill.
- `test_actual_received_turn_with_native_killed_mcp_is_bounded[True]`: **retargeted and renamed** to `..._is_bounded_only_with_witnessed_metadata`.
  - It expected a refusal from hook-review publication ("Hook review source or session changed") after the accepted request's MCP metadata is replaced. The turn never reaches `begin_hook_review` in either variant: instrumented, there were zero calls.
  - With the replaced (incomplete) metadata, `original_mcp_capture_matches` fails, so the MCP metadata is not witnessed and the turn runs under an OPAQUE claim. That is the designed response: it can never run bounded. The node now asserts bounded for `[False]` and opaque for `[True]`.
  - The run does not check that `record.request` is the request it accepted. The only writer is the received path itself (`console_received_dispatch.py`), right before the call, so this is not a reachable regression.

**New coverage.** `test_review_tool_widens_the_claim_or_is_refused_while_another_turn_runs` (`Tests/Chat/test_console_ordinary_tree_widening.py`):
- After `load_tools`, the guide and release leave the claim bounded.
- Uncontended, prepare widens the claim and then runs. The host has no review team, so the service, not the claim, refuses it with `review_team_empty`.
- Contended, prepare is refused with the busy text.

**Negative controls.**
- Without the change to `Agents/ordinary_tree.py`, both cases of the new test fail.
- Without converting `CapacityRefused`, the contended case fails: the model sees the bare `worktree_activity_busy` instead.
- Without the guide or release exemption, the uncontended case fails at that call.
- Without the projection, the review turn and the skill-child test are refused before the model, as on dev.

**AC #4, not laned.**
- A PR that changes this file, the widening file or a test file they import runs it whole in the light tier (TASK-42521).
- The release qualification (`.github/workflows/test.yml`, `pytest Tests --ignore=Tests/UI`) runs it. Owner intent (ADR-366 amendment) is that dev PRs merge after sanity checks, with nightly and release runs as the broad gates.
- A lane would add about 3.4 min serially to every full-tier run.

**Evidence.**
- 22 `Tests/Agents/test_code_review_*.py` files, the 2 `Tests/Chat/test_console_code_review_*.py` files, the widening file and two first-send census files, at `-n 4`: 808 passed / 1 failed on this branch at first review, and 806 / 1 on a clean dev export, with the same failure (`test_code_review_boundary_ledger.py::test_v24_upgrade_and_reopen_preserve_ordinary_history`).
- After the review fixes, the same set plus this file and `Tests/Chat/test_console_first_send_review_imports.py`: 909 passed, with only that failure.
- Tests/Architecture (serial): 1496 passed, 5 skipped.
- Local review: 1 Important finding (the gate read loaded the review service on every send) and 4 Minors (a capture/composition gate race, the widening comment and release, the test name, an unasserted prepare result). All fixed.
- Ruff `--select F` and `Tests/lint/ruff_gate.py` pass. The two ASYNC240 findings in this test file are on dev too.
<!-- SECTION:NOTES:END -->
