---
id: TASK-41314
title: "Eight native workflow-planning nodes are red on dev: child timeouts, a wrong failure marker and workflow_not_settled"
status: In Progress
created_date: 2026-11-16 19:55
labels:
- follow-up
- tests
- agents
- workflows
priority: medium
references:
- Tests/Chat/test_workflow_planning_native.py
- tldw_chatbook/Chat/console_workflow_preparation.py
- tldw_chatbook/DB/automatic_work.py
- tldw_chatbook/DB/workflow_runs.py
- https://github.com/rmusser01/tldw_chatbook/pull/3543
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the 2026-11-13..16 burn-down. TASK-40718 (PR #3543) found the same 8 nodes failing on its base and its head at load 12-16. Re-run on dev 5e3c8b2bd4 (2026-11-16, load 36-43, scratch-profile wrapper): all 8 still fail. Line numbers are that tree's.

**Why.** These nodes are the native planning custody controls: replaced receivers are refused without effects, required failures keep the original custody, and shutdown waits for the planning worker. While they are red they prove none of that, and they run on no PR lane. TASK-40718 judged them dev-red rather than load-sensitive.

**Nodes and failures (`Tests/Chat/test_workflow_planning_native.py`).** Each runs a private child process.
- `test_native_planning_refuses_replaced_receivers_without_effects` (4 params: `executor-replaced`, `reader-replaced`, `executor-hash-replaced`, `executor-resolver-replaced`): the child times out (`subprocess.TimeoutExpired` after 60 s).
- `test_native_planning_required_failure_retains_original_custody` (`required-write`, `required-return`, `late-source-return`): `pytest.raises(AssertionError, match="F_REQUIRED_CUSTODY_RETAINED_<mode>")` at `:92` does not match. For `required-write`, the child exits 1 at its own `exercise` assertion (script line 653) after `sqlite3.IntegrityError: required plan write failed` (`DB/workflow_runs.py:339`, `_save_plan`), instead of reporting the retained-custody marker.
- `test_shutdown_waits_for_same_native_planning_worker_return`: the child exits 1. The planning worker's `settle_workflow_phase` (`Chat/console_workflow_preparation.py:2676`) raises `AutomaticWorkRefused("workflow_not_settled")` (`DB/automatic_work.py:3049`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each of the 8 nodes is classified as test drift (naming the commit) or a product defect (fixed, with the test red first), and the classification is recorded in the task notes
- [x] #2 All 8 nodes pass on dev at `-n 4`, or are retargeted with the reason recorded
- [ ] #3 The file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. The cause of all three groups is ADR-356 (TASK-39147), which is Accepted; the retargets below assert its contract and add no new decision. Group (a) needs an owner decision and is left unchanged (see notes).

1. Reproduce each group on the base tree with full child logs (the `_run` window shows only the last 1000 characters). Run the same probe on the pre-rebuild original lineage (`e08627ac38`, local `dev`) to show the nodes passed before the rebuild.
2. (b) `required-write`, `required-return`, `late-source-return`: the failure lands after positive physical return, so ADR-356 decision 6 retires the owner and hands the tail to a recovery receipt. Move these three into their own test that asserts that contract: the same owner/claim/error in the receipt, workflow `recovery_required`, charges and committed plan kept, never a settled drain. Keep `worker-failure` and `response-close-failure` (failed native scopes) on the retained-custody test.
3. (c) shutdown: keep the wait assertions and add that disposal is still pending when the held worker is released. After release, assert the actual ADR-356 outcome (interrupted usage is uncertain, planning refuses settlement and hands its returned owner to recovery).
4. (a) replaced receivers: stop and report options; the property depended on implementation-identity pins that ADR-356 decision 4 removed.
5. Run the file on base and head at `-n 4`; preflight; Tests/Architecture.

## Implementation Notes

All 8 nodes are test drift. None is a product defect. Evidence: full child logs on the base tree (`fbf1c9ea74`), and the same probe run on the pre-rebuild original lineage `e08627ac38` (local `dev`), where all 8 pass in 22-46 s. The replayed PR #3480 commits before its integration commit cannot run: `68adde3960` fails at import (`config.py` reads `get_canvas_execution_enabled` as a bare name; `console_agent_bridge` names an undefined `code_review_provider`).

**(a) `refuses_replaced_receivers` ×4: drift from `06f8b0bf94` (TASK-39147, ADR-356 decision 4), merged to dev by PR #3480 (`e10ce63fe8`). Deleted by owner decision (2026-11-17).** This is not load. Run alone with a 300 s budget at load 30-52, `executor-replaced`, `reader-replaced` and `executor-resolver-replaced` still hang. The planning `fs_list`/`fs_read` calls now reach an MCP approval round, and the private App is not running (`on_pending_rounds_changed` -> `call_from_thread`: "App is not running"), so the round never resolves. `executor-hash-replaced` finishes in 31 s, but the replaced `__hash__` ran and wrote its file. The old refusal compared implementation-identity pins (`_INPUT_FUNCTIONS`, `resolve_workspace_path` among them; the 11-03 ruling records "47 actual pre/post pins equal"). ADR-356 removed those pins on purpose ("they do not provide a security boundary against arbitrary in-process code"). No retarget keeps the property.

Owner decision, 2026-11-17: delete the 4 params as superseded by ADR-356 decision 4. Authority now comes from runtime-owned registration and current state, not from code identity (`06f8b0bf94`). The params were the whole of `test_native_planning_refuses_replaced_receivers_without_effects`, so that function is deleted too. The probe code that only those modes used is removed as well:
- the receiver replacement in `provider()`;
- their tool-call response branch (`tools` keeps it);
- their end-of-run receipt assertions;
- their entries in the no-plan set.

**(b) `required-write`, `required-return`, `late-source-return`: drift from `06f8b0bf94` (ADR-356 decision 6, TASK-39147 AC #9). Retargeted.** Each failure lands after positive physical return. `run_planning` then calls `_retire_returned_planning`, which retires the owner once and records `recovery_required`. The child's own assertion (`primary_owner.is_admitted`) failed before it could print the marker. These three now run in `test_native_planning_returned_required_failure_hands_custody_to_recovery`. It asserts:
- the recovery receipt holds the same owner, claim and original error (so the failure is not hidden);
- the owner is retired and its claim released, and the folder recovery protection is captured;
- the workflow is `recovery_required`, `generation`/`model_call` charges are kept, and a committed plan survives;
- nothing reports a settled drain.

`worker-failure` and `response-close-failure` are failed native scopes and keep the retained-owner control.

**(c) shutdown: drift. Shutdown still waits for the same worker.** `01c1a0e332` (TASK-33666, on dev since #3169) lets Stop/quit end a reply's wait. The planning lineage only got it with the #3480 integration (`6125fbc5b8`). On shutdown, the stall watchdog now throws into the stream while the held worker returns. `_automatic_generation` then records the usage as interrupted (`tokens` `uncertain`), so `settle_workflow_phase` refuses `workflow_not_settled` and the `06f8b0bf94` recovery path runs. On `e08627ac38` the same mode settles tokens at 10. The test keeps the 0.1 s and 5 s wait checks, adds that disposal is still pending when the worker is released, and asserts the recovery outcome.

**Checks:**
- Mutations: skipping `record_planning_recovery` fails all 4 retargeted nodes. Making dispose skip the planning driver fails the shutdown node.
- Base vs head on the whole file at `-n 4`: base had 9 failed / 22 passed (the 8 + maintenance-tail, load 66-84). Head had 7 failed / 24 passed (the 4 (a) nodes + maintenance-tail, worker-tail and repair-instruction, load 73-84).
- Rerun alone at load 62-72, worker-tail and repair-instruction pass. `test_native_runtime_waits_for_original_maintenance_operation_return` times out alone on base too. It is outside this task.

**Preflight and Architecture.** `./scripts/preflight.sh` passes. On head, the serial `Tests/Architecture` run had 7 failed / 1414 passed at load 50-95. Three of those fail on base too: `test_reviewed_diagnostic_changes_are_metadata_only` and the two `test_timer_path_static_update_inventory` nodes. The other four (`test_task_15103_review_ledger_*`) pass when rerun alone on head (45 passed).

**After the deletion.** The whole file on head at `-n 4`: 27 passed, 0 failed in 283 s at load 50-107, maintenance-tail included.

**AC #3 is not ticked.** Workflow files are not edited here. The lane is recorded in the hand-off.

Files: `Tests/Chat/test_workflow_planning_native.py`, `Tests/Chat/workflow_planning_probe.py`.
