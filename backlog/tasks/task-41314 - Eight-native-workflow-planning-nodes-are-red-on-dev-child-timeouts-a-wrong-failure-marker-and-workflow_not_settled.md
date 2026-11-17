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
- [ ] #1 Each of the 8 nodes is classified as test drift (naming the commit) or a product defect (fixed, with the test red first), and the classification is recorded in the task notes
- [ ] #2 All 8 nodes pass on dev at `-n 4`, or are retargeted with the reason recorded
- [ ] #3 The file runs on a PR lane, or the task notes record why it should not
<!-- AC:END -->

## Implementation Plan

ADR required: no. The cause of all three groups is ADR-356 (TASK-39147), which is Accepted; the retargets below assert its contract and add no new decision. Group (a) needs an owner decision and is left unchanged (see notes).

1. Reproduce each group on the base tree with full child logs (the `_run` window shows only the last 1000 characters). Run the same probe on the pre-rebuild original lineage (`e08627ac38`, local `dev`) to show the nodes passed before the rebuild.
2. (b) `required-write`, `required-return`, `late-source-return`: the failure lands after positive physical return, so ADR-356 decision 6 retires the owner and hands the tail to a recovery receipt. Move these three into their own test that asserts that contract: the same owner/claim/error in the receipt, workflow `recovery_required`, charges and committed plan kept, never a settled drain. Keep `worker-failure` and `response-close-failure` (failed native scopes) on the retained-custody test.
3. (c) shutdown: keep the wait assertions and add that disposal is still pending when the held worker is released. After release, assert the actual ADR-356 outcome (interrupted usage is uncertain, planning refuses settlement and hands its returned owner to recovery).
4. (a) replaced receivers: stop and report options; the property depended on implementation-identity pins that ADR-356 decision 4 removed.
5. Run the file on base and head at `-n 4`; preflight; Tests/Architecture.
