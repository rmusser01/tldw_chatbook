# Fast Lane db424 CI diagnosis

Read-only review of [job 103961757003](https://github.com/rmusser01/tldw_chatbook/actions/runs/34839768380/job/103961757003), run34839768380, exact head `db4247b8db98179a9f8bc762572d83d3bea0b254`.

The Fast Lane job reports **1 failed, 1,128 passed** in652.11s. Sole failure: `Tests/CI/test_task2062_2_gguf_source_evidence.py::test_workflow_is_one_read_only_exact_three_os_matrix`, line128. Its exact tuple expects five step names; actual workflow includes a sixth, “Diagnose Windows backup startup against the dev source baseline”. That approved diagnostic was added in d3082ef982; this older contract was not updated. All23 execution-capacity cases passed in this same CI job (log line581), so this is not recurrence of the worker-cleanup regression.

Current workflow/test files are unchanged against db424. Focused local reproduction yields exact same failure: **1 failed in0.70s**, `/private/tmp/uat-fast-lane-db424-focused.log`. No source edits or test filtering changes were made. Downloaded job log: `/private/tmp/uat-fast-lane-db424-job.log`; run metadata: `/private/tmp/uat-fast-lane-db424-run.json` (other run jobs were still in progress when read).

## Minimal proposed test-only contract correction

Keep the original five-step tuple as the required exact prefix; require exactly one final diagnostic step rather than allowing arbitrary appended steps. Apply the existing strict key and no-continue-on-error checks to all original steps unchanged. Merely adding the new name is insufficient: current allowed keys reject `if`/`continue-on-error`, and the global text prohibition rejects any continuation directive.

Qualify the final diagnostic separately with exact allowed keys `{name, if, continue-on-error, run}`, exact condition `failure() && runner.os == 'Windows'`, and strict boolean `continue-on-error is True`. Keep job permissions contents:read, original three-OS matrix,20-minute enclosing job, triggers, original checkout SHA and dependencies, original exact18-node command, all four fresh-process UI nodes, each60-second timeout, and original failure propagation unchanged. No continuation or additional condition belongs on any primary test step.

Assert the diagnostic's existing command contract exactly: fetch pinned4631b60f8dd9623fc55bf16f4a37e29fcb1240c7, detached baseline worktree under RUNNER_TEMP, existing candidate-owned startup_timing_diagnostic.py, one fixed GGUF UI node, one dev and one candidate invocation with distinct explicit --source and --label, accumulate failed status, then exit that status. Keep other existing global forbidden tokens/actions/secrets/write permissions untouched; scope the `continue-on-error` exception only to this one inspected diagnostic mapping. Do not loosen the checker to arbitrary steps, scripts, conditions or error suppression.

Useful contract negatives, if root chooses to add tests: move continuation to a primary step, remove Windows/failure gating, broaden the diagnostic source/node, or append a second diagnostic; each should fail the same semantic checks. This is a test-contract synchronization proposal, not approval to change the workflow, timeouts, product behavior, or diagnostics.

## Independent review of root's implemented contract update

APPROVED, no actionable findings. Reviewed only the working-tree delta in `Tests/CI/test_task2062_2_gguf_source_evidence.py`, SHA-256 `daa75e510fd231e1dba7b20c21b3593fca12574af423ed25890a67ca3fbd812d`. The original five steps retain their allowed fields, exact commands, 60-second case deadlines, matrix, permissions, and failure semantics. Exactly one final diagnostic has the fixed Windows failure condition, boolean continuation, pinned baseline, and exact two observer invocations. The workflow itself is unchanged.

Independent focused verification: **5 passed in 0.70s**, `/private/tmp/uat-fast-lane-contract-independent.log`. A disposable in-memory contract probe rejected all four mutations: continuation added to an original step, Windows gating removed, diagnostic node broadened, and an extra appended step. No repository source was modified by review. Ruff passed. Bandit baseline/current contains only B101 test assertions (29/34); no new non-assert findings. Bandit reports are `/private/tmp/uat-fast-lane-contract-baseline-bandit.json` and `/private/tmp/uat-fast-lane-contract-independent-bandit.json`.
