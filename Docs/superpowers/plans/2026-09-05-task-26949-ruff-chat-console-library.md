# TASK-26949 Ruff Chat Console Library Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Ruff 0.15.22's formatter to the exact nine TASK-26000 Console library paths while proving Python semantics, comments, formatter directives, and focused behavior are unchanged.

**Architecture:** Treat TASK-26000's batch manifest as the immutable ownership boundary. Reconcile every recorded path against current `origin/dev`, capture Python 3.12.11 structural and focused-test evidence, run Ruff only on those paths, replay formatter output from immutable branch-base blobs, and close the task only after exact evidence parity and independent review.

**Tech Stack:** Python 3.12.11, Ruff 0.15.22, pytest, standard-library `ast` and `tokenize`, Git, Backlog.md.

**Spec:** Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md

## Global Constraints

- The nine paths in TASK-26949's Assigned Paths JSON are the exact allowlist: five `Tests/Chat` modules and four `tldw_chatbook/Chat` modules. Do not format or change any unassigned Python path.
- The canonical sorted-path digest is `cd664f7c1688da479c38f4c47b8c5b2c997aab2320c9ceef1e2a73bd12476b90`.
- Use Python 3.12.11 and Ruff 0.15.22 from `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/`.
- Normalize only `ast.TypeIgnore.lineno` when comparing AST dumps produced with `type_comments=True` and `include_attributes=False`.
- Preserve ordered comment tokens, inline directive anchors, standalone Ruff directive adjacency, and `# fmt: off` / `# fmt: on` enclosed-node intervals.
- Compute each inline directive's significant-token position from its nearest logical owner: a uniquely validated same-line `except` clause for an `ExceptHandler` header, otherwise its nearest containing AST statement. Exclude only parenthesis pairs proven AST-neutral by an independent shadow parse/dump comparison.
- Do not make handwritten production behavior changes or repair inherited focused-test failures.
- Capture the untouched current-base focused-test inventory before formatting and require the post-format normalized failure keys to match exactly.
- Use the exact five assigned test modules; do not run the full suite without user opt-in.
- Do not edit the worktree while any baseline or quoted verification run is in flight.
- After every rebase, rerun base-sensitive structural, replay, generated-inventory, governance, Ruff, and focused-test gates even if the assigned-path diff is unchanged.
- ADR required: no. This task directly implements TASK-26000's existing formatter contract and changes no architecture, schema, storage, security, dependency, or long-lived UX boundary.

---

### Task 1: Reconcile, Format, and Verify the Assigned Python Batch

**Files:**

- Inspect: `backlog/tasks/task-26949 - Clean Ruff formatter debt for ruff-chat-console-library.md`
- Inspect: `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Modify: only the assigned Python paths changed by Ruff 0.15.22
- Temporary evidence: `/tmp/task26949_before.json`, `/tmp/task26949_before.xml`, `/tmp/task26949_after.xml`

**Interfaces:**

- Consumes: batch label `ruff-chat-console-library`, authority cut `e555df102c950c29beed5e7119f433d35eee1f3c`, current branch base, and the nine-path digest above.
- Produces: one commit containing only deterministic Ruff output on assigned Python paths plus an evidence report with drift, structural, focused-test, lint, replay, diagnostic, and governance results.

- [x] **Step 1: Reconcile the manifest against the exact branch base**

  Parse the task and canonical evidence JSON, require nine unique paths (five tests and four production paths), recompute the canonical JSON-list digest, require every path to exist, and inspect `git diff --name-status --find-renames e555df102c950c29beed5e7119f433d35eee1f3c HEAD -- <paths>`. Record every modified, renamed, deleted, or already-formatted path; retain an upstream-clean path in the allowlist and structural proof.

- [x] **Step 2: Capture the structural baseline**

  Use the independently tested version-3 guard at `/tmp/task26947_format_guard.py` (SHA-256 `3fac070e94fe91cd152f956b19093c457c48787ea5449b54945b2305386b7471`) with Python 3.12.11 to capture `/tmp/task26949_before.json` for all nine explicit paths. The guard must enforce the Global Constraints above and fail closed on ambiguous directive ownership.

- [x] **Step 3: Capture the focused behavioral baseline**

  Run exactly:

  ```bash
  LOGURU_LEVEL=ERROR /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q --tb=line --disable-warnings --junitxml=/tmp/task26949_before.xml Tests/Chat/test_console_chat_store_library_policy.py Tests/Chat/test_console_library_activity_buffer.py Tests/Chat/test_console_library_destination.py Tests/Chat/test_console_library_policy_coordinator.py Tests/Chat/test_console_library_runtime_policy.py
  ```

  Record the exit code, totals, and normalized JUnit failure/error keys. Treat any red baseline as inherited evidence, not permission to change behavior.

- [x] **Step 4: Apply Ruff only to the allowlist**

  Invoke `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format` once with all nine assigned paths supplied explicitly. Do not use `.` or a directory operand.

- [x] **Step 5: Require structural and scope parity**

  Compare all nine paths against `/tmp/task26949_before.json` with the same guard. Require AST, ordered comments, inline directive attachment/position, standalone Ruff directive adjacency, and formatter-range equality. Require every changed Python path to belong to the nine-path allowlist and review the diff for handwritten behavior edits.

- [x] **Step 6: Run exact post-format verification**

  Run Ruff `check` and `format --check` on all nine paths. Re-run the exact Step 3 pytest command with only the JUnit output changed to `/tmp/task26949_after.xml`, and require identical normalized failure/error keys. Run `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/CI/test_backlog_task_id_uniqueness.py`, `git diff --check`, and `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/check_persistent_diagnostic_inventory.py --diff`.

- [x] **Step 7: Replay deterministic formatter bytes from the immutable base**

  For each assigned file Ruff changed, read the branch-base blob with `git show <branch-base>:<path>`, format that blob through Ruff 0.15.22 using `--stdin-filename <path> -`, and require the bytes to equal the worktree file. If an assigned path needs a separately justified safe lint fix, keep that proof and commit separate from the formatter replay.

- [x] **Step 8: Commit and report**

  Commit only the assigned Python paths changed by Ruff. Write the implementer report with the exact commit, changed/unchanged paths, every command and result, normalized failure-key comparison, and any persistent-diagnostic drift. Do not edit task closeout documentation in this commit.

### Task 2: Review Evidence and Close TASK-26949

**Files:**

- Modify: `backlog/tasks/task-26949 - Clean Ruff formatter debt for ruff-chat-console-library.md`
- Modify: `Docs/superpowers/plans/2026-09-05-task-26949-ruff-chat-console-library.md`
- Modify only if Task 1 proves source-layout-only drift: `Docs/security/production-diagnostic-inventory.json`

**Interfaces:**

- Consumes: Task 1's commit, report, and clean task-scoped review.
- Produces: checked acceptance criteria, concise implementation notes, a Done task record, and a merge-ready branch.

- [x] **Step 1: Validate the evidence package**

  Confirm Task 1's report contains the exact base, digest, lineage, structural result, Ruff output, focused before/after counts and normalized keys, byte replay, persistent-diagnostic result, backlog guard, and `git diff --check`. Resolve every task-review Critical or Important finding through the bounded SDD fix loop.

- [x] **Step 2: Handle only proven derived-artifact drift**

  If `scripts/check_persistent_diagnostic_inventory.py --diff` reports a source-layout-only change, inspect the affected statements, run the checker with `--statements <affected paths> --since <branch-base>`, regenerate with `--write`, re-run `--diff`, and commit only the generated inventory. Do not accept semantic diagnostic changes in this formatter task.

- [x] **Step 3: Close the Backlog record**

  Add concise Implementation Notes containing every exact verification command/result and the focused-test rationale. Check all eight acceptance criteria, set status to `Done`, update the date, and state the ADR determination. Mark completed plan steps accurately.

- [x] **Step 4: Review and prepare integration**

  Commit the task/plan closeout separately, generate a whole-branch review package from the immutable branch base, obtain an independent final review, and address all Critical or Important findings before using `superpowers:finishing-a-development-branch` for PR integration.

## Post-rebase Integration Evidence

The historical initial base `2b4973971e5dcf101c5a6ddcc55aa082ff22f814` advanced to `56376e1fc188938bf350c62d3a9f95e820b93c40`, now both `origin/dev` and the branch merge-base. The pre-refresh rebased head is `edeffdd357efe7a9d164a7fec17460b288738ae8`. Zero drift was found between the bases across all nine assigned Python paths and the task/plan documents, and all nine base blobs were independently byte-identical. `git range-diff` confirmed the four patch-equivalent mappings: `dfafa72930433b95c8eb86f0a8e1496757827c79` -> `8a4d5c542b1eca71aed23adde972b179e6b9e6f4`, `5102a54f71b4d7f2b5fa3543ab76b2cf70a52726` -> `6044f645e65103d8dc9c07f2786606ee244ded4e`, `8a565a91cb9a9a0018664e4b0947ed8e13b8e9fa` -> `d72fe36dc67f5d538750cbf0722777d6384302fd`, and `f0f949338a2aa41320d3391c76efd3ec65743837` -> `edeffdd357efe7a9d164a7fec17460b288738ae8`.

Fresh post-rebase structural comparison, Ruff lint/format, deterministic new-base replay, persistent diagnostic inventory, Backlog uniqueness, and merge-candidate diff gates passed. The exact five assigned test modules passed `103 passed, 2 warnings in 16.13s`; no full suite was run. The prior Qodo Critical 0 / Important 0 / Minor 0 review and green CI applied to old head `f0f949338a2aa41320d3391c76efd3ec65743837`; both will be rerun after force-pushing the rebased branch.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's accepted contract and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
