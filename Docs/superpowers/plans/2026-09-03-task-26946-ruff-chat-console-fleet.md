# TASK-26946 Ruff Chat Console Fleet Cleanup Implementation Plan

> **For Codex:** Use `superpowers:verification-before-completion` and `superpowers:requesting-code-review` before closeout.

**Goal:** Apply Ruff 0.15.22's formatter to the exact ten TASK-26000 Console fleet paths while proving Python semantics, comments, and formatter directives are unchanged.

**Architecture:** Treat TASK-26000's batch manifest as the ownership boundary. Reconcile every recorded path against current `origin/dev`, capture Python 3.12.11 structural evidence, run Ruff only on those ten paths, compare the same evidence after formatting, and use the recorded seven-module focused suite as behavioral parity evidence.

**Tech Stack:** Python 3.12.11, Ruff 0.15.22, pytest, standard-library `ast` and `tokenize`, Git, Backlog.md.

**Spec:** `Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md`

## Global Constraints

- The assigned path set is exact; do not format or change any unassigned Python path.
- Normalize only `ast.TypeIgnore.lineno` when comparing AST dumps.
- Preserve ordered comment tokens, inline directive anchors, standalone Ruff directive adjacency, and `# fmt: off` / `# fmt: on` enclosed-node intervals.
- Do not make handwritten production behavior changes.
- Preserve and report the untouched `origin/dev` focused-test baseline: the fleet-wake path currently dereferences `preparation.capture_mode` for AGENT_WAKE even though that origin intentionally has no preparation object. Do not repair that unassigned controller behavior in this task.
- Use the exact focused test modules owned by this batch. Do not run the full suite without user opt-in.

## Task 1: Reconcile Ownership and Capture the Baseline

**Files:**

- Inspect: `backlog/tasks/task-26946 - Clean Ruff formatter debt for ruff-chat-console-fleet.md`
- Inspect: `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Inspect: all ten Assigned Paths in TASK-26946

1. Verify the branch starts at fetched current `origin/dev` and record both current and TASK-26000 pinned commits.
2. Confirm all ten paths exist and inspect history for deletions, renames, upstream modifications, or already-formatted files.
3. Record that four paths changed between the TASK-26000 pin and current `origin/dev`, with no assigned-path rename or deletion; retain every path.
4. Run the seven assigned test modules before formatting and record the exact pre-existing failure inventory and root-cause evidence.
5. Capture AST, ordered comments, directive anchors, and formatter-range evidence for all ten paths using Python 3.12.11.

## Task 2: Format Only the Assigned Paths

**Files:**

- Modify: the exact ten Assigned Paths, when Ruff changes them

1. Run Ruff 0.15.22 `format` with the ten paths supplied explicitly.
2. Assert the Python diff outside the ten-path allowlist is empty.
3. Compare post-format structural evidence against the baseline and stop on any mismatch.

## Task 3: Run Focused and Governance Verification

**Files:**

- Verify: the exact ten Assigned Paths
- Verify: the seven assigned test modules

1. Run Ruff 0.15.22 lint and format checks on all ten paths.
2. Run the seven assigned Console fleet test modules in one Python 3.12.11 pytest command and compare its failure inventory with the untouched baseline.
3. Run `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
4. Confirm no unassigned Python path changed and inspect the formatter diff for behavioral edits.

## Task 4: Commit, Review, and Close TASK-26946

**Files:**

- Modify: `backlog/tasks/task-26946 - Clean Ruff formatter debt for ruff-chat-console-fleet.md`

1. Commit only the formatter-owned Python changes.
2. Request an independent review of the formatting commit and address every Critical or Important finding.
3. Add exact drift, structural, Ruff, focused-test, and governance results to Implementation Notes.
4. Check all acceptance criteria, set TASK-26946 to Done, and commit only the closeout record.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's accepted contract and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
