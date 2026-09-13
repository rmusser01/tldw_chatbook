# TASK-26945 Ruff Chat Console Context Cleanup Implementation Plan

> **For Codex:** Use `superpowers:verification-before-completion` and `superpowers:requesting-code-review` before closeout.

**Goal:** Apply Ruff 0.15.22's formatter to the exact twelve TASK-26000 Console context paths while proving Python semantics, comments, and formatter directives are unchanged.

**Architecture:** Treat TASK-26000's batch manifest as the ownership boundary. Reconcile every recorded path against current `origin/dev`, capture Python 3.12.11 structural evidence after the separately committed TASK-30040 baseline-test repair, run Ruff only on those twelve paths, compare the same evidence after formatting, and use the recorded eight-module focused suite as behavioral evidence.

**Tech Stack:** Python 3.12.11, Ruff 0.15.22, pytest, standard-library `ast` and `tokenize`, Git, Backlog.md.

**Spec:** `Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md`

## Global Constraints

- The assigned path set is exact; do not format or change any unassigned Python path.
- Normalize only `ast.TypeIgnore.lineno` when comparing AST dumps.
- Preserve ordered comment tokens, inline directive anchors, standalone Ruff directive adjacency, and `# fmt: off` / `# fmt: on` enclosed-node intervals.
- Do not make handwritten production behavior changes.
- TASK-30040 is a separate committed test-fixture repair; TASK-26945's formatter baseline begins after that commit.
- Use the exact focused test modules owned by this batch. Do not run the full suite without user opt-in.

## Task 1: Reconcile Ownership and Capture the Baseline

**Files:**

- Inspect: `backlog/tasks/task-26945 - Clean-Ruff-formatter-debt-for-ruff-chat-console-context.md`
- Inspect: `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`
- Inspect: all twelve Assigned Paths in TASK-26945

1. Verify the branch merge-base is the fetched current `origin/dev` and record both current and TASK-26000 pinned commits.
2. Confirm all twelve paths exist and inspect history for deletions, renames, upstream modifications, or already-formatted files.
3. Record that four paths changed between the TASK-26000 pin and current `origin/dev`, with no assigned-path rename or deletion.
4. Record TASK-30040's separate repair/format commit for `Tests/Chat/test_console_memory_selection.py`; keep the path in the batch and include it in every guard and Ruff command.
5. Capture AST, ordered comments, directive anchors, and formatter-range evidence for all twelve paths using Python 3.12.11.

## Task 2: Format Only the Assigned Paths

**Files:**

- Modify: the exact twelve Assigned Paths, when Ruff changes them

1. Run Ruff 0.15.22 `format` with the twelve paths supplied explicitly.
2. Assert the Python diff outside the twelve-path allowlist is empty.
3. Compare post-format structural evidence against the baseline and stop on any mismatch.

## Task 3: Run Focused and Governance Verification

**Files:**

- Verify: the exact twelve Assigned Paths
- Verify: the eight assigned test modules

1. Run Ruff 0.15.22 lint and format checks on all twelve paths.
2. Run the eight assigned Console context test modules in one Python 3.12.11 pytest command.
3. Run `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
4. Confirm no unassigned Python path changed and inspect the formatter diff for behavioral edits.

## Task 4: Commit, Review, and Close TASK-26945

**Files:**

- Modify: `backlog/tasks/task-26945 - Clean-Ruff-formatter-debt-for-ruff-chat-console-context.md`

1. Commit only the formatter-owned Python changes.
2. Request an independent review of the formatting commit and address every Critical or Important finding.
3. Add exact drift, structural, Ruff, focused-test, and governance results to Implementation Notes.
4. Check all acceptance criteria, set TASK-26945 to Done, and commit only the closeout record.

ADR required: no
ADR path: N/A
Reason: This is mechanical formatter cleanup under TASK-26000's accepted contract and introduces no architectural, persistence, security, dependency, or long-lived UX decision.
