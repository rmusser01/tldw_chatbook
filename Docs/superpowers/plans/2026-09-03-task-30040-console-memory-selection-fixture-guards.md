# TASK-30040 Console Memory-Selection Fixture Guard Repair Implementation Plan

> **For Codex:** Use `superpowers:test-driven-development` for the existing red regression and `superpowers:verification-before-completion` before closeout.

**Goal:** Restore the Console memory-selection regression module after ADR-097's fail-closed semantic mutation guard made four deliberate raw-row corruption fixtures invalid.

**Architecture:** Keep the production persistence boundary unchanged. Add one test-local helper that borrows the database's existing connection-scoped coordinator capability, authorizes only `message_update` for one named message inside a managed transaction, and executes one deliberate fixture mutation. Route only the guarded raw updates through that helper; leave ordinary test behavior and unguarded visibility/version setup unchanged.

**Tech Stack:** Python 3.12.11, pytest, SQLite, Ruff 0.15.22.

**Spec:** `backlog/tasks/task-30040 - Repair-Console-memory-selection-corruption-fixtures-for-semantic-guards.md`

## Global Constraints

- No production code or mutation-trigger changes.
- Keep authorization test-only, transaction-scoped, message-specific, and limited to `message_update`.
- Do not replace the real SQLite guard callback or install a blanket allow-all function.
- Verify only the affected module and repository-required governance checks; do not run the full suite without user opt-in.
- Preserve TASK-26945's formatter-only AST-equivalence baseline by landing this repair separately first.

## Task 1: Preserve the Red Regression Evidence

**Files:**

- Inspect: `Tests/Chat/test_console_memory_selection.py`
- Inspect: `tldw_chatbook/DB/base_db.py`
- Inspect: `tldw_chatbook/DB/migrations/chachanotes_v56_to_v57_semantic_mutation_guard.sql`

1. Run the assigned Console memory-selection module on current `origin/dev`.
2. Confirm the four failures occur at deliberate raw updates of referenced semantic message columns and raise the semantic-mutation authorization error.
3. Confirm the test file predates the ADR-097 mutation guard and that no open PR already owns the repair.

## Task 2: Add the Narrow Test-Only Mutation Helper

**Files:**

- Modify: `Tests/Chat/test_console_memory_selection.py`

1. Add a typed helper that accepts the database, message ID, SQL, and parameters.
2. Inside `db.transaction()`, obtain the exact connection's private coordinator capability and authorize only `{\"message_update\"}` for the supplied message ID.
3. Execute the caller's single mutation through the managed cursor.
4. Replace only the four failing guarded fixture writes with helper calls.

## Task 3: Verify and Close the Repair

**Files:**

- Modify: `backlog/tasks/task-30040 - Repair-Console-memory-selection-corruption-fixtures-for-semantic-guards.md`

1. Run `pytest -q Tests/Chat/test_console_memory_selection.py` with Python 3.12.11 and record the exact result.
2. Run Ruff 0.15.22 lint and format checks on the touched Python file.
3. Run `pytest -q Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
4. Self-review the diff, mark every acceptance criterion complete, add concise implementation notes, and set TASK-30040 to Done.

ADR required: no
ADR path: N/A
Reason: This is a test-fixture compatibility repair under the existing ADR-097 mutation boundary and changes no persistence architecture or product behavior.
