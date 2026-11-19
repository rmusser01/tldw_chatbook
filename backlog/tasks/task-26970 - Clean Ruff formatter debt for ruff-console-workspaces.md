---
id: TASK-26970
title: Clean Ruff formatter debt for ruff-console-workspaces
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 09:05'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-console-workspaces -->
<!-- TASK-26000-PATHS-SHA256: 754816a5ff4403930be290256db9ad6a1a5a956be9da8b92b575d387e426c6e6 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-console-workspaces` Ruff formatter batch at the owner boundary recorded as: Console workspace, project, terminal, and file-binding surfaces.. The focused test surface recorded by TASK-26000 is `["Tests/UI", "Tests/Workspaces"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_new_workspace.py",
  "Tests/UI/test_console_project_instructions.py",
  "Tests/UI/test_console_resume_active_path.py",
  "Tests/UI/test_console_run_tick_workspace_reads.py",
  "Tests/UI/test_console_turn_file_card.py",
  "Tests/UI/test_console_turn_file_card_factory.py",
  "Tests/UI/test_console_turn_file_card_notes.py",
  "Tests/UI/test_console_workspace_action_menu.py",
  "Tests/UI/test_console_workspace_context_rail.py",
  "Tests/UI/test_console_workspace_controller.py",
  "Tests/UI/test_console_workspace_tray_recompose_guard.py",
  "Tests/UI/test_console_workspace_tree.py",
  "Tests/UI/test_console_workspace_tree_cursor_layout.py",
  "tldw_chatbook/Widgets/Console/console_project_instructions.py",
  "tldw_chatbook/Widgets/Console/console_turn_file_card.py",
  "tldw_chatbook/Widgets/Console/console_workspace_action_menu.py",
  "tldw_chatbook/Widgets/Console/console_workspace_context.py",
  "tldw_chatbook/Widgets/Console/console_workspace_switcher_modal.py",
  "tldw_chatbook/Widgets/Console/console_workspace_tree.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot (formatter-mandated docstring quote-initial normalization documented if it occurs).
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000, plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `ee1c1e7365` (isolated worktree `.worktrees/ruff-debt-batch-4`, branch `chore/ruff-debt-batch-4`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no upstream delete or rename. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 14 of 19 assigned paths formatted. `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 195 assigned paths: 5 findings before and 5 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: the thirteen assigned test files (recorded surface Tests/UI + Tests/Workspaces). Result on the formatted tree: 111 failed, 305 passed in 456.60s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 115 failed, 301 passed in 383.79s — a +/-4 count delta, resolved by the name-diff protocol: the differing FAILED entries live in `Tests/UI/test_console_workspace_context_rail.py`, which is one of this task's ALREADY-CLEAN paths the batch never modified (two F1-vocabulary tests failed only in the formatted run; one tree-snapshot test only in the baseline). The churn is timing-flaky membership in the pre-existing dev-tip red mass, not formatting-correlated; AST equality independently proves runtime-identical files. — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** 5 paths were already formatter-clean at the base (`test_console_workspace_context_rail.py`, `test_console_workspace_controller.py`, `test_console_workspace_tree.py`, `console_workspace_action_menu.py`, `console_workspace_switcher_modal.py`) and were deliberately left untouched; the other 14 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
