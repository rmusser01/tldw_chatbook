---
id: TASK-26994
title: Clean Ruff formatter debt for ruff-ui-evals
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
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

<!-- TASK-26000-BATCH: ruff-ui-evals -->
<!-- TASK-26000-PATHS-SHA256: 4d152bd781bafc9274100582f425ec2e07d6af589615980bb57be57d18f0a613 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-evals` Ruff formatter batch at the owner boundary recorded as: Evaluation UI screens and directly named UI tests.. The focused test surface recorded by TASK-26000 is `["Tests/Evals", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_eval_file_picker_dialog.py",
  "Tests/UI/test_evals_authoring_e2e.py",
  "Tests/UI/test_evals_bench_editor.py",
  "Tests/UI/test_evals_cell_continuation_e2e.py",
  "Tests/UI/test_evals_character_bench_editor.py",
  "Tests/UI/test_evals_character_run_e2e.py",
  "Tests/UI/test_evals_continuation_e2e.py",
  "Tests/UI/test_evals_deletion_guard.py",
  "Tests/UI/test_evals_empty_states.py",
  "Tests/UI/test_evals_results_grid.py",
  "Tests/UI/test_evals_screen.py",
  "Tests/UI/test_evals_selection_scoped_regions.py",
  "Tests/UI/test_evals_snippet_editor.py",
  "Tests/UI/test_evals_steering_e2e.py",
  "tldw_chatbook/UI/Evals/bench_editor.py",
  "tldw_chatbook/UI/Evals/card_picker.py",
  "tldw_chatbook/UI/Evals/character_bench_editor.py",
  "tldw_chatbook/UI/Evals/evals_state.py",
  "tldw_chatbook/UI/Evals/inspector.py",
  "tldw_chatbook/UI/Evals/library_rail.py",
  "tldw_chatbook/UI/Evals/results_grid.py",
  "tldw_chatbook/UI/Evals/sample_bench.py",
  "tldw_chatbook/UI/Evals/snippet_editor.py",
  "tldw_chatbook/UI/Screens/evals_screen.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
