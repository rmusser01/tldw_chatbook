---
id: TASK-26965
title: Clean Ruff formatter debt for ruff-console-layout-rails
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

<!-- TASK-26000-BATCH: ruff-console-layout-rails -->
<!-- TASK-26000-PATHS-SHA256: 28d6f93f221c05b9f946df179a97f753b97a44ee8d592b165531a0d20ff885eb -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-console-layout-rails` Ruff formatter batch at the owner boundary recorded as: Console rails, layout, resize, geometry, and chip surfaces.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_console_avatar_geometry_offloop.py",
  "Tests/UI/test_console_chip_focus_contract.py",
  "Tests/UI/test_console_chip_strip_overflow.py",
  "Tests/UI/test_console_context_rail_header.py",
  "Tests/UI/test_console_context_rail_keyboard.py",
  "Tests/UI/test_console_context_rail_vocabulary.py",
  "Tests/UI/test_console_cost_chip_screen.py",
  "Tests/UI/test_console_left_rail.py",
  "Tests/UI/test_console_left_rail_focus_walk.py",
  "Tests/UI/test_console_model_apply_chips.py",
  "Tests/UI/test_console_narrow_layout.py",
  "Tests/UI/test_console_rail_reconciliation.py",
  "Tests/UI/test_console_rail_reflow_hover_budget.py",
  "Tests/UI/test_console_resize_reflow.py",
  "Tests/UI/test_console_right_rail.py",
  "Tests/UI/test_console_shell_chip_actions.py",
  "Tests/UI/test_console_staged_evidence_strip.py",
  "Tests/UI/test_console_tab_strip_budget.py",
  "Tests/UI/test_console_voice_chip.py",
  "tldw_chatbook/UI/Console_Modules/left_rail.py",
  "tldw_chatbook/UI/Console_Modules/right_rail.py"
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
