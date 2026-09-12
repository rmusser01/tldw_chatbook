---
id: TASK-26999
title: Clean Ruff formatter debt for ruff-ui-navigation-shell
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

<!-- TASK-26000-BATCH: ruff-ui-navigation-shell -->
<!-- TASK-26000-PATHS-SHA256: 873c1da1a5576dd80265f74fd8a4500159017d7ce30315e3c11b96fc5f7861f9 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-navigation-shell` Ruff formatter batch at the owner boundary recorded as: Application navigation, destination shells, footer, command palette, and startup shell tests.. The focused test surface recorded by TASK-26000 is `["Tests/App", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_app_footer_shortcut_context.py",
  "Tests/UI/test_command_palette_providers.py",
  "Tests/UI/test_command_palette_shell_routes.py",
  "Tests/UI/test_destination_headers.py",
  "Tests/UI/test_destination_shells.py",
  "Tests/UI/test_destination_visual_parity_correction.py",
  "Tests/UI/test_master_shell_design_system_contract.py",
  "Tests/UI/test_master_shell_navigation.py",
  "Tests/UI/test_product_maturity_phase1_navigation_smoke.py",
  "Tests/UI/test_screen_footer_hints.py",
  "Tests/UI/test_screen_navigation.py",
  "Tests/UI/test_screen_navigation_failure_recovery.py",
  "Tests/UI/test_splash_initial_screen_preimport.py",
  "tldw_chatbook/UI/Navigation/main_navigation.py",
  "tldw_chatbook/UI/Navigation/screen_registry.py"
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
