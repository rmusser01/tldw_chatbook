---
id: TASK-27007
title: Clean Ruff formatter debt for ruff-ui-visual-css
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

<!-- TASK-26000-BATCH: ruff-ui-visual-css -->
<!-- TASK-26000-PATHS-SHA256: 9247d991c715757685b8c0cddbb88d3610b5fd4a0d4f17bc3f0961bea3114772 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-visual-css` Ruff formatter batch at the owner boundary recorded as: Visual, CSS, focus, layout, rendering, and responsive UI probes.. The focused test surface recorded by TASK-26000 is `["Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_bulk_selection_tooltips.py",
  "Tests/UI/test_bundle_rendering.py",
  "Tests/UI/test_checkbox_height_render.py",
  "Tests/UI/test_compact_focus_outline_render.py",
  "Tests/UI/test_consolidated_css_harness.py",
  "Tests/UI/test_css_parse_cache_modal_probe.py",
  "Tests/UI/test_datatable_focus_outline_click.py",
  "Tests/UI/test_focus_token_parity.py",
  "Tests/UI/test_non_obscuring_focus_contract.py",
  "Tests/UI/test_product_maturity_phase1_visual_audit.py",
  "Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py",
  "Tests/UI/test_timer_path_layout_cost.py",
  "Tests/UI/test_trace_responsive.py",
  "Tests/UI/test_ui_responsiveness.py",
  "Tests/UI/test_widget_css_consolidation.py",
  "Tests/UI/test_workbench_focus_help.py",
  "Tests/UI/test_workbench_visual_snapshots.py"
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
