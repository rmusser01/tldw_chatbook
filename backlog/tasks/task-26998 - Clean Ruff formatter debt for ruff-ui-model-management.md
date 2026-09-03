---
id: TASK-26998
title: Clean Ruff formatter debt for ruff-ui-model-management
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

<!-- TASK-26000-BATCH: ruff-ui-model-management -->
<!-- TASK-26000-PATHS-SHA256: 9f47a66d655cd399e8ea2d3f45b4d5b94b6fa8b2c173588302b97f8476711001 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-ui-model-management` Ruff formatter batch at the owner boundary recorded as: Model installation, catalog, provider-resolution, and local-model UI with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Model_Artifacts", "Tests/UI"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/UI/test_llm_gguf_source_modes.py",
  "Tests/UI/test_llm_ollama_tick_gate.py",
  "Tests/UI/test_llm_screen_ollama_probe_nonblocking.py",
  "Tests/UI/test_llm_screen_ollama_ux_unchanged.py",
  "Tests/UI/test_model_artifact_widgets.py",
  "Tests/UI/test_model_catalog_consent_modal.py",
  "Tests/UI/test_model_installed_view.py",
  "Tests/UI/test_parakeet_v2_install_ui.py",
  "Tests/UI/test_provider_model_resolution.py",
  "tldw_chatbook/UI/Screens/model_installed_view.py",
  "tldw_chatbook/UI/Screens/provider_model_resolution.py"
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
