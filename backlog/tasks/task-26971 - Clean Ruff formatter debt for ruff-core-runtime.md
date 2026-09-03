---
id: TASK-26971
title: Clean Ruff formatter debt for ruff-core-runtime
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

<!-- TASK-26000-BATCH: ruff-core-runtime -->
<!-- TASK-26000-PATHS-SHA256: 808122cb7aa8c75ba86515f20f66e16879ce4d565d8179c7ce904bd609a1e154 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-core-runtime` Ruff formatter batch at the owner boundary recorded as: Cross-cutting package runtime modules outside narrower subsystem ownership.. The focused test surface recorded by TASK-26000 is `["Tests"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "tldw_chatbook/Constants.py",
  "tldw_chatbook/Event_Handlers/note_ingest_events.py",
  "tldw_chatbook/Study_Interop/local_study_service.py",
  "tldw_chatbook/Study_Interop/server_study_service.py",
  "tldw_chatbook/Study_Interop/study_scope_service.py",
  "tldw_chatbook/Sync_Interop/__init__.py",
  "tldw_chatbook/Sync_Interop/domain_adapters/__init__.py",
  "tldw_chatbook/Sync_Interop/notes_organization_inventory.py",
  "tldw_chatbook/Sync_Interop/notes_organization_sync_service.py",
  "tldw_chatbook/Third_Party/textual_fspicker/base_dialog.py",
  "tldw_chatbook/Third_Party/textual_fspicker/parts/directory_navigation.py",
  "tldw_chatbook/model_capabilities.py"
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
