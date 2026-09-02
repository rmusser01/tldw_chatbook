---
id: TASK-26991
title: Clean Ruff formatter debt for ruff-state-sync-wizards-tests
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

<!-- TASK-26000-BATCH: ruff-state-sync-wizards-tests -->
<!-- TASK-26000-PATHS-SHA256: c9d72a418572dc9ee8d7c1f6cc15792afe991bd916f2499444862d022ddc22b8 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-state-sync-wizards-tests` Ruff formatter batch at the owner boundary recorded as: State, sync-interoperability, event-handler, and wizard integration tests.. The focused test surface recorded by TASK-26000 is `["Tests/Event_Handlers", "Tests/State", "Tests/Sync_Interop", "Tests/Wizards"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Event_Handlers/test_note_ingest_import_offload.py",
  "Tests/State/test_screen_state_store.py",
  "Tests/Sync_Interop/test_chat_outbox_producer.py",
  "Tests/Sync_Interop/test_note_organization_receipt_finalization.py",
  "Tests/Sync_Interop/test_notes_organization_adapters.py",
  "Tests/Sync_Interop/test_notes_organization_app_wiring.py",
  "Tests/Sync_Interop/test_notes_organization_contract.py",
  "Tests/Sync_Interop/test_notes_organization_enrollment.py",
  "Tests/Sync_Interop/test_notes_organization_intent_dispatch.py",
  "Tests/Sync_Interop/test_notes_organization_legacy_inventory.py",
  "Tests/Sync_Interop/test_notes_organization_two_device.py",
  "Tests/Sync_Interop/test_notes_outbox_producer.py",
  "Tests/Wizards/test_first_run_setup_integration.py",
  "Tests/Wizards/test_first_run_setup_state.py",
  "Tests/Wizards/test_first_run_setup_wizard.py",
  "Tests/Wizards/test_first_run_speech_step_state.py"
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
